"""학생(Qwen2.5-VL-3B) LoRA 증류 — 4조건 동등 비교 (D-68/69).
  A hard : CE(교사 greedy1 문자열 + EOS)
  B soft : 교사 greedy 경로 위 각 접두사에서 numeric 토큰+종료 분포와의 KL (teacher forcing), 마지막(소수 첫째 자리 뒤)은 EOS 로 CE
  C wsoft: B × w_i, w_i = clip(1 − cov_i/0.3, 0.2, 1) 을 평균 1 로 정규화 (사전 고정)
  D gt   : CE(GT 소수 첫째 자리 + EOS), GT 있는 행만 (NYUv2)
공통: 같은 행 순서·시드·LoRA(r16, q/k/v/o)·lr 1e-4 cosine·grad accumulation 8·epoch 수. 학생 입력은 DepthLM 마커 이미지 f'=500 + 고정 프롬프트, 출력은 숫자(소수 첫째 자리)+EOS.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch, yaml
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depthlm_uncertainty.depthlm_data import DepthLMJsonl, draw_marker
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); MODEL = os.environ.get("STUDENT_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
PROMPT = ("The red arrow in the image points at a specific location. Estimate the distance from the camera to that location in meters. "
          "Answer with only a number, for example 2.35.")
STUDENT_FOCAL = 750.0   # 학생 입력 정규화 초점거리. 교사(750)와 동일 = 해상도 교란 제거 (2026-09-22 사용자 지적). --focal 로 변경 가능
DTYPE = torch.bfloat16
def resolve(p): p = os.path.expandvars(os.path.expanduser(p)); return p if os.path.isabs(p) else os.path.join(ROOT, p)

class Data:
    def __init__(self, cond, limit=0, seed=0, labels=("pools/mixed/teacher_labels.parquet",), pools=("configs/pool_mixed.yaml",)):
        d = pd.concat([pd.read_parquet(resolve(x)) for x in labels], ignore_index=True); d = d[d.teacher_greedy1.notna()]
        if cond == "gt": d = d[d["gt"] > 0]
        self.rows = d.sample(frac=1.0, random_state=seed).reset_index(drop=True)   # 같은 seed → A/B/C 동일 순서
        if limit: self.rows = self.rows.head(limit)
        self.dss, self.idx = [], {}
        for pc in pools:   # 여러 풀: image_id 로 어느 풀인지 찾는다
            cfg = yaml.safe_load(open(resolve(pc))); ds = DepthLMJsonl(resolve(cfg["jsonl"]), os.path.expandvars(cfg["image_folder"]), "pool", normalized_focal_length=STUDENT_FOCAL)
            k = len(self.dss); self.dss.append(ds); self.idx.update({ds.image_id(i): (k, i) for i in range(len(ds))})
        if cond == "wsoft":
            w = np.clip(1.0 - self.rows.teacher_cov.values / 0.3, 0.2, 1.0); self.w = w / w.mean()
        else: self.w = np.ones(len(self.rows))
    def image(self, r):
        k, i = self.idx[r.image_id]; ds = self.dss[k]
        s = ds.get_sample(i, int(r.pixel_index)); return None if s is None else draw_marker(ds.get_context(i).base_image, *s.pixel_xy)

def build(proc, im, answer, dev):
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": PROMPT}]}]
    pre = proc.apply_chat_template([msgs], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"); P = pre["input_ids"].shape[1]
    ans_ids = proc.tokenizer(answer, add_special_tokens=False)["input_ids"] + [proc.tokenizer.convert_tokens_to_ids("<|im_end|>")]
    ids = torch.cat([pre["input_ids"], torch.tensor([ans_ids])], 1); am = torch.ones_like(ids)
    enc = {"input_ids": ids.to(dev), "attention_mask": am.to(dev), "pixel_values": torch.as_tensor(pre["pixel_values"]).to(dev, dtype=DTYPE), "image_grid_thw": torch.as_tensor(pre["image_grid_thw"]).to(dev)}
    return enc, P, ans_ids

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cond", required=True, choices=["hard", "soft", "wsoft", "gt"]); ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0); ap.add_argument("--accum", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", default=os.environ.get("OUT_ROOT", "results") + "/checkpoints"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--fp32", action="store_true", help="CPU 테스트용"); ap.add_argument("--labels", nargs="+", default=["pools/mixed/teacher_labels.parquet"]); ap.add_argument("--pools", nargs="+", default=["configs/pool_mixed.yaml"]); ap.add_argument("--rows", default="", help="(image_id, pixel_index) parquet — 이 행만 학습 (예산·배분 arm)"); ap.add_argument("--tag", default=""); ap.add_argument("--focal", type=float, default=750.0, help="학생 입력 정규화 초점거리 (교사=750)"); ap.add_argument("--decimals", type=int, default=1, choices=[1, 2], help="라벨 소수 자릿수 (2 는 --subset 의 teacher_full 사용)"); ap.add_argument("--subset", default="", help="해상도 ablation: 이 parquet 의 (image_id,pixel_index) 행만 사용")
    args = ap.parse_args(); torch.manual_seed(args.seed); random.seed(args.seed); dev = args.device
    global STUDENT_FOCAL; STUDENT_FOCAL = args.focal; print(f"student focal {STUDENT_FOCAL}", flush=True)
    data = Data(args.cond, args.limit, seed=0, labels=tuple(args.labels), pools=tuple(args.pools))
    if args.rows:     # arm 선택: 행을 제한한 뒤 같은 seed 로 다시 섞어 조건 간 순서 동일
        want = pd.read_parquet(resolve(args.rows))[["image_id", "pixel_index"]].drop_duplicates(); data.rows = data.rows.merge(want, on=["image_id", "pixel_index"]).sample(frac=1.0, random_state=0).reset_index(drop=True)
        data.w = np.ones(len(data.rows)) if args.cond != "wsoft" else (lambda w: w / w.mean())(np.clip(1.0 - data.rows.teacher_cov.values / 0.3, 0.2, 1.0)); print(f"rows {len(data.rows)}", flush=True)
    if args.subset:   # 같은 부분집합·같은 순서에서 라벨 자릿수만 다르게 (D-75)
        sub = pd.read_parquet(resolve(args.subset))[["image_id", "pixel_index", "teacher_full"]]; sub = sub[sub.teacher_full.notna()]
        data.rows = data.rows.merge(sub, on=["image_id", "pixel_index"]).reset_index(drop=True); data.w = np.ones(len(data.rows)); print(f"subset rows {len(data.rows)}", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL); tok = proc.tokenizer
    global DTYPE; DTYPE = torch.float32 if args.fp32 else torch.bfloat16; DT = DTYPE
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, dtype=DT, attn_implementation="sdpa", device_map=dev)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")); model.print_trainable_parameters()
    num_ids = [tok.convert_tokens_to_ids(str(d)) for d in range(10)] + [tok.convert_tokens_to_ids(".")]; eos = tok.convert_tokens_to_ids("<|im_end|>"); sup = torch.tensor(num_ids + [eos], device=dev)
    tokstr = {tok.convert_tokens_to_ids(str(d)): str(d) for d in range(10)}; tokstr[tok.convert_tokens_to_ids(".")] = "."; str2id = {v: k for k, v in tokstr.items()}
    total = int(len(data.rows) * args.epochs) if not args.steps else args.steps; opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, min(100, total // args.accum // 20)), num_training_steps=max(1, total // args.accum))
    out_dir = os.path.join(os.path.expanduser(args.out), args.cond + args.tag); os.makedirs(out_dir, exist_ok=True); log = open(os.path.join(out_dir, "train.log"), "a")
    t0 = time.time(); model.train(); run = 0.0; nrun = 0
    for step in range(total):
        r = data.rows.iloc[step % len(data.rows)]; im = data.image(r)
        if im is None: continue
        if args.cond == 'gt': answer = f"{r.gt:.1f}"
        elif args.decimals == 2: answer = f"{r.teacher_full:.2f}"                       # 교사 숫자 그대로(둘째 자리)
        elif args.subset: answer = f"{np.floor(r.teacher_full * 10) / 10:.1f}"          # 같은 숫자를 첫째 자리에서 절사 (v1 라벨 규칙과 동일)
        else: answer = f"{r.teacher_greedy1:.1f}"
        enc, P, ans_ids = build(proc, im, answer, dev); out = model(**enc); logits = out.logits[0, P - 1: P - 1 + len(ans_ids)].float()   # 각 답 토큰 위치의 예측 로짓
        if args.cond in ("hard", "gt"):
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(ans_ids, device=dev))
        else:
            kd = {k["prefix"]: k for k in json.loads(r.kd_nodes)}; losses = []
            for k, tid in enumerate(ans_ids):
                prefix = answer[:k]
                if prefix in kd and k < len(ans_ids) - 1:
                    node = kd[prefix]; p = torch.zeros(len(sup), device=dev)
                    for tstr, lp in node["children"]:
                        if tstr in str2id: p[num_ids.index(str2id[tstr])] += math.exp(lp)
                    p[-1] += node["end_mass"]; p = p / p.sum().clamp_min(1e-8)
                    logq = torch.log_softmax(logits[k][sup], -1); losses.append(torch.sum(p * (torch.log(p.clamp_min(1e-8)) - logq)))
                else:
                    losses.append(torch.nn.functional.cross_entropy(logits[k][None], torch.tensor([tid], device=dev)))
            loss = torch.stack(losses).mean() * float(data.w[step % len(data.rows)])
        (loss / args.accum).backward(); run += loss.item(); nrun += 1
        if (step + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step(); sch.step(); opt.zero_grad()
        if (step + 1) % 200 == 0 or step + 1 == total:
            msg = f"step {step+1}/{total} loss {run/max(nrun,1):.4f} lr {sch.get_last_lr()[0]:.2e} {(time.time()-t0)/(step+1):.3f}s/step"; print(msg, flush=True); log.write(msg + "\n"); log.flush(); run = 0.0; nrun = 0
        if (step + 1) % 5000 == 0: model.save_pretrained(out_dir)
    model.save_pretrained(out_dir); torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if "lora" in k.lower()}, os.path.join(out_dir, "lora_adapter.pt")); print(f"저장 → {out_dir}  총 {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
