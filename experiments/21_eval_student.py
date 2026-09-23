"""학생 평가 — 어댑터(또는 zero-shot 베이스)로 iBims1 test / NYUv2 test / ETH3D 부분집합에서 숫자 디코딩 + fd 트리.
지표: δ₁·AbsRel, 학생 CoV 의 AUSE/AUC, 픽셀당 시간·VRAM. 출력: outputs/distill/eval_<tag>.parquet + 콘솔."""
from __future__ import annotations
import argparse, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch, yaml
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, DynamicCache
from peft import PeftModel
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depthlm_uncertainty.depthlm_data import DepthLMJsonl, draw_marker
from depthlm_uncertainty.number_distribution import enumerate_number_distribution, _is_numeric_token
from depthlm_uncertainty.uq_metrics import compute_aucs
from depthlm_uncertainty.metrics import _FLOAT_RE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); MODEL = os.environ.get("STUDENT_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
PROMPT = ("The red arrow in the image points at a specific location. Estimate the distance from the camera to that location in meters. "
          "Answer with only a number, for example 2.35.")
def resolve(p): p = os.path.expandvars(os.path.expanduser(p)); return p if os.path.isabs(p) else os.path.join(ROOT, p)

@torch.no_grad()
def decode(model, tok, inputs, max_steps=8):
    dev = inputs["input_ids"].device; ids = inputs["input_ids"]; P = ids.shape[1]
    extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=DynamicCache(), use_cache=True, logits_to_keep=1, **extra); cache, logits = out.past_key_values, out.logits[0, -1]; s = ""; n = 0
    rope_deltas = getattr(getattr(model, "model", None), "rope_deltas", None)   # Qwen2.5-VL mrope: number_distribution 과 동일하게 위치를 명시
    while n < max_steps:
        tid = int(torch.argmax(logits)); t = tok.decode([tid], skip_special_tokens=False)
        if not _is_numeric_token(t): break
        s += t; kw = {}
        if rope_deltas is not None: kw["position_ids"] = torch.tensor([P + n], device=dev).view(1, 1, 1).expand(3, 1, 1) + rope_deltas.to(dev).view(1, 1, 1)
        n += 1; out = model(input_ids=torch.tensor([[tid]], device=dev), attention_mask=torch.ones(1, P + n, device=dev, dtype=torch.long), past_key_values=cache, use_cache=True, logits_to_keep=1, **kw); cache, logits = out.past_key_values, out.logits[0, -1]
    m = _FLOAT_RE.search(s); return float(m.group(0)) if m else np.nan

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", default=""); ap.add_argument("--tag", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit_img", type=int, default=0, help="스모크용: 데이터셋당 이미지 수"); ap.add_argument("--per_max", type=int, default=0, help="스모크용: 이미지당 픽셀 상한"); ap.add_argument("--decimals", type=int, default=1, choices=[1, 2]); ap.add_argument("--focal", type=float, default=750.0); ap.add_argument("--eval_set", default="small", choices=["small", "large"], help="small = ref/tree_px (300/320/302), large = ref/dist_* 전체 (3,000/2,000/4,503)"); args = ap.parse_args()
    dev = args.device; is_cuda = dev.startswith("cuda")
    proc = AutoProcessor.from_pretrained(MODEL); tok = proc.tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=dev)
    if args.adapter: model = PeftModel.from_pretrained(model, os.path.expanduser(args.adapter)).merge_and_unload()
    model.eval(); rows = []
    for name in ["ibims1", "nyuv2", "eth3d"]:
        # 교사 표(고속 복호·저해상도·캐스케이드)와 같은 픽셀 집합: outputs/tree_px_<name>.parquet (iBims1 300 / NYUv2 320 / ETH3D 303 px)
        cfg = yaml.safe_load(open(resolve(f"configs/{name}.yaml"))); ds = DepthLMJsonl(resolve(cfg["jsonl"]), os.path.expandvars(cfg["image_folder"]), name, normalized_focal_length=args.focal)
        px = (pd.read_parquet(resolve(f"ref/tree_px_{name}.parquet")) if args.eval_set == "small" else pd.read_parquet(resolve(f"ref/dist_{name}.parquet")).query("dist_ok == True"))[["image_id", "pixel_index"]]; idx = {ds.image_id(i): i for i in range(len(ds))}
        if args.limit_img: px = px[px.image_id.isin(sorted(px.image_id.unique())[:args.limit_img])]
        if args.per_max: px = px.groupby("image_id").head(args.per_max)
        chunk = 4 if name == "eth3d" else 8; t0 = time.time(); n = 0
        for r in px.itertuples():
            i = idx[r.image_id]; j = int(r.pixel_index); s = ds.get_sample(i, j)
            if s is None: continue
            im = draw_marker(ds.get_context(i).base_image, *s.pixel_xy)
            msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": PROMPT}]}]
            enc = proc.apply_chat_template([msgs], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
            inputs = {k: (torch.as_tensor(v).to(dev, dtype=torch.bfloat16) if k == "pixel_values" else torch.as_tensor(v).to(dev, dtype=torch.long)) for k, v in enc.items()}
            is_cuda and torch.cuda.synchronize(); t1 = time.time(); val = decode(model, tok, inputs); is_cuda and torch.cuda.synchronize(); t_dec = time.time() - t1
            nd = enumerate_number_distribution(model, tok, inputs, torch.tensor([], dtype=torch.long), min_branch_p=0.005, chunk=chunk, stop_first_decimal=(args.decimals == 1), max_depth=6)
            rows.append({"dataset": name, "image_id": s.image_id, "pixel_index": j, "gt": s.depth_gt, "pred": val, "pred_mid": (val + 0.5 * 10 ** (-args.decimals)) if val == val else val, "cov": nd.stats()["dist_cov"], "ev": nd.expected_value(), "mass": nd.covered_mass, "sec_decode": t_dec}); n += 1; del inputs
        print(f"{name}: {n} px, {(time.time()-t0)/max(n,1):.2f} s/px", flush=True)
    d = pd.DataFrame(rows); OUT = os.environ.get("OUT_ROOT", "results"); os.makedirs(resolve(f"{OUT}/eval"), exist_ok=True); d.to_parquet(resolve(f"{OUT}/eval/eval_{args.tag}{'_large' if args.eval_set == 'large' else ''}.parquet"), index=False)
    for name, g in d.groupby("dataset"):
        ok = g[g.pred.notna() & (g.pred > 0)]; gt = ok["gt"].values; p = ok.pred_mid.values; err = np.abs(p - gt) / gt; fail = (np.maximum(p / gt, gt / p) >= 1.25).astype(int)
        a = compute_aucs(gt, p, ok["cov"].values, metrics=("abs_rel",))["abs_rel"]["ause"]; auc = roc_auc_score(fail, ok["cov"].values) if fail.min() != fail.max() else np.nan
        print(f"[{args.tag}] {name}: n={len(ok)} 파싱 {len(ok)/len(g)*100:.1f}%  δ₁ {np.mean(fail == 0):.3f}  AbsRel {err.mean():.3f}  | CoV AUSE {a:.4f} AUC {auc:.3f} | decode {ok.sec_decode.mean():.3f} s/px  VRAM {(torch.cuda.max_memory_allocated()/1e9 if is_cuda else 0):.1f} GB", flush=True)

if __name__ == "__main__":
    main()
