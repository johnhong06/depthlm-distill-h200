"""교사(DepthLM 12B) 라벨링 — 템플릿 강제 prefill + 소수 첫째 자리 트리 (D-66 방식, 픽셀당 ~0.6 s).
저장 (픽셀당): teacher_greedy1 (트리 argmax 경로: 정수부·소수 첫째 자리), ev, map, cov, mass,
              kd_nodes = greedy 경로 위 각 접두사("" , "2", "2.")에서의 자식 numeric 토큰 분포 [(tok, logp)] + 비숫자(종료) 질량 → soft-label KD 용.
GT 는 풀에 있으면 그대로 복사 (-1 = 없음). 재개 가능."""
from __future__ import annotations
import argparse, json, math, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depthlm_uncertainty.depthlm_data import DepthLMJsonl, build_problem_prompt, draw_marker
from depthlm_uncertainty.inference import DepthLMRunner
from depthlm_uncertainty.number_distribution import enumerate_number_distribution, _is_numeric_token
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TEMPLATE = "<think> The point is around "
def resolve(p): p = os.path.expanduser(p); return p if os.path.isabs(p) else os.path.join(ROOT, p)
def atomic(df, path):
    tmp = path + ".tmp"; df.to_parquet(tmp, index=False)
    if os.path.exists(path): os.replace(path, path + ".bak")
    os.replace(tmp, path)

def greedy_path(nodes):
    """기록 노드에서 argmax 경로 (소수 첫째 자리까지). 반환: 문자열, 경로 위 노드 리스트"""
    by = {n["path"]: n for n in nodes}; path = ""; used = []
    while path in by:
        n = by[path]; used.append(n)
        if not n["children"]: break
        tok, lp = max(n["children"], key=lambda c: c[1])
        # 종료 질량이 최대 자식보다 크면 여기서 끝 (숫자가 아닌 토큰이 argmax)
        if (1.0 - n["numeric_mass"]) > math.exp(lp) and path: break
        path += tok
        if "." in path and len(path.split(".", 1)[1]) >= 1: break
    return path, used

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/distill_pool.yaml"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--chunk", type=int, default=8); ap.add_argument("--pool", default="", help="DepthLM jsonl (절대 경로 이미지). 주어지면 --config 무시"); ap.add_argument("--image_folder", default="/"); ap.add_argument("--todo", default="", help="(image_id, pixel_index) parquet — 이 목록만 라벨링"); ap.add_argument("--dry", action="store_true", help="대상 수만 출력하고 종료"); ap.add_argument("--out", default="outputs/distill/teacher_labels.parquet"); args = ap.parse_args()
    if args.pool: cfg = {"jsonl": args.pool, "image_folder": args.image_folder, "dataset": "pool", "pixels_per_image": 10**6, "model_path": os.environ.get("TEACHER_MODEL", "facebook/DepthLM")}
    else: cfg = yaml.safe_load(open(resolve(args.config)))
    name = cfg["dataset"]; ds = DepthLMJsonl(resolve(cfg["jsonl"]), resolve(cfg["image_folder"]), name); problem = build_problem_prompt()
    out_path = resolve(args.out); os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done_df = pd.read_parquet(out_path) if os.path.exists(out_path) else pd.DataFrame(); done = set(zip(done_df.image_id, done_df.pixel_index)) if len(done_df) else set(); rows = done_df.to_dict("records") if len(done_df) else []
    if args.todo:
        idx = {ds.image_id(i): i for i in range(len(ds))}; want = pd.read_parquet(resolve(args.todo))[["image_id", "pixel_index"]].drop_duplicates()
        todo = [(idx[r.image_id], int(r.pixel_index)) for r in want.itertuples() if r.image_id in idx and (r.image_id, int(r.pixel_index)) not in done]
    else: todo = [(i, j) for i in range(len(ds)) for j in range(min(cfg["pixels_per_image"], ds.num_pixels(i))) if (ds.image_id(i), j) not in done]
    if args.limit: todo = todo[:args.limit]
    print(f"{name}: 완료 {len(done)}, 남은 {len(todo)}", flush=True)
    if args.dry:
        s0 = ds.get_sample(*todo[0]); print(f"dry: 첫 대상 {ds.image_id(todo[0][0])} px{todo[0][1]} → 좌표 {s0.pixel_xy} 이미지 {ds.get_context(todo[0][0]).base_image.size}"); return
    if not todo: return
    runner = DepthLMRunner(cfg["model_path"], max_new_tokens=128); model, tok = runner.model, runner.processor.tokenizer
    tmpl = torch.tensor(tok(TEMPLATE, add_special_tokens=False)["input_ids"], dtype=torch.long); t0 = time.time(); n = 0
    for i, j in todo:
        s = ds.get_sample(i, j)
        if s is None: continue
        im = draw_marker(ds.get_context(i).base_image, *s.pixel_xy); inputs = runner._prepare([im], problem); rec = []; chunk = args.chunk; nd = None
        while nd is None:
            oom = False
            try: nd = enumerate_number_distribution(model, tok, inputs, tmpl, min_branch_p=0.005, chunk=chunk, stop_first_decimal=True, record=rec)
            except torch.cuda.OutOfMemoryError: oom = True
            if oom: torch.cuda.empty_cache(); rec = []; chunk = max(1, chunk // 2)
        del inputs
        gstr, used = greedy_path(rec)
        try: gval = float(gstr)
        except ValueError: gval = float("nan")
        kd = [{"prefix": u["path"], "children": u["children"], "end_mass": max(0.0, 1.0 - u["numeric_mass"])} for u in used]
        st = nd.stats(); src = ds.records[i].get("source", "")
        rows.append({"image_id": s.image_id, "pixel_index": j, "source": src, "pixel_x_orig": s.pixel_xy_orig[0], "pixel_y_orig": s.pixel_xy_orig[1], "gt": s.depth_gt,
                     "teacher_greedy1": gval, "teacher_ev": nd.expected_value(), "teacher_map": nd.map_value(), "teacher_cov": st["dist_cov"], "teacher_mass": st["dist_covered"],
                     "kd_nodes": json.dumps(kd), "n_forward": nd.n_forward}); n += 1
        if n % 100 == 0:
            atomic(pd.DataFrame(rows), out_path); el = time.time() - t0; print(f"  {n}/{len(todo)}  {el/n:.2f}s/px  남은 {(len(todo)-n)*el/n/3600:.2f}h  VRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
    atomic(pd.DataFrame(rows), out_path); d = pd.DataFrame(rows)
    print(f"완료 {len(d)} → {out_path}   greedy1 NaN {d.teacher_greedy1.isna().mean()*100:.2f}%  mass median {d.teacher_mass.median():.3f}  {(time.time()-t0)/max(n,1):.2f}s/px", flush=True)

if __name__ == "__main__":
    main()
