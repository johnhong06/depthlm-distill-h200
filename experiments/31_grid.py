"""격자 분석 — 이미지 수 N × 이미지당 픽셀 k. outputs/eval/eval_soft_N{N}_k{k}[_large].parquet 에서:
  (a) 셀별 δ1 (+ 이미지 클러스터 부트스트랩 95 % CI)
  (b) 같은 행(N 고정, k 증가) = 라벨 밀도의 한계효용
  (c) 같은 열(k 고정, N 증가) = 이미지 수의 한계효용
  (d) 같은 예산(N·k 동일) 셀 간 쌍대 Δδ1 = 고정 예산 배분 비교
출력: paper/table_grid[_large].md, outputs/figures/fig_grid[_large].png"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); NAMES = {"ibims1": "iBims-1", "nyuv2": "NYUv2", "eth3d": "ETH3D (held-out)"}
ap = argparse.ArgumentParser(); ap.add_argument("--cond", default="soft"); ap.add_argument("--eval_set", default="small", choices=["small", "large"]); ap.add_argument("--suffix", default="", help="태그 접미사 (예: _f500)"); ap.add_argument("--arms", default="pools/mixed/arms.json"); args = ap.parse_args()
OUT = os.environ.get("OUT_ROOT", "results")
suf = "_large" if args.eval_set == "large" else ""
cells = json.load(open(os.path.join(ROOT, args.arms)))["cells"]
def hit(p, g): p, g = np.asarray(p, float), np.asarray(g, float); return (np.maximum(p / g, g / p) < 1.25).astype(float)
def boot(v, imgs, n=2000, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(imgs); grp = [v[imgs == i] for i in u]
    return np.percentile([np.concatenate([grp[k] for k in rng.integers(0, len(u), len(u))]).mean() for _ in range(n)], [2.5, 97.5])
def load(tag):
    p = os.path.join(ROOT, f"{OUT}/eval/eval_{args.cond}_{tag}{args.suffix}{suf}.parquet")
    if not os.path.exists(p): return None
    d = pd.read_parquet(p); d = d[d.pred.notna() & (d.pred > 0)].copy(); d["pm"] = d["pred_mid"] if "pred_mid" in d else d["pred"] + 0.05; return d
D = {c["tag"]: load(c["tag"]) for c in cells}; have = [c for c in cells if D[c["tag"]] is not None]
if not have: print("평가 결과 없음"); raise SystemExit
rows = []
for c in have:
    for ds in NAMES:
        g = D[c["tag"]][D[c["tag"]].dataset == ds]
        if not len(g): continue
        h = hit(g.pm, g["gt"]); lo, hi = boot(h, g.image_id.values)
        rows.append({"N": c["N"], "k": c["k"], "budget": c["budget"], "dataset": NAMES[ds], "n": len(g), "δ1": h.mean(), "CI": f"[{lo:.3f}, {hi:.3f}]", "AbsRel": float(np.mean(np.abs(g.pm - g["gt"]) / g["gt"]))})
cell_df = pd.DataFrame(rows)
def paired(t1, t2, ds):
    a, b = D[t1][D[t1].dataset == ds], D[t2][D[t2].dataset == ds]
    m = a.merge(b[["image_id", "pixel_index", "pm"]], on=["image_id", "pixel_index"], suffixes=("", "_b"))
    if len(m) < 10: return None
    dd = hit(m.pm, m["gt"]) - hit(m.pm_b, m["gt"]); lo, hi = boot(dd, m.image_id.values); return dd.mean(), lo, hi, len(m)
comp = []
for ds in NAMES:
    for b in sorted({c["budget"] for c in have}):     # (d) 같은 예산
        cs = [c for c in have if c["budget"] == b]
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                r = paired(cs[j]["tag"], cs[i]["tag"], ds)     # 많은 이미지 − 적은 이미지
                if r: comp.append({"비교": f"예산 {b}: N{cs[j]['N']}k{cs[j]['k']} − N{cs[i]['N']}k{cs[i]['k']}", "종류": "고정 예산 배분", "dataset": NAMES[ds], "Δδ1": f"{r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}]", "n": r[3]})
    for N in sorted({c["N"] for c in have}):           # (b) 같은 행
        ks = sorted(c["k"] for c in have if c["N"] == N)
        for i in range(len(ks) - 1):
            r = paired(f"N{N}_k{ks[i+1]}", f"N{N}_k{ks[i]}", ds)
            if r: comp.append({"비교": f"N={N} 고정: k {ks[i]}→{ks[i+1]}", "종류": "라벨 밀도 한계효용", "dataset": NAMES[ds], "Δδ1": f"{r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}]", "n": r[3]})
    for k in sorted({c["k"] for c in have}):           # (c) 같은 열
        Ns = sorted(c["N"] for c in have if c["k"] == k)
        for i in range(len(Ns) - 1):
            r = paired(f"N{Ns[i+1]}_k{k}", f"N{Ns[i]}_k{k}", ds)
            if r: comp.append({"비교": f"k={k} 고정: N {Ns[i]}→{Ns[i+1]}", "종류": "이미지 수 한계효용", "dataset": NAMES[ds], "Δδ1": f"{r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}]", "n": r[3]})
comp_df = pd.DataFrame(comp)
md = (f"## Grid: images N × pixels-per-image k (loss {args.cond}, eval_set {args.eval_set}, 2 epochs, nested pool)\n\n"
      "Same teacher labels, same pixel indices, same init/shuffle seed; only N and k differ. Budget = N·k. Midpoint decoding. CI = image-cluster bootstrap 95%.\n\n"
      + cell_df.to_markdown(index=False, floatfmt=".3f") + "\n\n### Paired comparisons\n\n" + (comp_df.to_markdown(index=False) if len(comp_df) else "n/a") + "\n")
os.makedirs(os.path.join(ROOT, f"{OUT}/tables"), exist_ok=True); open(os.path.join(ROOT, f"{OUT}/tables/table_grid_{args.cond}{args.suffix}{suf}.md"), "w").write(md); print(cell_df.to_string(index=False)); print(); print(comp_df.to_string(index=False) if len(comp_df) else "")
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6))
cols = {400: "#1f77b4", 1600: "#d62728", 6400: "#2ca02c"}
for a_, ds in zip(ax, NAMES):
    sub = cell_df[cell_df.dataset == NAMES[ds]]
    for N, g in sub.groupby("N"):
        g = g.sort_values("budget"); a_.plot(g.budget, g["δ1"], marker="o", lw=1.6, color=cols.get(N, None), label=f"N={N} images")
        for _, r in g.iterrows(): a_.annotate(f"k={int(r.k)}", (r.budget, r["δ1"]), textcoords="offset points", xytext=(4, -10), fontsize=7, color=cols.get(N, "k"))
    a_.set_xscale("log"); a_.set_xticks(sorted(sub.budget.unique())); a_.set_xticklabels([str(int(x)) for x in sorted(sub.budget.unique())])
    a_.set_title(NAMES[ds], fontsize=10); a_.set_xlabel("teacher-label budget (pixels)"); a_.grid(alpha=0.3)
ax[0].set_ylabel("δ1"); ax[0].legend(fontsize=8)
fig.suptitle("Same budget → vertical gap between lines = allocation effect; along a line = more pixels on the same images", fontsize=9)
os.makedirs(os.path.join(ROOT, f"{OUT}/figures"), exist_ok=True); fig.tight_layout(); fig.savefig(os.path.join(ROOT, f"{OUT}/figures/fig_grid_{args.cond}{args.suffix}{suf}.png"), dpi=150); print("figure saved")
