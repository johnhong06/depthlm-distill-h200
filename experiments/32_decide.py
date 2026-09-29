"""사전등록 판정 (NOTES D-22, D-24) — 한 풀의 격자 결과에 규칙을 그대로 적용해 표로 쓴다.

비교 (같은 평가 픽셀에서 짝지은 δ1 차이, 군집 부트스트랩 95 % 구간: DDAD·nuScenes 는 장면, 나머지는 사진 단위)
  배분: 같은 예산에서 이미지 많은 셀 − 적은 셀. 1,600: N1600_k1 − N400_k4 / 6,400: N1600_k4 − N400_k16, N6400_k1 − N1600_k4,
        N6400_k1 − N400_k16 / 25,600: N6400_k4 − N1600_k16. soft·hard 각각.
  손실: 같은 셀에서 soft − hard.
판정 (풀의 세트마다 따로 계산한 뒤)
  모든 세트가 같은 방향이고 구간이 모두 0 을 빼면 → 주장 / 같은 방향이고 일부만 → 약한 근거 /
  방향이 갈리고 하나라도 0 을 빼면 → 세트 의존 / 모든 구간이 0 을 포함하면 → 근거 없음 (주장하지 않는 두 경우를 나눠 적은 것).
  혼합 풀은 실내(iBims-1·NYUv2)와 주행(DDAD·nuScenes)에 따로 적용하고, 둘 다 같은 방향으로 주장일 때만 풀 결론.
민감도 (판정에는 쓰지 않음): DDAD 에서 전면이 아닌 카메라의 아래쪽 1/4 픽셀(차체가 보이는 영역)을 뺀 값.
사용: DATA_ROOT=… python experiments/32_decide.py --pool indoor --soft_root <OUT_ROOT> --hard_root <OUT_ROOT> [--out decision.md]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from depthlm_uncertainty.eval_clusters import boot_ci, clusters

NAMES = {"ibims1": "iBims-1", "nyuv2": "NYUv2", "ddad": "DDAD", "nuscenes": "nuScenes"}
GROUPS = {"indoor": {"실내": ["ibims1", "nyuv2"]}, "outdoor": {"주행": ["ddad", "nuscenes"]}, "mixed": {"실내": ["ibims1", "nyuv2"], "주행": ["ddad", "nuscenes"]}}
ALLOC = [(1600, "N1600_k1", "N400_k4"), (6400, "N1600_k4", "N400_k16"), (6400, "N6400_k1", "N1600_k4"), (6400, "N6400_k1", "N400_k16"), (25600, "N6400_k4", "N1600_k16")]

def hit(p, g): p, g = np.asarray(p, float), np.asarray(g, float); return (np.maximum(p / g, g / p) < 1.25).astype(float)

def load(root, cond, tag, ds):   # 예전 한 파일과 데이터셋별 파일 모두
    b = os.path.join(os.path.expanduser(root), "eval", f"eval_{cond}_{tag}_large")
    fs = [f for f in (f"{b}__{ds}.parquet", f"{b}.parquet") if os.path.exists(f)]
    if not fs: return None
    d = pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates(["dataset", "image_id", "pixel_index"]); d = d[(d.dataset == ds) & d.pred.notna() & (d.pred > 0)].copy()
    d["pm"] = d["pred_mid"] if "pred_mid" in d else d["pred"] + 0.05; return d if len(d) else None

def ddad_body_pixels(data_root):   # 전면이 아닌 카메라의 아래쪽 1/4 = 차체가 찍힐 수 있는 영역
    p = os.path.join(os.path.expanduser(data_root), "eval", "ddad", "ddad_val.jsonl")
    if not os.path.exists(p): return set()
    out = set()
    for r in map(json.loads, open(p)):
        H = r["intrinsics"][5]
        if r.get("camera") != "CAMERA_01": out |= {(r["image"], j) for j, (u, v) in enumerate(r["pixel_coords"]) if v > 0.75 * H}
    return out

def paired(a, b, ds, drop=None):
    if a is None or b is None: return None
    m = a.merge(b[["image_id", "pixel_index", "pm"]], on=["image_id", "pixel_index"], suffixes=("", "_b"))
    if drop: m = m[[(i, int(j)) not in drop for i, j in zip(m.image_id, m.pixel_index)]]
    if len(m) < 10: return None
    dd = hit(m.pm, m["gt"]) - hit(m.pm_b, m["gt"]); g = clusters(ds, m.image_id.values); lo, hi = boot_ci(dd, g)
    return float(dd.mean()), lo, hi, len(m), len(np.unique(g))

def verdict(res):   # res: 세트별 (mean, lo, hi) — 하나라도 없으면 미완
    if any(r is None for r in res): return "미완 (세트 결과 부족)", 0
    s = [int(np.sign(r[0])) for r in res]; sig = [r[1] > 0 or r[2] < 0 for r in res]
    if len(set(s)) == 1 and s[0] != 0 and all(sig): return "주장", s[0]
    if len(set(s)) == 1 and s[0] != 0 and any(sig): return "약한 근거", s[0]
    if len(set(s)) > 1 and any(sig): return "세트 의존", 0
    return "근거 없음", 0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True, choices=list(GROUPS)); ap.add_argument("--soft_root", default=os.environ.get("OUT_ROOT", "results"))
    ap.add_argument("--hard_root", default=None); ap.add_argument("--focal", default="750"); ap.add_argument("--out", default=""); ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT", ""))
    a = ap.parse_args(); roots = {"soft": a.soft_root, "hard": a.hard_root or a.soft_root}; groups = GROUPS[a.pool]; sets = [d for g in groups.values() for d in g]
    tag = lambda cell: f"{cell}_{a.pool}_f{a.focal}"; body = ddad_body_pixels(a.data_root) if "ddad" in sets else set()
    D = {(c, cell, ds): load(roots[c], c, tag(cell), ds) for c in roots for cell in {x for _, p, q in ALLOC for x in (p, q)} | {"N400_k1", "N1600_k16", "N6400_k4"} for ds in sets}
    fmt = lambda r: "—" if r is None else f"{r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}]"
    word = {"alloc": {1: "이미지 많은 쪽", -1: "픽셀 많은 쪽"}, "loss": {1: "soft", -1: "hard"}}
    md = [f"# 판정 — {a.pool} 풀 (사전등록 규칙, NOTES D-22·D-24)", "", "Δδ1 [95 % 군집 부트스트랩 구간]. DDAD·nuScenes 는 장면, 나머지는 사진 단위로 다시 뽑는다.", ""]
    rows, summary = [], []
    comps = [("alloc", c, f"{c} 예산 {B}: {p} − {q}", c, p, c, q) for c in ("soft", "hard") for B, p, q in ALLOC] + \
            [("loss", "soft−hard", f"{cell}: soft − hard", "soft", cell, "hard", cell) for cell in ["N400_k1", "N400_k4", "N400_k16", "N1600_k1", "N1600_k4", "N1600_k16", "N6400_k1", "N6400_k4"]]
    for kind, cond, name, c1, t1, c2, t2 in comps:
        res = {ds: paired(D[(c1, t1, ds)], D[(c2, t2, ds)], ds) for ds in sets}
        gv = {g: verdict([res[d] for d in ds]) for g, ds in groups.items()}
        vals = list(gv.values())
        if len(groups) > 1:   # 혼합 풀: 두 도메인 모두 같은 방향으로 주장일 때만 풀 결론
            if any(v[0].startswith("미완") for v in vals): pool_v = ("미완 (세트 결과 부족)", 0)
            elif all(v[0] == "주장" for v in vals) and len({v[1] for v in vals}) == 1: pool_v = ("주장", vals[0][1])
            elif len({(v[0], v[1]) for v in vals}) == 1: pool_v = vals[0]
            else: pool_v = ("도메인별로 다름", 0)
        else: pool_v = vals[0]
        row = {"종류": "배분" if kind == "alloc" else "손실", "비교": name, **{NAMES[d]: fmt(res[d]) for d in sets}}
        if "ddad" in sets and body: row["DDAD (차체 영역 제외)"] = fmt(paired(D[(c1, t1, "ddad")], D[(c2, t2, "ddad")], "ddad", drop=body))
        row.update({f"판정 ({g})": (v[0] + (f": {word[kind][v[1]]}" if v[1] else "")) for g, v in gv.items()})
        if len(groups) > 1: row["풀 판정"] = pool_v[0] + (f": {word[kind][pool_v[1]]}" if pool_v[1] else "")
        rows.append(row)
    df = pd.DataFrame(rows); md += [df.to_markdown(index=False), "", "판정 규칙: 모든 세트가 같은 방향 + 모든 구간이 0 제외 → 주장 / 같은 방향 + 일부만 → 약한 근거 / 방향이 갈리고 하나라도 0 제외 → 세트 의존 / 모두 0 포함 → 근거 없음. "
            "혼합 풀은 실내·주행에 따로 적용해 둘 다 같은 방향으로 주장일 때만 풀 결론. DDAD 차체 영역 제외 값은 민감도로만 본다."]
    out = a.out or os.path.join(os.path.expanduser(roots["soft"]), "tables", f"decision_{a.pool}.md"); os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(md) + "\n"); print(df.to_string(index=False)); print(f"→ {out}")

if __name__ == "__main__":
    main()
