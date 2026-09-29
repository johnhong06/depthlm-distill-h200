"""풀 순서에서 거의 같은 이미지를 맨 뒤로 보낸다 (풀 v5, 2026-09-29).

DepthLM 은 "아주 비슷한 영상 프레임은 도움이 안 된다"며 일부만 썼다(원문 부록 Table 4). 우리 주행 풀은 KITTI 2012·2015 의 장면당
두 프레임(_10, _11, 0.1 초 간격)과 정차 중 raw 프레임 때문에 N=1600 의 30 % 가 앞 이미지와 거의 같았다. 그러면 "이미지를 많이" 쪽
셀이 명목상 N 보다 적은 이미지로 비교된다.

규칙: 지금 순서대로 훑으며, 이미 남긴 이미지 중 dHash(9×8) 해밍 ≤ 10 이고 32×32 회색조 상관 > 0.9 인 것이 있으면 뒤로 보낸다.
새 순서 = 남긴 이미지(원래 상대 순서) + 뒤로 보낸 이미지(원래 상대 순서). 이미지 집합·픽셀 좌표·도메인은 그대로라 기존 라벨이
유효하고, 풀 전체를 쓰는 N=6400 셀은 이미지 집합이 바뀌지 않는다. 바뀐 순서에서 새로 필요한 라벨(앞 1,600 장에 새로 들어온
이미지의 픽셀 4..15)은 run.sh label 이 todo_label.parquet 와 기존 라벨의 차이만 라벨링한다.
출력(제자리): pools/<pool>/pool.jsonl, rows_N*_k*.parquet, todo_label.parquet, arms.json, dedup_report.md
사용: python experiments/07_dedup_pool_order.py --pool outdoor --data_root $DATA_ROOT [--dry]
"""
import argparse, json, os
import numpy as np, pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def feats(path):
    im = Image.open(path).convert("L")
    g = np.asarray(im.resize((9, 8), Image.BILINEAR), np.int16); h = np.packbits((g[:, 1:] > g[:, :-1]).flatten())
    v = np.asarray(im.resize((32, 32), Image.BILINEAR), np.float32).flatten(); return h, (v - v.mean()) / (v.std() + 1e-6)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True); ap.add_argument("--data_root", required=True); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--ham", type=int, default=10); ap.add_argument("--corr", type=float, default=0.9)
    a = ap.parse_args(); P = os.path.join(ROOT, "pools", a.pool)
    recs = [json.loads(l) for l in open(f"{P}/pool.jsonl")]; arms = json.load(open(f"{P}/arms.json"))
    cells = [(c["N"], c["k"]) for c in arms["cells"]]
    F = [feats(os.path.join(a.data_root, r["image"])) for r in recs]; H = np.stack([f[0] for f in F]); V = np.stack([f[1] for f in F])

    # 도메인마다 원래 순서대로 훑어 중복을 그 도메인의 뒤로 보내고, 위치 i 는 원래 위치 i 의 도메인의 다음 이미지로 채운다
    # → 어느 N 에서든 도메인 비율이 원래와 같다 (도메인이 하나인 풀은 "남긴 것 + 뒤로 보낸 것" 과 같다)
    dom = [r["domain"] for r in recs]; queue, kept, moved, match = {}, [], [], {}
    for d in dict.fromkeys(dom):
        kd, md = [], []
        for i in (i for i in range(len(recs)) if dom[i] == d):
            if kd:
                K = np.array(kd); ham = np.unpackbits(H[K] ^ H[i], axis=1).sum(1); cand = K[ham <= a.ham]
                c = V[cand] @ V[i] / V.shape[1] if len(cand) else np.array([])
                if len(c) and c.max() > a.corr: md.append(i); match[i] = int(cand[c.argmax()]); continue
            kd.append(i)
        queue[d] = iter(kd + md); kept += kd; moved += md
    new = [next(queue[d]) for d in dom]; nrec = [recs[i] for i in new]; moved.sort()
    print(f"{a.pool}: {len(recs)}장 중 같은 도메인의 앞 이미지와 거의 같은 {len(moved)}장을 도메인 안에서 뒤로 → 서로 다른 이미지 {len(kept)}장", flush=True)

    rows = {(N, k): pd.DataFrame([(r["image"], j) for r in nrec[:N] for j in range(k)], columns=["image_id", "pixel_index"]) for N, k in cells}
    todo = pd.concat(rows.values()).drop_duplicates().reset_index(drop=True)
    lab = f"{P}/teacher_labels.parquet"
    have = pd.read_parquet(lab)[["image_id", "pixel_index"]] if os.path.exists(lab) else todo.iloc[:0]
    need = todo.merge(have, on=["image_id", "pixel_index"], how="left", indicator=True); need = need[need._merge == "left_only"]

    src = pd.Series([r["source"] for r in recs]); nsrc = pd.Series([r["source"] for r in nrec])
    def distinct(order_idx, N):   # 앞 N 장 안에서 앞 이미지와 거의 같지 않은 이미지 수 (위와 같은 기준, 순서 기준)
        ks = []
        for i in order_idx[:N]:
            if ks:
                K = np.array(ks); ham = np.unpackbits(H[K] ^ H[i], axis=1).sum(1); cand = K[ham <= a.ham]
                if len(cand) and (V[cand] @ V[i] / V.shape[1]).max() > a.corr: continue
            ks.append(i)
        return len(ks)
    Ns = sorted({N for N, _ in cells})
    md = [f"# 풀 v5: 거의 같은 이미지를 순서 뒤로 ({a.pool}, 2026-09-29)", "",
          f"- 규칙: dHash 해밍 ≤ {a.ham} 이고 32×32 회색조 상관 > {a.corr} 인 이미지가 이미 남긴 이미지 중에 있으면 뒤로 (`experiments/07_dedup_pool_order.py`)",
          f"- 뒤로 보낸 이미지 {len(moved)}장, 소스별 {src.iloc[moved].value_counts().to_dict()}",
          f"- 서로 다른 이미지 수 (앞 N 장 안): 전 {{{', '.join(f'N{N}: {distinct(list(range(len(recs))), N)}' for N in Ns)}}} → 후 {{{', '.join(f'N{N}: {distinct(new, N)}' for N in Ns)}}}",
          f"- 장면 라벨 수 (앞 N 장): 전 {{{', '.join(f'N{N}: {len({r['scene'] for r in recs[:N]})}' for N in Ns)}}} → 후 {{{', '.join(f'N{N}: {len({r['scene'] for r in nrec[:N]})}' for N in Ns)}}}",
          f"- 도메인 비율 (앞 N 장, 전 = 후): {{{', '.join(f'N{N}: { {k: int(v) for k, v in pd.Series([r["domain"] for r in nrec[:N]]).value_counts().items()} }' for N in Ns)}}}",
          f"- 소스 구성 (앞 1600 장): 전 {src.iloc[:1600].value_counts().to_dict()} → 후 {nsrc.iloc[:1600].value_counts().to_dict()}",
          f"- 필요한 라벨 {len(todo)} px 중 기존 라벨에 없는 {len(need)} px → `bash run.sh label {a.pool}` 가 이것만 라벨링한다",
          "", "뒤로 보낸 이미지와 짝(처음 20개):", ""] + [f"- `{recs[i]['image']}` ≈ `{recs[match[i]]['image']}`" for i in moved[:20]]
    print("\n".join(md[:9]), flush=True)
    if a.dry: return
    with open(f"{P}/pool.jsonl", "w") as f:
        for r in nrec: f.write(json.dumps(r) + "\n")
    for (N, k), df in rows.items(): df.to_parquet(f"{P}/rows_N{N}_k{k}.parquet", index=False)
    todo.to_parquet(f"{P}/todo_label.parquet", index=False)
    arms.update({"version": "v5", "dedup": {"ham": a.ham, "corr": a.corr, "moved": len(moved)}, "n_label_px": int(len(todo))})
    json.dump(arms, open(f"{P}/arms.json", "w"), indent=1, ensure_ascii=False)
    open(f"{P}/dedup_report.md", "w").write("\n".join(md) + "\n"); print(f"저장 → {P}", flush=True)

if __name__ == "__main__":
    main()
