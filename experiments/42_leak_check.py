"""평가 세트 ↔ 풀 이미지 중복 검사 (dHash 64비트, 해밍 거리 ≤ 6 이면 후보로 보고).
풀 이미지는 두 풀(outdoor, mixed)의 교사 라벨에 나오는 KITTI 이미지 전부. 후보는 32×32 회색조 상관으로 한 번 더 본다 (> 0.9 = 실제 중복).
사용: python experiments/42_leak_check.py --eval_jsonl ~/data/drive_eval/kitti_ho/kitti_ho_val.jsonl --data_root $DATA_ROOT
"""
import argparse, json, os
import numpy as np, pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def dhash(im):
    g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), np.int16); b = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if x else "0" for x in b), 2)

def small(im): g = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), np.float32).flatten(); return (g - g.mean()) / (g.std() + 1e-6)

ap = argparse.ArgumentParser(); ap.add_argument("--eval_jsonl", required=True); ap.add_argument("--data_root", required=True); ap.add_argument("--thr", type=int, default=6)
a = ap.parse_args(); ev_dir = os.path.dirname(os.path.expanduser(a.eval_jsonl))
ev = [json.loads(l)["image"] for l in open(os.path.expanduser(a.eval_jsonl))]
pool = sorted(set().union(*[set(pd.read_parquet(os.path.join(ROOT, f"pools/{p}/teacher_labels.parquet")).query("source.str.startswith('kitti')", engine="python").image_id) for p in ("outdoor", "mixed")]))
print(f"평가 {len(ev)}장 vs 풀 KITTI {len(pool)}장", flush=True)
ph = np.array([dhash(Image.open(os.path.join(a.data_root, p))) for p in pool], dtype=np.uint64)
hits = []
for e in ev:
    im = Image.open(os.path.join(ev_dir, e)); h = np.uint64(dhash(im))
    d = np.array([bin(int(x)).count("1") for x in np.bitwise_xor(ph, h)])
    for j in np.nonzero(d <= a.thr)[0]:
        c = float(np.dot(small(im), small(Image.open(os.path.join(a.data_root, pool[j])))) / 1024)
        hits.append((e, pool[j], int(d[j]), round(c, 3)))
print(f"해밍 ≤ {a.thr} 후보 {len(hits)}쌍, 상관 > 0.9 {sum(h[3] > 0.9 for h in hits)}쌍")
for h in sorted(hits, key=lambda x: -x[3])[:20]: print("  ", *h)
