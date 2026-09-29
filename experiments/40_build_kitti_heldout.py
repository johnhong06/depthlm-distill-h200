"""KITTI held-out 주행 평가 세트 — 풀이 쓰지 않은 주행에서 이미지·질의 픽셀·GT 를 만든다 (평가 설계 v3, "가까운 주행 세트").

제외 주행: 두 풀(outdoor, mixed)이 쓰는 KITTI 주행 전부 + KITTI 2015 학습 영상의 원본 주행(devkit mapping) + 캠퍼스 주행(2011_09_28) 전부.
캠퍼스 주행은 번호가 달라도 같은 장소라 풀의 캠퍼스 주행과 겹친다. KITTI 2012 는 주행 매핑이 없어, 남은 주행의 5 프레임마다 풀 이미지와
비교(dHash 해밍 ≤ 10 후 32×32 상관 > 0.9)하고 겹친 프레임의 앞뒤 50 프레임(±5 초)을 뺀다 = 장소 단위 제외.
GT: data_depth_annotated 의 image_02 (uint16/256 = 정류 카메라 z). 저장값은 기존 평가 세트와 같은 유클리드 거리 (z 는 depth_z).
선정 (고정 시드 0): 남은 도시·주거·도로 주행에서 200 장, 주행당 균등 간격, 상한 = 남은 프레임 수 // 10 (1 초 이상 간격).
이미지당 10 픽셀: GT 유효(0 < z ≤ 80 m), 테두리 10 px 제외, 균등 무작위.
원격 zip 은 HTTP 구간 요청으로 필요한 파일만 읽는다 (원본 zip 전체를 받지 않음).
사용: python experiments/40_build_kitti_heldout.py --out ~/data/drive_eval/kitti_ho --data_root $DATA_ROOT
"""
from __future__ import annotations
import argparse, io, json, os, re, zipfile, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from PIL import Image

K = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class HTTPFile(io.RawIOBase):
    """zipfile 이 읽을 수 있는 구간 요청 파일 객체."""
    def __init__(self, url):
        self.url = url; self.pos = 0
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off; return self.pos
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.size, self.pos + n) - 1
        for attempt in range(5):
            try:
                b = urllib.request.urlopen(urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"}), timeout=60).read(); break
            except Exception:
                if attempt == 4: raise
        self.pos += len(b); return b
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

def remote_zip(url): return zipfile.ZipFile(io.BufferedReader(HTTPFile(url), buffer_size=1 << 20))

def pool_drives():
    s = set()
    for p in ("outdoor", "mixed"):
        ids = pd.read_parquet(os.path.join(ROOT, f"pools/{p}/teacher_labels.parquet")).image_id
        s |= {m.group(1) for m in map(lambda x: re.search(r"(20\d\d_\d\d_\d\d_drive_\d{4})", x), ids) if m}
    return s

def kitti15_drives():
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(f"{K}/devkit_scene_flow.zip").read()))
    txt = z.read(next(n for n in z.namelist() if n.endswith("mapping/train_mapping.txt"))).decode()
    return {l.split()[1].replace("_sync", "") for l in txt.splitlines() if l.strip()}

def p_rect_02(date, calib_dir):
    z = zipfile.ZipFile(os.path.join(calib_dir, f"{date}_calib.zip"))
    for l in z.read(f"{date}/calib_cam_to_cam.txt").decode().splitlines():
        if l.startswith("P_rect_02:"): P = np.array(l.split()[1:], float).reshape(3, 4); return P[0, 0], P[1, 1], P[0, 2], P[1, 2]

def dhash(im):
    g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), np.int16); b = (g[:, 1:] > g[:, :-1]).flatten()
    return np.uint64(int("".join("1" if x else "0" for x in b), 2))

def small(im): g = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), np.float32).flatten(); return (g - g.mean()) / (g.std() + 1e-6)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--calib_dir", default=os.path.expanduser("~/data/kitti/raw"))
    ap.add_argument("--data_root", required=True, help="풀 이미지가 있는 DATA_ROOT (장소 중복 검사용)")
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--px", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scan_stride", type=int, default=5); ap.add_argument("--window", type=int, default=50, help="풀과 겹친 프레임 앞뒤로 제외할 프레임 수 (10 Hz)")
    a = ap.parse_args(); out = os.path.expanduser(a.out); os.makedirs(f"{out}/image", exist_ok=True); rng = np.random.default_rng(a.seed)

    excl = pool_drives() | kitti15_drives(); print(f"제외 주행 {len(excl)}개 (풀 + KITTI 2015 원본)", flush=True)
    ann = remote_zip(f"{K}/data_depth_annotated.zip")
    gt = {}   # drive -> [(frame, 멤버 이름)]
    for n in ann.namelist():
        m = re.match(r"(train|val)/(20\d\d_\d\d_\d\d_drive_\d{4})_sync/proj_depth/groundtruth/image_02/(\d+)\.png$", n)
        if m and m.group(2) not in excl: gt.setdefault(m.group(2), []).append((m.group(3), n))
    # 캠퍼스 주행(2011_09_28)은 주행 번호가 달라도 같은 장소를 찍어 풀의 캠퍼스 주행과 겹친다 (v0 세트에서 50장 중 19장이 상관 > 0.9) → 전부 제외
    city = sorted(d for d in gt if not d.startswith("2011_09_28"))
    print(f"후보 주행: 도시·주거·도로 {len(city)} (캠퍼스 {sum(d.startswith('2011_09_28') for d in gt)}개 제외)", flush=True)

    # 장소 단위 중복 제거: 풀 KITTI 이미지(KITTI 2012 는 주행 매핑이 없어 여기서만 걸러진다)와 겹치는 프레임의 앞뒤 window 프레임을 뺀다
    pool = sorted(set().union(*[set(pd.read_parquet(os.path.join(ROOT, f"pools/{p}/teacher_labels.parquet")).query("source.str.startswith('kitti')", engine="python").image_id) for p in ("outdoor", "mixed")]))
    ph = np.array([dhash(Image.open(os.path.join(a.data_root, p))) for p in pool], dtype=np.uint64); print(f"풀 KITTI 이미지 {len(pool)}장 해시", flush=True)
    def match(im):   # 풀 이미지와 같은 장소면 (풀 경로, 상관) 반환
        d = np.array([bin(int(x)).count("1") for x in np.bitwise_xor(ph, dhash(im))]); v = small(im)
        for j in np.nonzero(d <= 10)[0]:
            c = float(np.dot(v, small(Image.open(os.path.join(a.data_root, pool[j])))) / 1024)
            if c > 0.9: return pool[j], c
        return None
    def scan(d):
        z = remote_zip(f"{K}/raw_data/{d}/{d}_sync.zip"); fr = sorted(gt[d]); bad = set()
        for i in range(0, len(fr), a.scan_stride):
            m = match(Image.open(io.BytesIO(z.read(f"{d[:10]}/{d}_sync/image_02/data/{fr[i][0]}.png"))))
            if m: bad |= set(range(i - a.window, i + a.window + 1)); print(f"  {d} {fr[i][0]} ↔ {m[0]} ({m[1]:.3f})", flush=True)
        return d, [f for i, f in enumerate(fr) if i not in bad]
    with ThreadPoolExecutor(6) as ex: allowed = dict(ex.map(scan, city))
    for d in city: print(f"{d}: GT 프레임 {len(gt[d])} → 장소 중복 제외 후 {len(allowed[d])}", flush=True)

    # 주행당 상한 = 남은 프레임 수 // 10 (10 Hz 에서 1 초 이상 간격, 연속 프레임 중복 방지). 상한에 걸린 몫은 나머지 주행에 고르게 나눈다
    city = [d for d in city if allowed[d]]
    cap = np.array([max(1, len(allowed[d]) // 10) for d in city]); per = np.zeros(len(city), int)
    while per.sum() < a.n and (per < cap).any():
        open_ = np.nonzero(per < cap)[0]; share = max(1, (a.n - per.sum()) // len(open_))
        for i in open_:
            if per.sum() < a.n: per[i] += min(share, cap[i] - per[i])
    print("주행별 장수:", dict(zip(city, per.tolist())), flush=True)
    pick = []   # (drive, frame, gt 멤버)
    for d, k in zip(city, per):
        fr = allowed[d]; idx = np.linspace(0, len(fr) - 1, k + 2)[1:-1].round().astype(int); pick += [(d, *fr[i]) for i in idx]
    assert len({(d, f) for d, f, _ in pick}) == len(pick), "같은 프레임이 두 번 뽑혔다"

    recs = []; raw = {}; leaks = 0
    for d, frame, gname in pick:
        date = d[:10]
        if d not in raw: raw[d] = remote_zip(f"{K}/raw_data/{d}/{d}_sync.zip")
        img = raw[d].read(f"{date}/{d}_sync/image_02/data/{frame}.png")
        if match(Image.open(io.BytesIO(img))): leaks += 1; print(f"!!! {d} {frame} 가 풀 이미지와 겹친다", flush=True)
        z = np.asarray(Image.open(io.BytesIO(ann.read(gname))), np.float32) / 256.0
        H, W = z.shape; fx, fy, cx, cy = p_rect_02(date, a.calib_dir)
        v, u = np.nonzero((z > 0) & (z <= 80)); keep = (u >= 10) & (u < W - 10) & (v >= 10) & (v < H - 10); v, u = v[keep], u[keep]
        s = rng.choice(len(u), size=a.px, replace=False); u, v = u[s], v[s]; zz = z[v, u]
        eu = zz * np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)   # 유클리드 거리 (기존 평가 세트와 같은 정의)
        name = f"image/{d}_{frame}.png"; open(f"{out}/{name}", "wb").write(img)
        recs.append({"image": name, "scene": d, "intrinsics": [fx, fy, cx, cy, W, H], "pixel_coords": [[int(x), int(y)] for x, y in zip(u, v)],
                     "depth": [float(x) for x in eu], "depth_z": [float(x) for x in zz]})
        print(f"{len(recs)}/{len(pick)} {d} {frame} z {zz.min():.1f}–{zz.max():.1f} m", flush=True)
    with open(f"{out}/kitti_ho_val.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    print(f"완료: 이미지 {len(recs)}장, 주행 {len({r['scene'] for r in recs})}개, 픽셀 {sum(len(r['depth']) for r in recs)}, 풀과 겹친 선정 이미지 {leaks}장 → {out}/kitti_ho_val.jsonl", flush=True)

if __name__ == "__main__":
    main()
