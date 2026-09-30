"""nuScenes 평가 세트 — DepthLM 공식 절차(third_party/DepthLM_Official/utils/curate_nuscenes_eval.py)를 따른 부분 표본.

공식 절차: 평가 = v1.0-mini 10장면 전부(학습에서 제외, DepthLM_Official 이슈 #17 저자 답변. 공개 스크립트의 trainval 마지막 5 % 는 공개 시 실수) × 카메라 6대 전부,
LIDAR_TOP 키프레임 점을 캘리브레이션(센서→차량)만으로 카메라에 투영(차량 자세·시간차 보정 없음), 픽셀마다 가장 가까운 점,
깊이 0 이 아닌 픽셀에서 100 개 무작위(시드 없음), GT = z 깊이.
여기서는 (샘플, 카메라) 쌍 10,248 개 중 250 쌍을 무작위로 (시드 0), 이미지당 10 픽셀, 마커를 그릴 수 있도록 테두리 10 px 제외.
GT: depth = 유클리드 거리 (다른 평가 세트와 같은 정의), depth_z = z 깊이 (공식·교사 논문값 δ1 0.819 비교용).
단계: --list 로 필요한 파일 목록을 쓰고 → tar -xzf <blob> -T <목록> 으로 그 파일만 꺼낸 뒤 → 목록 없이 다시 실행해 세트를 만든다.
사용: python experiments/43_build_nuscenes.py --root ~/data/nuscenes --out ~/data/drive_eval/nuscenes [--list files.txt]
"""
import argparse, json, os, shutil
import numpy as np

CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]

def quat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--list", default="")
    ap.add_argument("--version", default="v1.0-mini", help="v1.0-mini = 논문 실제 평가(이슈 #17), v1.0-trainval = 공개 스크립트의 마지막 5 퍼센트"); ap.add_argument("--n", type=int, default=250); ap.add_argument("--px", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); root, out = os.path.expanduser(a.root), os.path.expanduser(a.out); M = f"{root}/{a.version}"
    sample = json.load(open(f"{M}/sample.json")); sd = json.load(open(f"{M}/sample_data.json"))
    cs = {c["token"]: c for c in json.load(open(f"{M}/calibrated_sensor.json"))}; sen = {s["token"]: s["channel"] for s in json.load(open(f"{M}/sensor.json"))}
    scene = {s["token"]: s["name"] for s in json.load(open(f"{M}/scene.json"))}
    key = {}   # (sample_token, channel) -> 키프레임 sample_data
    for d in sd:
        if d["is_key_frame"]: key[(d["sample_token"], sen[cs[d["calibrated_sensor_token"]]["sensor_token"]])] = d
    # 논문 실험은 v1.0-mini 10장면 전부로 평가했고 학습에서는 뺐다(DepthLM_Official 이슈 #17, 저자 답변). 공개 스크립트의 trainval 마지막 5 % 는 공개 시 실수
    ev = sample if a.version == "v1.0-mini" else sample[int(len(sample) * 0.95):]
    pairs = [(s["token"], c) for s in ev for c in CAMS]
    rng = np.random.default_rng(a.seed); pick = [pairs[i] for i in sorted(rng.choice(len(pairs), size=a.n, replace=False))]
    print(f"평가 샘플 {len(ev)} (장면 {len({s['scene_token'] for s in ev})}), (샘플, 카메라) 쌍 {len(pairs)} 중 {len(pick)}", flush=True)
    need = sorted({key[(t, c)]["filename"] for t, c in pick} | {key[(t, "LIDAR_TOP")]["filename"] for t, c in pick})
    if a.list:
        open(os.path.expanduser(a.list), "w").write("\n".join(need) + "\n"); print(f"필요한 파일 {len(need)}개 → {a.list}"); return
    miss = [f for f in need if not os.path.exists(f"{root}/{f}")]
    if miss: print(f"!!! 없는 파일 {len(miss)}개 (예: {miss[:3]}) — 다른 blob 파트가 필요하다"); raise SystemExit(1)
    os.makedirs(f"{out}/image", exist_ok=True); stok = {s["token"]: s for s in sample}; recs = []
    for t, c in pick:
        cam, lid = key[(t, c)], key[(t, "LIDAR_TOP")]; cc, lc = cs[cam["calibrated_sensor_token"]], cs[lid["calibrated_sensor_token"]]
        P = np.fromfile(f"{root}/{lid['filename']}", dtype=np.float32).reshape(-1, 5)[:, :3]
        X = P @ quat(lc["rotation"]).T + np.array(lc["translation"])                  # 라이다 → 차량 (공식과 같이 차량 자세는 쓰지 않음)
        X = (X - np.array(cc["translation"])) @ quat(cc["rotation"])                   # 차량 → 카메라 (R^T (x - t))
        X = X[X[:, 2] > 0]; K = np.array(cc["camera_intrinsic"]); W, H = cam["width"], cam["height"]
        uv = (X @ K.T); uv = uv[:, :2] / uv[:, 2:3]; u, v = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
        ok = (u >= 10) & (u < W - 10) & (v >= 10) & (v < H - 10); u, v, X = u[ok], v[ok], X[ok]
        zb = {}
        for j in np.argsort(-X[:, 2]): zb[(u[j], v[j])] = j                            # 픽셀마다 가장 가까운 점
        keys = list(zb.keys()); sel = rng.choice(len(keys), size=a.px, replace=False); Xs = X[[zb[keys[s]] for s in sel]]
        name = f"image/{os.path.basename(cam['filename'])}"; shutil.copyfile(f"{root}/{cam['filename']}", f"{out}/{name}")
        recs.append({"image": name, "scene": scene[stok[t]["scene_token"]], "camera": c, "intrinsics": [K[0, 0], K[1, 1], K[0, 2], K[1, 2], W, H],
                     "pixel_coords": [[int(keys[s][0]), int(keys[s][1])] for s in sel], "depth": [float(x) for x in np.linalg.norm(Xs, axis=1)], "depth_z": [float(x) for x in Xs[:, 2]]})
    with open(f"{out}/nuscenes_val.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    print(f"완료: 이미지 {len(recs)}장, 장면 {len({r['scene'] for r in recs})}개, 픽셀 {sum(len(r['depth']) for r in recs)} → {out}/nuscenes_val.jsonl", flush=True)

if __name__ == "__main__":
    main()
