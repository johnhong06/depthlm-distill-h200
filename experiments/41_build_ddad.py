"""DDAD val 평가 세트 — DepthLM 공식 절차(third_party/DepthLM_Official/utils/curate_ddad.py)를 따른 부분 표본.

공식 절차: val 50 장면의 모든 샘플 × 카메라 6대, LiDAR 를 투영한 깊이맵에서 0 이 아닌 픽셀 100 개를 무작위로 (시드 없음).
여기서는 장면마다 (샘플, 카메라) 5 쌍을 무작위로 골라 250 장, 이미지당 10 픽셀 (시드 0). 마커를 그릴 수 있도록 테두리 10 px 는 뺀다.
투영: LiDAR 점 → 월드(LiDAR 자세) → 카메라(카메라 자세의 역), 픽셀마다 가장 가까운 점 (DGP 의 generate_depth_from_datum 과 같은 방식).
GT: depth = 카메라 중심까지 유클리드 거리 (다른 평가 세트와 같은 정의), depth_z = z 깊이 (공식 스크립트와 교사 논문값 δ1 0.670 비교용).
알려진 한계: 공식 절차처럼 자차 마스크를 쓰지 않는다. 본닛·차체가 보이는 카메라에서는 차체 너머 도로 점이 차체 픽셀에 찍힐 수 있다.
사용: python experiments/41_build_ddad.py --src ~/data/ddad/ddad_train_val --out ~/data/drive_eval/ddad
"""
import argparse, glob, json, os, shutil
import numpy as np

CAMS = ["CAMERA_01", "CAMERA_05", "CAMERA_06", "CAMERA_07", "CAMERA_08", "CAMERA_09"]

def pose(p):
    q = p["rotation"]; w, x, y, z = q["qw"], q["qx"], q["qy"], q["qz"]
    M = np.eye(4); t = p["translation"]; M[:3, 3] = [t["x"], t["y"], t["z"]]
    M[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return M

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--per_scene", type=int, default=5); ap.add_argument("--px", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); src, out = os.path.expanduser(a.src), os.path.expanduser(a.out); os.makedirs(f"{out}/image", exist_ok=True)
    rng = np.random.default_rng(a.seed); split = json.load(open(f"{src}/ddad.json"))["scene_splits"]["1"]
    val = split["filenames"] if isinstance(split, dict) else split
    recs = []
    for sf in sorted(val):
        sd = os.path.join(src, os.path.dirname(sf)); s = json.load(open(os.path.join(src, sf))); D = {x["key"]: x for x in s["data"]}
        pairs = [(i, c) for i in range(len(s["samples"])) for c in CAMS]
        for k in sorted(rng.choice(len(pairs), size=a.per_scene, replace=False)):
            i, cam = pairs[k]; smp = s["samples"][i]; dat = {D[x]["id"]["name"]: D[x] for x in smp["datum_keys"]}
            cal = json.load(open(f"{sd}/calibration/{smp['calibration_key']}.json")); K = cal["intrinsics"][cal["names"].index(cam)]
            im = dat[cam]["datum"]["image"]; li = dat["LIDAR"]["datum"]["point_cloud"]; W, H = im["width"], im["height"]
            P = np.load(f"{sd}/{li['filename']}")["data"][:, :3]
            X = (np.linalg.inv(pose(im["pose"])) @ pose(li["pose"]) @ np.c_[P, np.ones(len(P))].T)[:3].T; X = X[X[:, 2] > 0]
            u = np.floor(K["fx"] * X[:, 0] / X[:, 2] + K["cx"]).astype(int); v = np.floor(K["fy"] * X[:, 1] / X[:, 2] + K["cy"]).astype(int)
            ok = (u >= 10) & (u < W - 10) & (v >= 10) & (v < H - 10); u, v, X = u[ok], v[ok], X[ok]
            o = np.argsort(-X[:, 2]); zb = {}                      # 픽셀마다 가장 가까운 점 (먼 것부터 덮어쓴다)
            for j in o: zb[(u[j], v[j])] = j
            keys = list(zb.keys()); sel = rng.choice(len(keys), size=a.px, replace=False); idx = [zb[keys[t]] for t in sel]
            Xs = X[idx]; eu = np.linalg.norm(Xs, axis=1)
            name = f"image/{os.path.basename(sd)}_{cam}_{os.path.splitext(os.path.basename(im['filename']))[0]}.png"
            shutil.copyfile(f"{sd}/{im['filename']}", f"{out}/{name}")
            recs.append({"image": name, "scene": os.path.basename(sd), "camera": cam, "intrinsics": [K["fx"], K["fy"], K["cx"], K["cy"], W, H],
                         "pixel_coords": [[int(keys[t][0]), int(keys[t][1])] for t in sel], "depth": [float(x) for x in eu], "depth_z": [float(x) for x in Xs[:, 2]]})
        print(f"{os.path.basename(sd)}: {len(recs)}장 누적", flush=True)
    with open(f"{out}/ddad_val.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    print(f"완료: 이미지 {len(recs)}장, 장면 {len({r['scene'] for r in recs})}개, 픽셀 {sum(len(r['depth']) for r in recs)} → {out}/ddad_val.jsonl", flush=True)

if __name__ == "__main__":
    main()
