"""평가 픽셀의 부트스트랩 군집 — 같은 군집의 픽셀은 함께 다시 뽑는다 (NOTES D-24).

주행 세트(DDAD, nuScenes)는 장면당 여러 장(DDAD 5장, nuScenes 약 6장)이라 같은 장면의 사진끼리 상관이 있어 장면을 군집으로 쓴다.
실내 세트(iBims-1, NYUv2 — 사진 한 장이 사실상 장면 하나)와 그 밖의 세트는 사진을 군집으로 쓴다.
장면은 평가 세트 jsonl 의 "scene" 필드에서 읽는다 ($DATA_ROOT/eval/<name>/<name>_val.jsonl). 파일이 없으면 사진 단위로 물러나고 알린다.
"""
from __future__ import annotations
import json, os
import numpy as np

SCENE_CLUSTERED = ("ddad", "nuscenes")
_cache: dict[str, dict[str, str]] = {}

def scene_map(name: str, data_root: str | None = None) -> dict[str, str] | None:
    """평가 세트의 이미지 → 장면. 장면 단위 세트가 아니거나 jsonl 이 없으면 None."""
    if name not in SCENE_CLUSTERED: return None
    if name in _cache: return _cache[name]
    root = data_root or os.environ.get("DATA_ROOT", "")
    p = os.path.join(os.path.expanduser(root), "eval", name, f"{name}_val.jsonl")
    if not os.path.exists(p):
        print(f"[clusters] {name}: {p} 없음 → 사진 단위 부트스트랩으로 대신한다", flush=True); return None
    _cache[name] = {r["image"]: r["scene"] for r in map(json.loads, open(p))}
    return _cache[name]

def clusters(name: str, image_ids, data_root: str | None = None) -> np.ndarray:
    """픽셀마다 군집 id (장면 단위 세트는 장면, 나머지는 사진)."""
    ids = np.asarray(image_ids).astype(str); m = scene_map(name, data_root)
    return ids if m is None else np.array([m.get(i, i) for i in ids])

def boot_ci(values, groups, n: int = 2000, seed: int = 0, stat=np.mean) -> tuple[float, float]:
    """군집 부트스트랩 95 % 구간: 군집을 복원 추출하고 그 군집의 픽셀을 모두 넣는다."""
    v = np.asarray(values, float); g = np.asarray(groups); u, inv = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(seed); sums = np.bincount(inv, weights=v, minlength=len(u)); cnts = np.bincount(inv, minlength=len(u))
    draws = rng.integers(0, len(u), size=(n, len(u)))
    est = sums[draws].sum(1) / np.maximum(cnts[draws].sum(1), 1) if stat is np.mean else np.array([stat(np.concatenate([v[inv == k] for k in d])) for d in draws])
    lo, hi = np.percentile(est, [2.5, 97.5]); return float(lo), float(hi)
