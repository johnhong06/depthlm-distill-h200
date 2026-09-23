"""예측 집합의 흩어진 정도. grounding / stochastic uncertainty가 공유한다."""

from __future__ import annotations

import numpy as np


def dispersion_stats(preds: list[float] | np.ndarray, prefix: str) -> dict[str, float]:
    """dispersion 지표 한 세트.

    주 지표는 **CoV**(변동계수)로 정한다. std를 주 지표로 쓰면 먼 점일수록
    절대분산이 커지기 때문에 "uncertainty ↔ error" 상관이 사실상
    "depth ↔ error" 상관으로 오염된다. CoV는 depth 스케일에 불변이다.
    """
    p = np.asarray([v for v in np.asarray(preds, dtype=np.float64) if np.isfinite(v)])
    out: dict[str, float] = {f"{prefix}_n": float(p.size)}

    if p.size == 0:
        for k in ("std", "mad", "range_norm", "cov", "iqr", "median", "mean"):
            out[f"{prefix}_{k}"] = float("nan")
        return out

    median = float(np.median(p))
    mean = float(np.mean(p))
    denom = median if median > 1e-9 else float("nan")

    out[f"{prefix}_mean"] = mean
    out[f"{prefix}_median"] = median
    out[f"{prefix}_std"] = float(np.std(p, ddof=1)) if p.size > 1 else 0.0
    # MAD: median absolute deviation (robust)
    out[f"{prefix}_mad"] = float(np.median(np.abs(p - median)))
    out[f"{prefix}_range_norm"] = float((p.max() - p.min()) / denom)
    out[f"{prefix}_iqr"] = float(np.percentile(p, 75) - np.percentile(p, 25))
    out[f"{prefix}_cov"] = out[f"{prefix}_std"] / denom
    return out
