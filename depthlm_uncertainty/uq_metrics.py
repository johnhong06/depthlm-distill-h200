"""UQ quality metric — Sparsification / AUSE / AURG / Risk-Coverage.

**지시서 13-R: 새 metric을 만들지 않는다.** 기존 depth-UQ 문헌의 표준 구현인

    Poggi et al., "On the uncertainty of self-supervised monocular depth
    estimation", CVPR 2020.   third_party/mono-uncertainty/evaluate.py

의 `compute_aucs()` 를 **정의 그대로** 옮겼다. 아래는 원본에서 확인한 규약이다.

  intervals = 50                      → 2% 간격 (13-E 요구와 일치)
  uncert    = -uncert                 → 부호 반전 후 percentile 로 subset 구성
  quants    = [0, 2, 4, ..., 98]
  thresholds= [percentile(uncert, q) for q in quants]
  subs      = [uncert >= t]           → q 가 커질수록 high-uncertainty 부터 제거
  plotx     = [0, 1/50, ..., 1]       (51점)
  sparse    = [error(sub) for sub in subs] + [0]      ← 마지막에 0 을 붙인다
  opt       = 같은 절차를 **실제 per-sample error** 로 (metric 별로 따로)
  rnd       = 전체집합 error 로 채운 상수 curve
  AUSE[m]   = trapz(sparse) - trapz(opt)
  AURG[m]   = rnd[0]        - trapz(sparse)

원본의 두 가지 특징을 **일부러 그대로 유지**했다. 바꾸면 기존 논문 수치와
비교가 안 된다:
  1. sparse/opt 는 끝에 0 을 붙이는데 random 은 안 붙인다 (비대칭)
  2. AURG 의 첫 항이 trapz(rnd) 가 아니라 rnd[0] 이다. random curve 가 상수라
     두 값은 같지만, 코드상 표현이 그렇다.

metric 정의도 원본 `compute_eigen_errors_v2` 를 따른다:
  abs_rel : mean(|gt-pred| / gt)
  rmse    : sqrt(mean((gt-pred)^2))
  a1      : mean(max(gt/pred, pred/gt) >= 1.25)   ← δ1 이 아니라 **outlier 비율**
                                                     (낮을수록 좋음)

우리가 더한 것 (13-C 가 Absolute Error 도 요구):
  abs     : mean(|gt-pred|)          ← reference 집합에 없음. 표에 구분해 표기한다.

--- query-level 적용 (지시서 13-G) --------------------------------------------

원본은 이미지 한 장의 dense pixel 배열에 compute_aucs 를 돌리고, 그 AUSE/AURG 를
이미지들에 대해 평균한다. DepthLM 은 dense head 가 없고 질의 픽셀만 예측하므로,
13-G 지시대로 **질의 집합을 픽셀 집합과 동일한 역할로** 두고 전체를 pool 해서
한 번 계산한다.

이건 reference 와의 유일한 절차적 차이다. 이미지당 질의가 iBims1 30개 /
NYUv2 10개뿐이라, 이미지 단위로 50-interval sparsification 을 돌리면
subset 이 0~1개가 되어 곡선이 무의미해진다. `per_image=True` 로 원본 방식도
계산할 수 있게 남겨두었고 (13-R: 두 방식 모두 계산해 차이를 명시), 표본이
부족한 경우 경고를 낸다.
"""

from __future__ import annotations

import numpy as np

REFERENCE_METRICS = ("abs_rel", "rmse", "a1")
EXTRA_METRICS = ("abs",)
ALL_METRICS = REFERENCE_METRICS + EXTRA_METRICS

_trapz = getattr(np, "trapezoid", None) or np.trapz


def per_sample_error(gt: np.ndarray, pred: np.ndarray, metric: str) -> np.ndarray:
    """oracle 정렬에 쓰는 per-sample 값 (원본 reduce_mean=False 경로)."""
    if metric == "abs_rel":
        return np.abs(gt - pred) / gt
    if metric == "rmse":
        return (gt - pred) ** 2
    if metric == "a1":
        return np.maximum(gt / pred, pred / gt)
    if metric == "abs":
        return np.abs(gt - pred)
    raise ValueError(f"알 수 없는 metric: {metric}")


def reduce_error(gt: np.ndarray, pred: np.ndarray, metric: str) -> float:
    """subset 하나의 대표 error (원본 reduce_mean=True 경로)."""
    if gt.size == 0:
        return float("nan")
    if metric == "abs_rel":
        return float((np.abs(gt - pred) / gt).mean())
    if metric == "rmse":
        return float(np.sqrt(((gt - pred) ** 2).mean()))
    if metric == "a1":
        return float((np.maximum(gt / pred, pred / gt) >= 1.25).mean())
    if metric == "abs":
        return float(np.abs(gt - pred).mean())
    raise ValueError(f"알 수 없는 metric: {metric}")


def sparsification_curves(
    gt: np.ndarray,
    pred: np.ndarray,
    uncert: np.ndarray,
    metrics: tuple[str, ...] = ALL_METRICS,
    intervals: int = 50,
) -> dict:
    """uncertainty / oracle / random 세 곡선 (13-K: 곡선 자체를 저장한다)."""
    gt, pred, uncert = np.asarray(gt, float), np.asarray(pred, float), np.asarray(uncert, float)

    # 원본 그대로: 부호를 뒤집고 percentile 로 자른다
    u = -uncert
    quants = [100.0 / intervals * t for t in range(intervals)]
    plotx = [1.0 / intervals * t for t in range(intervals + 1)]

    thresholds = [np.percentile(u, q) for q in quants]
    subs = [(u >= t) for t in thresholds]

    out = {"plotx": np.asarray(plotx), "removed_fraction": np.asarray(plotx)}
    for m in metrics:
        sparse = [reduce_error(gt[s], pred[s], m) for s in subs] + [0.0]

        true_u = -per_sample_error(gt, pred, m)
        opt_subs = [(true_u >= np.percentile(true_u, q)) for q in quants]
        opt = [reduce_error(gt[s], pred[s], m) for s in opt_subs] + [0.0]

        full = reduce_error(gt, pred, m)
        rnd = [full] * (intervals + 1)

        out[m] = {"sparse": np.asarray(sparse), "oracle": np.asarray(opt),
                  "random": np.asarray(rnd)}
    return out


def compute_aucs(
    gt: np.ndarray,
    pred: np.ndarray,
    uncert: np.ndarray,
    metrics: tuple[str, ...] = ALL_METRICS,
    intervals: int = 50,
) -> dict[str, dict[str, float]]:
    """AUSE(낮을수록 좋음) / AURG(높을수록 좋음). Poggi et al. 정의 그대로."""
    c = sparsification_curves(gt, pred, uncert, metrics, intervals)
    plotx = c["plotx"]
    res = {}
    for m in metrics:
        sparse_area = float(_trapz(c[m]["sparse"], x=plotx))
        res[m] = {
            "ause": sparse_area - float(_trapz(c[m]["oracle"], x=plotx)),
            "aurg": float(c[m]["random"][0]) - sparse_area,
        }
    return res


def risk_coverage(
    gt: np.ndarray,
    pred: np.ndarray,
    uncert: np.ndarray,
    coverages: np.ndarray | None = None,
) -> dict:
    """13-N: uncertainty 가 낮은 것부터 남겼을 때 coverage 별 depth 품질.

    AUSE 와 달리 여기서는 **depth accuracy 지표를 그대로** 본다 (13-S 구분:
    AUSE 는 '틀린 걸 잘 찾는가', selective prediction 은 '실제로 쓸모가 있는가').
    δ1 은 outlier 비율이 아니라 통상 정의(높을수록 좋음)로 낸다.
    """
    gt, pred, uncert = np.asarray(gt, float), np.asarray(pred, float), np.asarray(uncert, float)
    if coverages is None:
        coverages = np.arange(1.0, 0.05, -0.05)

    order = np.argsort(uncert, kind="stable")  # 확신 높은(=U 낮은) 것부터
    n = len(gt)
    rows = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        idx = order[:k]
        g, p = gt[idx], pred[idx]
        rows.append({
            "coverage": float(cov), "n": k,
            "mae": float(np.abs(g - p).mean()),
            "absrel": float((np.abs(g - p) / g).mean()),
            "rmse": float(np.sqrt(((g - p) ** 2).mean())),
            "delta1": float((np.maximum(g / p, p / g) < 1.25).mean()),
        })
    return {"rows": rows}
