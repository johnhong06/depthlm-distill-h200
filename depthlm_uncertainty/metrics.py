"""답변 파싱 + depth error metric.

공식 `utils/metrics.py`는 `math_verify.parse`에 의존하는데, 이 패키지는
`~/venv/main`에 없고 심볼릭 수식 파서라 "12.37" 하나 뽑는 데는 과하다.
δ₁ 정의(max(d̂/d, d/d̂) < 1.25)는 공식 구현과 동일하게 맞췄다.
"""

from __future__ import annotations

import re

import numpy as np

# <answer> ... </answer> 안의 첫 실수
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# 부호를 일부러 포함시킨다. 부호를 빼면 "-3" 이 "3" 으로 조용히 읽혀
# 양수 검사를 통과해버린다. 붙여서 읽은 뒤 value <= 0 으로 걸러내야 한다.
_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

# "around 1.88 meters" — depth 값이 처음 결정되는 지점을 잡는 주 패턴.
# 숫자 앞뒤 표현이 조금 달라도 걸리도록 느슨하게 둔다.
_AROUND_METERS_RE = re.compile(
    r"around\s+([-+]?\d+(?:\.\d+)?)\s*(?:m\b|meter)", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL | re.IGNORECASE)

DELTA1_THRESHOLD = 1.25


def parse_depth_answer(text: str) -> float | None:
    """생성 텍스트에서 예측 depth(m)를 뽑는다.

    1순위: <answer> 태그 안의 첫 숫자.
    2순위: 태그가 닫히지 않은 경우 (max_new_tokens 절단 등) '<answer>' 뒤의 첫 숫자.
    실패하면 None — 호출부에서 parse_ok=False로 기록한다. 텍스트 전체의
    마지막 숫자로 넘어가는 fallback은 일부러 두지 않았다. <think> 안의 숫자를
    답으로 착각하면 조용히 틀린 값이 섞이기 때문이다.
    """
    m = _ANSWER_RE.search(text)
    if m is not None:
        span = m.group(1)
    else:
        idx = text.lower().find("<answer>")
        if idx < 0:
            return None
        span = text[idx + len("<answer>") :]

    num = _FLOAT_RE.search(span)
    if num is None:
        return None
    try:
        value = float(num.group(0))
    except ValueError:
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    return value


def _token_offsets(token_strings: list[str]) -> tuple[list[tuple[int, int]], str]:
    offsets, cursor = [], 0
    for t in token_strings:
        offsets.append((cursor, cursor + len(t)))
        cursor += len(t)
    return offsets, "".join(token_strings)


def _span_of_number(
    token_strings: list[str], lo: int, hi: int
) -> tuple[int, int] | None:
    """문자 구간 [lo, hi) 안의 첫 숫자와 겹치는 토큰 인덱스 구간 [start, end)."""
    offsets, full = _token_offsets(token_strings)
    num = _FLOAT_RE.search(full, lo, hi)
    if num is None:
        return None
    nlo, nhi = num.start(), num.end()
    idxs = [i for i, (a, b) in enumerate(offsets) if a < nhi and b > nlo]
    if not idxs:
        return None
    return idxs[0], idxs[-1] + 1


def find_depth_token_span(
    token_strings: list[str],
) -> tuple[tuple[int, int] | None, str]:
    """모델이 depth 값을 **처음 결정하는** 숫자 토큰 구간과, 그걸 찾은 방법.

    DepthLM 답변은
        <think> The point is around 1.88 meters away from the camera. </think>
        <answer> 1.88 </answer>
    형태다. <answer>의 숫자는 <think>에서 이미 말한 값을 그대로 **복사**하는 것이라
    토큰 확률이 1.0에 붙어버린다 (Gate 0 실측: 2,000샘플 전부 min_prob > 0.999).
    모델이 실제로 값을 **결정**하는 지점은 <think> 안의 depth 숫자다.

    "첫 숫자"로 찾으면 reasoning trace 에 depth 앞에 다른 숫자가 섞이는 순간
    (예: "the marker at pixel 320, the point is around 5 meters") 엉뚱한 토큰을
    재게 된다. 그래서 **"around X meters" 패턴을 1순위**로 쓴다.

    Returns:
        (토큰 구간 [start, end), 찾은 방법)
        방법: "pattern" | "think_first_number" | "first_number" | "none"
    """
    _, full = _token_offsets(token_strings)

    # 1순위: "around <숫자> meter(s)" — DepthLM 답변 템플릿
    m = _AROUND_METERS_RE.search(full)
    if m is not None:
        span = _span_of_number(token_strings, m.start(1), m.end(1))
        if span is not None:
            return span, "pattern"

    # 2순위: <think> 구간 안의 첫 숫자
    tm = _THINK_RE.search(full)
    if tm is not None:
        span = _span_of_number(token_strings, tm.start(1), tm.end(1))
        if span is not None:
            return span, "think_first_number"

    # 3순위: 생성물 전체의 첫 숫자
    span = _span_of_number(token_strings, 0, len(full))
    return (span, "first_number") if span is not None else (None, "none")


def span_number_value(
    token_strings: list[str], span: tuple[int, int] | None
) -> float | None:
    """토큰 구간이 실제로 담고 있는 숫자 값. span 식별이 맞았는지 검증용."""
    if span is None:
        return None
    num = _FLOAT_RE.search("".join(token_strings[span[0] : span[1]]))
    if num is None:
        return None
    try:
        return float(num.group(0))
    except ValueError:
        return None


def find_answer_token_span(token_strings: list[str]) -> tuple[int, int] | None:
    """<answer>…</answer> 안의 숫자 토큰 구간. 대조군용 (위 설명 참고)."""
    _, full = _token_offsets(token_strings)
    m = _ANSWER_RE.search(full)
    if m is not None:
        lo, hi = m.start(1), m.end(1)
    else:
        idx = full.lower().find("<answer>")
        if idx < 0:
            return None
        lo, hi = idx + len("<answer>"), len(full)
    return _span_of_number(token_strings, lo, hi)


# ---------------------------------------------------------------------------
# depth error metric
# ---------------------------------------------------------------------------

def absolute_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.abs(pred - gt)


def absolute_relative_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """AbsRel = |d̂ − d| / d"""
    return np.abs(pred - gt) / gt


def squared_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return (pred - gt) ** 2


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean(squared_error(pred, gt))))


def delta1_hits(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """공식 정의: max(d̂/d, d/d̂) < 1.25 인가."""
    ratio = np.maximum(pred / gt, gt / pred)
    return (ratio < DELTA1_THRESHOLD).astype(np.float64)


def summarize(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    return {
        "n": int(pred.size),
        "delta1": float(np.mean(delta1_hits(pred, gt))),
        "mae": float(np.mean(absolute_error(pred, gt))),
        "absrel": float(np.mean(absolute_relative_error(pred, gt))),
        "rmse": rmse(pred, gt),
    }
