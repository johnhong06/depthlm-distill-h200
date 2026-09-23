"""U_token — 언어 생성 uncertainty.

**어느 숫자를 재느냐가 전부다.** DepthLM 답변은

    <think> The point is around 1.88 meters away from the camera. </think>
    <answer> 1.88 </answer>

인데, <answer> 안의 숫자는 <think>에서 이미 말한 값의 **복사**다. Gate 0 실측에서
iBims1 2,000샘플 전부 min_prob > 0.999, NLL ~1e-6 으로 사실상 상수였다. 복사 연산에
대한 확신을 재고 있었던 셈이다.

따라서 **주 지표는 <think>의 첫 숫자**(모델이 값을 실제로 결정하는 지점)로 잡고,
<answer> 쪽은 `token_ans_*` 로 함께 기록해 대조군으로 남긴다. 후자가 degenerate 하다는
사실 자체가 text-based depth prediction에 대한 보고할 만한 관찰이다.

주의(연구 설계상): 여기서 재는 것은 토큰 확률이지 physical depth confidence가 아니다.
둘이 같다고 가정하지 않는다. 그 관계를 검증하는 것이 이 연구의 목적이다.
"""

from __future__ import annotations

import numpy as np
import torch

from .metrics import (
    find_answer_token_span,
    find_depth_token_span,
    parse_depth_answer,
    span_number_value,
)

_KEYS = ("nll_mean", "entropy_mean", "min_prob", "seq_logprob", "max_nll", "n")


def _stats_for_span(
    span: tuple[int, int] | None,
    logits_per_step: list[torch.Tensor],
    generated_ids: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    keys = {f"{prefix}{k}": float("nan") for k in _KEYS}
    if span is None or not logits_per_step:
        return keys

    start, end = span
    end = min(end, len(logits_per_step), generated_ids.shape[0])
    if start >= end:
        return keys

    nlls, entropies, probs = [], [], []
    for t in range(start, end):
        logprobs = torch.log_softmax(logits_per_step[t].to(torch.float32), dim=-1)
        lp = float(logprobs[int(generated_ids[t].item())].item())
        nlls.append(-lp)
        probs.append(float(np.exp(lp)))
        p = torch.exp(logprobs)
        entropies.append(float(-(p * logprobs).sum().item()))  # full-vocab entropy

    return {
        f"{prefix}nll_mean": float(np.mean(nlls)),
        f"{prefix}entropy_mean": float(np.mean(entropies)),
        f"{prefix}min_prob": float(np.min(probs)),
        f"{prefix}seq_logprob": float(-np.sum(nlls)),
        f"{prefix}max_nll": float(np.max(nlls)),
        f"{prefix}n": float(len(nlls)),
    }


def token_uncertainty_from_logits(
    logits_per_step: list[torch.Tensor],
    generated_ids: torch.Tensor,
    token_strings: list[str],
) -> dict[str, float]:
    """한 시퀀스의 토큰 uncertainty.

    Args:
        logits_per_step: generate(output_logits=True)의 `logits` 중 이 시퀀스 몫.
            길이 T, 각 원소 (vocab,). `scores`가 아니라 `logits`를 쓴다 —
            scores는 logits processor를 통과한 뒤라 모델의 원분포가 아니다.
        generated_ids: (T,) 실제 생성된 토큰 id.
        token_strings: (T,) 각 토큰을 디코딩한 문자열.

    Returns:
        `token_*`     — <think> 첫 숫자 (주 지표)
        `token_ans_*` — <answer> 복사본 (대조군)
    """
    span, method = find_depth_token_span(token_strings)
    out = {
        **_stats_for_span(span, logits_per_step, generated_ids, "token_"),
        **_stats_for_span(find_answer_token_span(token_strings),
                          logits_per_step, generated_ids, "token_ans_"),
    }

    # span 식별이 맞았는지 검증한다. 우리가 확률을 잰 토큰들이 실제로 담고 있는
    # 숫자가 <answer> 의 depth 와 같아야 한다. 다르면 엉뚱한 숫자를 잰 것이므로
    # 분석에서 걸러낼 수 있게 기록해 둔다 (지시서 §4).
    span_val = span_number_value(token_strings, span)
    answer_val = parse_depth_answer("".join(token_strings))
    out["token_span_method"] = method
    out["token_span_value"] = float("nan") if span_val is None else span_val
    out["token_span_verified"] = bool(
        span_val is not None and answer_val is not None
        and abs(span_val - answer_val) < 1e-6
    )
    # 지시서 §3: token span metadata 저장
    out["depth_token_text"] = (
        "".join(token_strings[span[0]:span[1]]) if span is not None else "")
    out["depth_token_ids"] = (
        [int(i) for i in generated_ids[span[0]:span[1]]] if span is not None else [])
    return out
