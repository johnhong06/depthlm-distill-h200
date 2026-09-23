"""U_grounding — pixel-grounding uncertainty.

같은 질의 픽셀에 대해 marker를 아주 조금씩 옮겨 렌더링한 뒤 depth를 다시 예측하고,
그 예측들의 흩어진 정도를 잰다.

  "VLM이 marker가 가리키는 공간 위치를 얼마나 안정적으로 이해하는가?"

DepthLM 논문(Finding 1)은 marker 기반 pixel reference가 text 좌표보다 낫다고
보고하므로, 이 측정은 DepthLM의 핵심 메커니즘을 직접 겨냥한다.
"""

from __future__ import annotations

from .dispersion import dispersion_stats


def grounding_uncertainty(preds: list[float]) -> dict[str, float]:
    """오프셋별 예측 리스트 → dispersion 지표. 주 지표는 grounding_cov."""
    return dispersion_stats(preds, prefix="grounding")
