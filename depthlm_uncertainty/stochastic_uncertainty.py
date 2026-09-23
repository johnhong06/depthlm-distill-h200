"""U_stochastic — 출력 확률성 uncertainty.

동일한 image + marker + prompt에 대해 T=1.0으로 M회 샘플링하고 예측 분포의
흩어진 정도를 잰다. prefix가 완전히 같으므로 prefill은 1회만 하면 된다
(num_return_sequences=M).

주의(연구 설계상): prediction consistency가 곧 accuracy라고 가정하지 않는다.
모델이 일관되게 틀릴 수 있다. 그 여부를 확인하는 것이 이 실험이다.
"""

from __future__ import annotations

from .dispersion import dispersion_stats


def stochastic_uncertainty(preds: list[float]) -> dict[str, float]:
    """샘플링된 예측 리스트 → dispersion 지표. 주 지표는 stochastic_cov."""
    return dispersion_stats(preds, prefix="stochastic")
