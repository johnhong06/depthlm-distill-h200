"""숫자 토큰 트리 열거 — DepthLM 출력 숫자의 **명시적 확률분포** p(depth | image, marker).

배경 (D-25, 09_smoke_beam):
  DepthLM 출력은 "<think> The point is around #.## meters ..." 고정 템플릿이라, 자유도는
  숫자 토큰 4~5개뿐이다. 자릿수가 개별 토큰(['3','.','7','8'])이므로 숫자 위치에서 분기해
  트리를 열거하면 모델이 정의하는 depth 분포를 거의 전부 덮을 수 있다.
  beam search(K=8)는 끝자리가 거의 균등분포라 8개 후보의 질량이 ~0.11 에 그친다.

방법:
  1. 프롬프트 + 숫자 앞 템플릿 토큰을 prefill → prefix KV 캐시 + 첫 자릿수 분포
  2. 확률 ≥ min_branch_p 인 숫자/소수점 토큰으로 분기. 같은 레벨의 분기들을 batch 로 묶어
     prefix 캐시를 복제(batch_repeat_interleave)한 뒤 짧은 continuation 만 forward
  3. 숫자가 아닌 토큰(' meters' 등)으로 이어지는 질량 = 그 자리에서 숫자가 끝나는 확률

산출: [(value, prob)], covered_mass(열거한 질량), 그리고 이 분포로부터의 디코더/불확실도.
GT 는 어디에도 들어가지 않는다.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import DynamicCache


def _is_numeric_token(s: str) -> bool:
    return len(s) > 0 and all(c.isdigit() or c == "." for c in s)


@dataclass
class NumberDistribution:
    values: np.ndarray            # 후보 숫자
    probs: np.ndarray             # 각 후보의 절대 확률 (합 = covered_mass)
    covered_mass: float           # 열거된 총 질량 (1 에 가까울수록 완전)
    invalid_mass: float           # 파싱 불가 문자열(예: "3.")로 끝난 질량
    n_forward: int                # prefill 제외 branch forward 호출 수
    n_nodes: int                  # 평가한 트리 노드 수

    # ---- 디코더 (전부 GT 무관) ----
    def normalized(self):
        p = self.probs / self.probs.sum()
        return self.values, p

    def expected_value(self) -> float:
        v, p = self.normalized(); return float((v * p).sum())

    def geometric_mean(self) -> float:
        v, p = self.normalized(); return float(math.exp((p * np.log(v)).sum()))

    def median(self) -> float:
        v, p = self.normalized(); o = np.argsort(v); c = np.cumsum(p[o])
        return float(v[o][np.searchsorted(c, 0.5)])

    def map_value(self) -> float:
        v, p = self.normalized(); return float(v[np.argmax(p)])

    def delta1_optimal(self, thr: float = 1.25) -> float:
        """δ₁ 기준 Bayes-최적 점추정: 후보 d 중 [d/thr, d·thr) 안 질량이 최대인 것."""
        v, p = self.normalized()
        best, best_m = None, -1.0
        for d in v:
            m = p[(np.maximum(v / d, d / v) < thr)].sum()
            if m > best_m:
                best, best_m = d, m
        return float(best)

    def delta1_mass_at(self, d: float, thr: float = 1.25) -> float:
        """예측 d 에 대해 δ₁ 창 안의 확률질량 — 분포 기반 '적중 확률'."""
        v, p = self.normalized()
        return float(p[(np.maximum(v / d, d / v) < thr)].sum())

    # ---- 불확실도 ----
    def stats(self) -> dict[str, float]:
        v, p = self.normalized()
        mu = (v * p).sum(); sd = math.sqrt(max((p * (v - mu) ** 2).sum(), 0.0))
        ent = float(-(p * np.log(np.clip(p, 1e-30, None))).sum())
        lv = np.log(v); lmu = (p * lv).sum(); lsd = math.sqrt(max((p * (lv - lmu) ** 2).sum(), 0.0))
        return {"dist_mean": float(mu), "dist_std": sd, "dist_cov": float(sd / mu),
                "dist_logstd": lsd, "dist_entropy": ent, "dist_n": int(len(v)),
                "dist_covered": self.covered_mass, "dist_invalid": self.invalid_mass}


@torch.no_grad()
def enumerate_number_distribution(
    model, tok, inputs: dict, prefix_gen_ids: torch.Tensor,
    min_branch_p: float = 0.005, min_path_p: float = 5e-4,
    max_depth: int = 7, topk: int = 16, chunk: int = 8, record: list | None = None,
    stop_first_decimal: bool = False,
) -> NumberDistribution:
    """
    inputs: processor 출력 (batch 1, 이미지 포함). prefix_gen_ids: 숫자 직전까지의 생성 토큰 id (1D).
    stop_first_decimal: True 면 소수 첫째 자리가 확정된 접두사("2.8", "12.3")에서 더 펼치지 않고 접두사 질량 전체를
            x.x5 대표값에 준다 (D-54 fd 규칙). branch forward 가 ~30 → ~2-3 으로 준다.
    record: 리스트를 주면 평가한 노드마다 {"path", "logp", "numeric_mass", "children":[(tok_str, tlp), ...]} 를
            덧붙인다. 느슨한 임계값으로 한 번 돌려 두면 더 엄격한 가지치기·조기 종료를 사후에 정확히 재현할 수 있다
            (엄격한 설정의 확장 집합은 느슨한 설정의 부분집합이므로).
    """
    dev = inputs["input_ids"].device
    ids = torch.cat([inputs["input_ids"], prefix_gen_ids.to(dev)[None]], dim=1)
    P = ids.shape[1]
    extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=DynamicCache(),
                use_cache=True, logits_to_keep=1, **extra)
    base_cache = out.past_key_values
    dist = {(): torch.log_softmax(out.logits[0, -1].float(), dim=-1)}
    del out
    # Qwen2.5-VL 류(mrope): 캐시가 있는 forward 에서 attention_mask 로 위치를 재구성하면 길이가 P+L 이 되어 L 토큰 입력과
    # 어긋난다. rope_deltas 가 있으면 텍스트 위치를 명시적으로 넘긴다 (3, B, L). DepthLM(Llava) 에는 이 속성이 없어 영향 없음.
    rope_deltas = getattr(getattr(model, "model", None), "rope_deltas", None)

    frontier: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    finished: list[tuple[str, float]] = []   # (문자열, 절대 확률)
    n_forward = n_nodes = 0
    depth = 0
    while frontier and depth < max_depth:
        to_eval: list[tuple[tuple[int, ...], float]] = []
        for path, lp in frontier:
            n_nodes += 1
            lg = dist[path]
            top = torch.topk(lg, topk)
            numeric_mass = 0.0; children = []
            for tid, tlp in zip(top.indices.tolist(), top.values.tolist()):
                s = tok.decode([tid], skip_special_tokens=False)
                if _is_numeric_token(s):
                    numeric_mass += math.exp(tlp); children.append((s, tlp))
                    if math.exp(tlp) >= min_branch_p and math.exp(lp + tlp) >= min_path_p:
                        cs = tok.decode(list(path) + [tid], skip_special_tokens=False)
                        if stop_first_decimal and "." in cs and len(cs.split(".", 1)[1]) >= 1:
                            finished.append((cs + "5" if len(cs.split(".", 1)[1]) == 1 else cs, math.exp(lp + tlp)))   # 접두사 질량 → x.x5
                        else:
                            to_eval.append((path + (tid,), lp + tlp))
            if record is not None:
                record.append({"path": tok.decode(list(path), skip_special_tokens=False) if path else "",
                               "logp": lp, "numeric_mass": numeric_mass, "children": children})
            # 숫자가 아닌 토큰으로 넘어가는 질량 = 여기서 숫자가 끝남
            if path:
                finished.append((tok.decode(list(path), skip_special_tokens=False),
                                 math.exp(lp) * max(0.0, 1.0 - numeric_mass)))
        # 같은 레벨의 분기를 batch 로 평가 (continuation 길이 동일)
        for k in range(0, len(to_eval), chunk):
            grp = to_eval[k:k + chunk]
            B, L = len(grp), len(grp[0][0])
            c = copy.deepcopy(base_cache)
            if B > 1:
                c.batch_repeat_interleave(B)
            inp = torch.tensor([list(p) for p, _ in grp], device=dev)
            kw = {}
            if rope_deltas is not None:
                pos = torch.arange(P, P + L, device=dev).view(1, 1, L).expand(3, B, L) + rope_deltas.to(dev).view(1, 1, 1)
                kw["position_ids"] = pos
            o = model(input_ids=inp, attention_mask=torch.ones(B, P + L, device=dev, dtype=torch.long),
                      past_key_values=c, use_cache=True, logits_to_keep=1, **kw)
            lg = torch.log_softmax(o.logits[:, -1].float(), dim=-1)
            for b, (p, _) in enumerate(grp):
                dist[p] = lg[b]
            n_forward += 1
            del o, c
        frontier = to_eval
        depth += 1
    del base_cache

    vals, probs, invalid = [], [], 0.0
    for s, pr in finished:
        if pr <= 0:
            continue
        try:
            v = float(s)
            if not (math.isfinite(v) and v > 0):
                raise ValueError
        except ValueError:
            invalid += pr; continue
        vals.append(v); probs.append(pr)
    # 같은 값이 다른 문자열("3.5" vs "3.50")로 나올 수 있으므로 합친다
    agg: dict[float, float] = {}
    for v, pr in zip(vals, probs):
        agg[v] = agg.get(v, 0.0) + pr
    v = np.array(sorted(agg)); p = np.array([agg[x] for x in v])
    return NumberDistribution(values=v, probs=p, covered_mass=float(p.sum()), invalid_mass=invalid,
                              n_forward=n_forward, n_nodes=n_nodes)
