"""DepthLM 공식 전처리 재사용 + marker 좌표 오버라이드.

공식 `utils/datasets.py`의 `dataset_inference`가 하는 일을 그대로 따르되,
grounding uncertainty를 위해 **marker 위치만 오프셋으로 흔들 수 있게** 분리했다.

공식 코드와 다른 점은 딱 두 가지이고 둘 다 의도적이다:
  1. focal 정규화된 base 이미지를 이미지 단위로 캐시한다.
     (공식 코드는 픽셀마다 undistort+resize를 다시 한다. 한 이미지에서 픽셀 100개 ×
      generation 10.5회를 뽑는 이 실험에서는 같은 연산을 1000번 반복하게 된다.)
  2. `.convert("RGB")` 를 명시한다. putpixel((r,g,b)) 가 모드에 상관없이 동작해야 한다.

그 외 undistort / focal 정규화 / 좌표 rescale / 화살표 렌더링은 공식 구현을 그대로 쓴다.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

# 공식 리포의 헬퍼를 그대로 import 한다 (복사하지 않는다).
_THIRD_PARTY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party",
    "DepthLM_Official",
)
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

from utils.datasets import (  # noqa: E402
    generate_prompt_depth_sft,
    normalizing_focal_length,
    undistort_image,
)

# 공식 dataset_inference 와 동일한 값. 화살표 몸통 5px + arrowhead.
CROSS_SIZE = 5

# 12B(Pixtral) 전용. 3B/7B는 1000. 논문 Appendix F / eval.py 와 일치.
UNIFIED_FOCAL_LENGTH_12B = 750.0


@dataclass
class ImageContext:
    """한 이미지에 대해 focal 정규화까지 끝난 상태. 픽셀들이 공유한다."""

    image_id: str
    base_image: Image.Image  # marker 없는 정규화 이미지
    intrinsics_orig: list[float]  # [fx, fy, cx, cy] (원본)
    intrinsics_new: list[float]  # [fx, fy, cx, cy, W, H] (정규화 후)


@dataclass
class QuerySample:
    """(이미지, 질의 픽셀) 한 쌍."""

    image_id: str
    dataset: str
    pixel_index: int
    pixel_xy_orig: tuple[int, int]  # 원본 좌표계
    pixel_xy: tuple[int, int]  # 정규화 이미지 좌표계
    depth_gt: float  # 카메라 중심으로부터의 Euclidean 거리 (m)
    ctx: ImageContext = field(repr=False)


def draw_marker(image: Image.Image, x: int, y: int) -> Image.Image | None:
    """공식 `dataset_inference` Step 5와 동일한 빨간 화살표를 그린 **사본**을 돌려준다.

    화살표 몸통은 x-1..x-5 에 놓이고 끝(tip)이 (x, y)를 가리킨다.
    질의 픽셀 (x, y) 자체는 칠하지 않는다 — 공식 구현 그대로다.
    경계에 너무 가까워 그릴 수 없으면 None.
    """
    if not (
        CROSS_SIZE <= x < image.width - CROSS_SIZE
        and CROSS_SIZE <= y < image.height - CROSS_SIZE
    ):
        return None

    marked = image.copy()
    for dx in range(1, CROSS_SIZE + 1):
        marked.putpixel((x - dx, y), (255, 0, 0))  # 수평선
    for dy in range(1, CROSS_SIZE // 2 + 1):  # arrowhead
        marked.putpixel((x - dy - 1, y + dy), (255, 0, 0))
        marked.putpixel((x - dy - 1, y - dy), (255, 0, 0))
    return marked


def can_draw_marker(ctx: ImageContext, x: int, y: int) -> bool:
    return (
        CROSS_SIZE <= x < ctx.base_image.width - CROSS_SIZE
        and CROSS_SIZE <= y < ctx.base_image.height - CROSS_SIZE
    )


def build_problem_prompt() -> str:
    """공식 프롬프트 문자열. GT depth는 여기에 들어가지 않는다 (leakage 방지)."""
    problem, _thinking, _solution = generate_prompt_depth_sft(0.0, is_eval=True)
    return problem


class DepthLMJsonl:
    """iBims1 스키마 jsonl 로더. NYUv2 큐레이션 결과도 같은 스키마를 쓴다.

    스키마: {"image": str, "intrinsics": [fx,fy,cx,cy,W,H],
             "pixel_coords": [[x,y], ...], "depth": [float, ...]}
    """

    def __init__(
        self,
        json_path: str,
        image_folder: str,
        dataset_name: str,
        normalized_focal_length: float = UNIFIED_FOCAL_LENGTH_12B,
    ) -> None:
        self.json_path = json_path
        self.image_folder = image_folder
        self.dataset_name = dataset_name
        self.normalized_focal_length = normalized_focal_length

        # 공식 코드와 동일한 읽기 방식
        if json_path.endswith(".jsonl"):
            with open(json_path, "r") as f:
                content = f.read()
            self.records: list[dict[str, Any]] = pd.read_json(
                StringIO(content), lines=True
            ).to_dict(orient="records")
        else:
            self.records = json.load(open(json_path, "r"))

        self._ctx_cache: dict[int, ImageContext] = {}
        self._cache_order: list[int] = []
        self._cache_limit = 4  # 이미지 단위로 순회하므로 작게 유지

    def __len__(self) -> int:
        return len(self.records)

    def num_pixels(self, image_index: int) -> int:
        return len(self.records[image_index]["pixel_coords"])

    def image_id(self, image_index: int) -> str:
        return str(self.records[image_index]["image"])

    def get_context(self, image_index: int) -> ImageContext:
        if image_index in self._ctx_cache:
            return self._ctx_cache[image_index]

        rec = self.records[image_index]

        image = Image.open(
            os.path.join(self.image_folder, str(rec["image"]).lstrip("/"))
        ).convert("RGB")

        # 공식 코드의 intrinsic 오류 방어 로직 그대로
        intrinsics = list(rec["intrinsics"][:4])
        if intrinsics[0] == 0.0:
            intrinsics[0] = intrinsics[1]
        if intrinsics[1] == 0.0:
            intrinsics[1] = intrinsics[0]

        image, intrinsics_new = undistort_image(intrinsics, image)
        image, intrinsics_new = normalizing_focal_length(
            self.normalized_focal_length, intrinsics_new, image
        )
        image = image.convert("RGB")

        ctx = ImageContext(
            image_id=str(rec["image"]),
            base_image=image,
            intrinsics_orig=intrinsics,
            intrinsics_new=list(intrinsics_new),
        )

        self._ctx_cache[image_index] = ctx
        self._cache_order.append(image_index)
        while len(self._cache_order) > self._cache_limit:
            self._ctx_cache.pop(self._cache_order.pop(0), None)
        return ctx

    def scale_pixel(
        self, ctx: ImageContext, coord: list[int] | tuple[int, int]
    ) -> tuple[int, int]:
        """원본 좌표 → 정규화 이미지 좌표. 공식 코드 Step 3과 동일한 식."""
        io_, in_ = ctx.intrinsics_orig, ctx.intrinsics_new
        x = int((coord[0] - io_[2]) * (in_[0] / io_[0]) + in_[2])
        y = int((coord[1] - io_[3]) * (in_[1] / io_[1]) + in_[3])
        return x, y

    def get_sample(self, image_index: int, pixel_index: int) -> QuerySample | None:
        """marker를 그릴 수 없는 픽셀은 None (공식 코드도 이런 샘플을 버린다)."""
        ctx = self.get_context(image_index)
        rec = self.records[image_index]
        coord = rec["pixel_coords"][pixel_index]
        x, y = self.scale_pixel(ctx, coord)
        if not can_draw_marker(ctx, x, y):
            return None
        return QuerySample(
            image_id=ctx.image_id,
            dataset=self.dataset_name,
            pixel_index=pixel_index,
            pixel_xy_orig=(int(coord[0]), int(coord[1])),
            pixel_xy=(x, y),
            depth_gt=float(rec["depth"][pixel_index]),
            ctx=ctx,
        )

    def iter_samples(self):
        """모든 (이미지, 픽셀)을 이미지 순서대로 순회. 캐시 적중률이 높다."""
        for i in range(len(self.records)):
            for j in range(self.num_pixels(i)):
                s = self.get_sample(i, j)
                if s is not None:
                    yield s


# ---------------------------------------------------------------------------
# grounding perturbation 오프셋
# ---------------------------------------------------------------------------

def grounding_offsets(radius: int = 2) -> list[tuple[int, int]]:
    """반경 `radius` 의 3x3 격자 9개. 지시서 §5 명세.

        (x-r,y-r) (x,y-r) (x+r,y-r)
        (x-r,y)   (x,y)   (x+r,y)
        (x-r,y+r) (x,y+r) (x+r,y+r)

    중심 (0,0) 이 항상 첫 원소다 — 호출부가 gens[0] 을 deterministic 예측으로 쓴다.

    radius 는 본 수집에서 2로 고정한다. ±1/±3 민감도는 §5 권장대로 최종 분석에서
    subset 에 대해 따로 수행한다 (전체를 3개 반경으로 수집하지 않는다).
    """
    r = radius
    return [(0, 0)] + [
        (dx, dy)
        for dy in (-r, 0, r)
        for dx in (-r, 0, r)
        if not (dx == 0 and dy == 0)
    ]


def render_grounding_variants(
    sample: QuerySample, offsets: list[tuple[int, int]]
) -> tuple[list[Image.Image], list[tuple[int, int]]]:
    """오프셋별 marker 이미지. 경계 밖으로 나가는 오프셋은 조용히 건너뛴다."""
    images, used = [], []
    x0, y0 = sample.pixel_xy
    for dx, dy in offsets:
        img = draw_marker(sample.ctx.base_image, x0 + dx, y0 + dy)
        if img is not None:
            images.append(img)
            used.append((dx, dy))
    return images, used
