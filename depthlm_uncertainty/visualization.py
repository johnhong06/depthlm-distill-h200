"""figure 생성. figure 안 텍스트는 영문으로 통일한다 (NOTES D-10)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea  # noqa: E402
from PIL import Image  # noqa: E402

from .depthlm_data import DepthLMJsonl, draw_marker, grounding_offsets

BOX = dict(boxstyle="round,pad=0.45", fc="#eef3fb", ec="#5b7fb0", lw=1.3)
BOX_U = dict(boxstyle="round,pad=0.45", fc="#fff3e6", ec="#d98b3a", lw=1.3)
ARROW = dict(arrowstyle="-|>", color="#5b7fb0", lw=1.6,
             shrinkA=2, shrinkB=2, mutation_scale=14)


def _imshow(ax, img, title: str, fs: int = 8) -> None:
    ax.imshow(img)
    ax.set_title(title, fontsize=fs, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#5b7fb0")


def pipeline_figure(ds: DepthLMJsonl, image_index: int, pixel_index: int, path: str) -> None:
    """Figure 1 — 실험 파이프라인.

    윗줄은 DepthLM 원래 추론 경로, 아랫줄은 이 연구가 더한 세 uncertainty 분기.
    실제 데이터셋 이미지와 실제 marker 렌더링을 그대로 쓴다 (모형 그림이 아니다).
    """
    s = ds.get_sample(image_index, pixel_index)
    if s is None:
        raise ValueError("marker를 그릴 수 없는 픽셀")
    ctx = s.ctx
    x, y = s.pixel_xy
    raw = Image.open(f"{ds.image_folder}/{ctx.image_id.lstrip('/')}").convert("RGB")
    marked = draw_marker(ctx.base_image, x, y)
    H = 42  # zoom 반경

    fig = plt.figure(figsize=(13.0, 8.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.34], hspace=0.50, wspace=0.28,
                          left=0.04, right=0.97, top=0.905, bottom=0.035)
    sub = gs[1, :].subgridspec(1, 3, wspace=0.28)

    # ---- 윗줄: DepthLM 추론 경로 -----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    _imshow(ax, raw, f"1. Input RGB\n{raw.size[0]}x{raw.size[1]}, "
                     f"$f_x$={ctx.intrinsics_orig[0]:.0f} px")

    ax = fig.add_subplot(gs[0, 1])
    _imshow(ax, marked, f"2. Focal-normalized + marker\n"
                        f"{marked.size[0]}x{marked.size[1]}, $f_{{uni}}$=750 px")
    ax.add_patch(mpatches.Rectangle((x - H, y - H), 2 * H, 2 * H,
                                    ec="#d98b3a", fc="none", lw=1.4))

    ax = fig.add_subplot(gs[0, 2])
    _imshow(ax, marked.crop((x - H, y - H, x + H, y + H)).resize((320, 320), Image.NEAREST),
            "3. Query pixel (zoom)\nred arrow tip = queried pixel")
    for sp in ax.spines.values():
        sp.set_edgecolor("#d98b3a")

    ax_model = ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    ax.text(0.5, 0.90, "4. DepthLM  (Pixtral-12B)", ha="center", va="top",
            fontsize=9.5, weight="bold", bbox=BOX, transform=ax.transAxes)
    ax.text(0.5, 0.62,
            '"Given this image, how far is the point\n'
            'pointed by the red arrow from the camera?"',
            ha="center", va="top", fontsize=7.4, style="italic", transform=ax.transAxes)
    ax.text(0.5, 0.36,
            "<think> The point is around\n"
            "$\\bf{X}$ meters away from the camera. </think>\n"
            "<answer> $\\bf{X}$ </answer>",
            ha="center", va="top", fontsize=7.6, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#999"),
            transform=ax.transAxes)
    ax.text(0.5, 0.06, "no dense prediction head\nno regression loss",
            ha="center", va="bottom", fontsize=7.2, color="#666", transform=ax.transAxes)

    # ---- 가운뎃줄: 세 uncertainty 분기 ------------------------------------
    # A. token
    ax_a = ax = fig.add_subplot(sub[0, 0])
    ax.axis("off")
    ax.text(0.5, 1.02, "A.  $U_{token}$   language generation", ha="center", va="top",
            fontsize=9.5, weight="bold", bbox=BOX_U, transform=ax.transAxes)
    # 토큰 박스는 HPacker에 맡긴다. 문자 폭을 직접 추정하면 폰트·축 크기에 따라
    # 겹치거나 잘린다 (실제로 첫 버전에서 그랬다).
    toks = ["<answer>", "12", ".", "37", "</answer>"]
    hot = [False, True, True, True, False]
    boxes = [TextArea(t, textprops=dict(
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="square,pad=0.22", fc="#f6b26b" if h else "#ededed", ec="none")))
        for t, h in zip(toks, hot)]
    ax.add_artist(AnchoredOffsetbox(
        loc="center", child=HPacker(children=boxes, align="center", pad=0, sep=4),
        pad=0, frameon=False, bbox_to_anchor=(0.5, 0.70),
        bbox_transform=ax.transAxes))
    ax.text(0.5, 0.53, "logits of the numeric tokens only",
            ha="center", va="top", fontsize=7.6, color="#666", transform=ax.transAxes)
    ax.text(0.5, 0.34, "mean NLL  ·  entropy\nmin prob  ·  seq logprob",
            ha="center", va="top", fontsize=8.2, transform=ax.transAxes)
    ax.text(0.5, 0.02, "1 greedy generation", ha="center", va="bottom",
            fontsize=7.4, color="#d98b3a", transform=ax.transAxes)

    # B. grounding — 9개 오프셋을 실제 crop 위에 표시
    ax_b = ax = fig.add_subplot(sub[0, 1])
    R = 14
    crop = np.array(ctx.base_image.crop((x - R, y - R, x + R, y + R))
                    .resize((420, 420), Image.NEAREST))
    ax.imshow(crop, extent=[-R, R, R, -R])
    offs = grounding_offsets()
    for dx, dy in offs[1:]:
        ax.plot(dx, dy, marker="x", color="#d94545", ms=8, mew=2.0)
    ax.plot(0, 0, marker="x", color="#ffffff", ms=10, mew=2.6)
    ax.plot(0, 0, marker="x", color="#d94545", ms=9, mew=1.8)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#d98b3a")
        sp.set_linewidth(1.3)
    # 캡션을 축 아래로 내리면 아래쪽 화살표와 겹친다. 제목/xlabel 로 올려 붙인다.
    ax.set_title("B.  $U_{grounding}$   pixel grounding\n"
                 "9 marker offsets  →  dispersion of $\\hat{d}$",
                 fontsize=9.5, weight="bold", pad=6)
    ax.set_xlabel("8 extra generations  (center reused)", fontsize=7.4, color="#d98b3a")

    # C. stochastic
    ax_c = ax = fig.add_subplot(sub[0, 2])
    ax.axis("off")
    ax.text(0.5, 1.02, "C.  $U_{stochastic}$   output variability", ha="center", va="top",
            fontsize=9.5, weight="bold", bbox=BOX_U, transform=ax.transAxes)
    ax.text(0.5, 0.78, "same image + marker + prompt\n$T$ = 1.0,  $M$ = 10 samples",
            ha="center", va="top", fontsize=8, transform=ax.transAxes)
    vals = [12.1, 12.3, 12.0, 12.2, 13.8]
    ax.text(0.5, 0.52, "  ".join(f"{v:.1f}" for v in vals) + "  ...",
            ha="center", va="top", fontsize=8.6, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#999"),
            transform=ax.transAxes)
    ax.text(0.5, 0.32, "std  ·  MAD  ·  IQR  ·  CoV",
            ha="center", va="top", fontsize=8.2, transform=ax.transAxes)
    ax.text(0.5, 0.02, "1 prefill + $M$ decodes", ha="center", va="bottom",
            fontsize=7.4, color="#d98b3a", transform=ax.transAxes)

    # ---- 아랫줄: 최종 질문 (전폭) -----------------------------------------
    ax_foot = ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    # 라벨을 가운데 위에 두면 위에서 내려오는 화살표와 부딪힌다. 왼쪽으로 뺀다.
    ax.text(0.035, 0.44, "Research\nquestion", ha="left", va="center",
            fontsize=10, weight="bold", bbox=BOX, transform=ax.transAxes)
    ax.text(0.33, 0.44, "$E_i = |\\hat{d}_i - d_i^{GT}|$", ha="center", va="center",
            fontsize=13, transform=ax.transAxes)
    ax.text(0.52, 0.44, "$U_i \\;\\longrightarrow\\; E_i$  ?", ha="center", va="center",
            fontsize=13, transform=ax.transAxes)
    ax.text(0.79, 0.44,
            "Spearman  ·  partial Spearman (GT depth controlled)\n"
            "binning  ·  risk-coverage  ·  calibration",
            ha="center", va="center", fontsize=8.2, transform=ax.transAxes)
    ax.text(0.5, -0.02, "GT depth is used ONLY to compute $E_i$ — never in the prediction path",
            ha="center", va="bottom", fontsize=8, color="#c0392b", weight="bold",
            transform=ax.transAxes)

    # ---- 화살표: 좌표를 손으로 찍지 않고 축 위치에서 계산한다 ---------------
    fig.canvas.draw()  # get_position() 이 확정되도록
    top = [fig.axes[i].get_position() for i in range(4)]
    pa, pb, pc = ax_a.get_position(), ax_b.get_position(), ax_c.get_position()
    pm, pf = ax_model.get_position(), ax_foot.get_position()

    # 윗줄 가로 흐름
    ymid = (top[0].y0 + top[0].y1) / 2
    for left, right in zip(top[:-1], top[1:]):
        fig.add_artist(mpatches.FancyArrowPatch(
            (left.x1 + 0.006, ymid), (right.x0 - 0.006, ymid),
            transform=fig.transFigure, **ARROW))

    # 모델 → 분기 버스 → A/B/C.  A/C 헤더 박스와 B 2줄 제목이 축 위로 튀어나오므로
    # 화살표 끝을 축 상단보다 충분히 위에서 멈춘다.
    head_y = pa.y1 + 0.058
    bus_y = head_y + 0.040
    xs = [(p.x0 + p.x1) / 2 for p in (pa, pb, pc)]
    xm = (pm.x0 + pm.x1) / 2
    fig.add_artist(plt.Line2D([xm, xm], [pm.y0, bus_y], transform=fig.transFigure,
                              color="#d98b3a", lw=1.5))
    fig.add_artist(plt.Line2D([min(xs), max(xm, max(xs))], [bus_y, bus_y],
                              transform=fig.transFigure, color="#d98b3a", lw=1.5))
    for xc in xs:
        fig.add_artist(mpatches.FancyArrowPatch(
            (xc, bus_y), (xc, head_y), transform=fig.transFigure,
            arrowstyle="-|>", color="#d98b3a", lw=1.5, mutation_scale=13))

    # A/B/C → 최종 질문 (B는 xlabel이 축 아래에 있으므로 조금 더 아래에서 시작)
    for xc, p in zip(xs, (pa, pb, pc)):
        y_start = p.y0 - (0.032 if p is pb else 0.012)
        fig.add_artist(mpatches.FancyArrowPatch(
            (xc, y_start), (xc, pf.y1 + 0.004), transform=fig.transFigure,
            arrowstyle="-|>", color="#d98b3a", lw=1.5, mutation_scale=13))

    fig.suptitle("Figure 1.  Uncertainty estimation on top of DepthLM's text-based metric depth prediction",
                 fontsize=12, y=0.975)
    fig.savefig(path, dpi=160)
    plt.close(fig)
