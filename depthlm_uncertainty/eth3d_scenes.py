"""ETH3D 고해상도 다시점 training 장면의 실내/실외 분류.

출처: T. Schöps et al., "A Multi-view Stereo Benchmark with High-Resolution Images and Multi-Camera Videos," CVPR 2017,
Figure 6 캡션 — 장면 이름을 indoor(회색) / outdoor(청록)으로 색 구분. 2026-09-18 원문 PDF 에서 확인.
"""
OUTDOOR = {"courtyard", "electro", "facade", "meadow", "playground", "terrace"}
INDOOR = {"delivery_area", "kicker", "office", "pipes", "relief", "relief_2", "terrains"}

def scene_of(image_id: str) -> str:
    """'rgb/<scene>_DSC_0286.jpg' → '<scene>'"""
    return image_id.split("/")[-1].rsplit("_DSC", 1)[0]

def domain_of(image_id: str) -> str:
    s = scene_of(image_id)
    if s in OUTDOOR: return "outdoor"
    if s in INDOOR: return "indoor"
    raise KeyError(s)
