import numpy as np

from src.common.imaging import (
    detect_legal_roi,
    place_on_legal_roi,
)
from src.synthetic.copy_paste import allocate_quotas, sample_rng


def test_allocate_quotas_is_exact_and_proportional() -> None:
    assert allocate_quotas({0: 16, 1: 7}, 500) == {0: 348, 1: 152}
    assert allocate_quotas({0: 9, 1: 3}, 500) == {0: 375, 1: 125}


def test_placement_is_deterministic_and_contained() -> None:
    image = np.full((100, 120, 3), 30, dtype=np.uint8)
    image[20:80, 25:95] = (20, 170, 80)
    roi = detect_legal_roi(
        image,
        min_component_area_ratio=0.001,
        close_kernel=5,
        open_kernel=3,
        erosion_px=1,
        border_fraction=0.05,
    )
    component = np.ones((8, 10), dtype=bool)
    first = place_on_legal_roi(component, roi, sample_rng(42, "pcb1", 0), max_tries=50)
    second = place_on_legal_roi(component, roi, sample_rng(42, "pcb1", 0), max_tries=50)

    assert first == second
    x0, y0 = first
    assert np.all(roi[y0 : y0 + 8, x0 : x0 + 10][component])
