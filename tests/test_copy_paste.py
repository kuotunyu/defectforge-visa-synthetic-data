import numpy as np

from src.common.imaging import (
    detect_legal_roi,
    place_on_legal_roi,
)
from src.synthetic.copy_paste import allocate_quotas, sample_rng
from src.synthetic.procedural import (
    base_motif,
    even_schedule,
    generate_local_mask,
    load_shape_bounds,
)


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


def test_procedural_schedule_is_exact_and_balanced() -> None:
    schedule = even_schedule(("perlin", "crack", "scratch", "spot"), 10)
    assert schedule.count("perlin") == 3
    assert schedule.count("crack") == 3
    assert schedule.count("scratch") == 2
    assert schedule.count("spot") == 2


def test_all_procedural_shapes_satisfy_requested_bounds() -> None:
    bounds = {
        "area_ratio": (0.002, 0.01),
        "aspect_ratio": (0.7, 2.0),
    }
    for index, shape in enumerate(("perlin", "crack", "scratch", "spot")):
        rng = np.random.Generator(np.random.PCG64(index))
        assert base_motif(shape, rng).any()
        mask = generate_local_mask(
            shape,
            rng,
            image_shape=(300, 400),
            bounds=bounds,
        )
        area_ratio = mask.sum() / (300 * 400)
        ys, xs = np.nonzero(mask)
        aspect_ratio = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
        assert bounds["area_ratio"][0] <= area_ratio <= bounds["area_ratio"][1]
        assert bounds["aspect_ratio"][0] <= aspect_ratio <= bounds["aspect_ratio"][1]


def test_no_real_stats_branch_does_not_call_json_loader(monkeypatch) -> None:
    class DummyPaths:
        reports = None
        splits = None

    def forbidden_loader(_path):
        raise AssertionError("real_mask_stats.json must not be opened")

    monkeypatch.setattr("src.synthetic.procedural.load_json", forbidden_loader)
    config = {
        "procedural": {
            "fixed_bounds": {
                "pcb1": {
                    "area_ratio": [0.001, 0.03],
                    "aspect_ratio": [0.75, 4.0],
                }
            }
        }
    }
    observed = load_shape_bounds(
        paths=DummyPaths(),
        config=config,
        objects=("pcb1",),
        no_real_stats=True,
        manifest_sha256="unused",
    )
    assert observed["pcb1"]["area_ratio"] == (0.001, 0.03)
