import numpy as np

from src.synthetic.mask_placement import (
    dino_structured_roi,
    placement_rng,
    placement_seed,
    transform_mask,
    weighted_schedule,
)


def test_weighted_schedule_preserves_and_interleaves_exact_quotas() -> None:
    schedule = weighted_schedule({0: 12, 1: 5})

    assert schedule.count(0) == 12
    assert schedule.count(1) == 5
    assert 1 in schedule[:4]
    assert 1 in schedule[-4:]


def test_placement_rng_is_sample_local_and_reproducible() -> None:
    first = placement_rng(20260727, "pcb1", 42).integers(0, 1_000_000, size=8)
    second = placement_rng(20260727, "pcb1", 42).integers(0, 1_000_000, size=8)
    other = placement_rng(20260727, "pcb1", 43).integers(0, 1_000_000, size=8)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert placement_seed(20260727, "pcb1", 42) == placement_seed(
        20260727,
        "pcb1",
        42,
    )
    assert placement_seed(20260727, "pcb1", 42) != placement_seed(
        20260727,
        "pcb1",
        43,
    )


def test_affine_transform_expands_canvas_without_clipping() -> None:
    source = np.zeros((7, 15), dtype=bool)
    source[1:6, 2:13] = True
    source[2:5, 6:9] = False

    transformed = transform_mask(
        source,
        rotation_deg=47.0,
        scale=3.2,
        flip=True,
    )

    assert transformed.any()
    assert transformed[0].any()
    assert transformed[-1].any()
    assert transformed[:, 0].any()
    assert transformed[:, -1].any()
    assert transformed.sum() > source.sum()


def test_dino_structured_roi_respects_reference_and_border() -> None:
    scores = np.zeros((8, 8), dtype=np.float32)
    scores[2:7, 2:7] = 0.7
    scores[3:6, 3:6] = 1.0
    reference = np.zeros((80, 80), dtype=bool)
    reference[15:70, 15:70] = True

    roi = dino_structured_roi(
        scores,
        (80, 80),
        reference_roi=reference,
        score_quantile=0.25,
        close_kernel=3,
        open_kernel=3,
        dilation_px=0,
        border_fraction=0.05,
        min_component_area_ratio=0.005,
    )

    assert roi.any()
    assert not roi[:4].any()
    assert not roi[-4:].any()
    assert not roi[:, :4].any()
    assert not roi[:, -4:].any()
    assert roi[40, 40]
