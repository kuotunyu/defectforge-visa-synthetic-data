import numpy as np
import pytest

from src.filtering.metrics import (
    FilterMetricError,
    GeometryReference,
    context_bbox,
    crop_phash,
    fitted_zscore,
    geometry_scores,
    phash_distance,
    roi_containment,
    seam_score,
)


def reference() -> GeometryReference:
    return GeometryReference(
        area_values=(0.01, 0.02, 0.04, 0.08),
        area_p05=0.01,
        area_p95=0.08,
        aspect_values=(0.5, 1.0, 2.0, 4.0),
    )


def test_geometry_scores_use_log_reference_distributions() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 3:8] = 255

    scores = geometry_scores(mask, reference())

    assert scores["area_ratio"] == pytest.approx(0.2)
    assert scores["aspect_ratio"] == pytest.approx(1.25)
    assert scores["area_zscore"] == pytest.approx(
        fitted_zscore(0.2, reference().area_values, log_transform=True)
    )
    assert np.isfinite(scores["aspect_zscore"])


def test_roi_containment_is_exact_fraction() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    roi = np.ones((8, 8), dtype=bool)
    roi[2, 2] = False

    assert roi_containment(mask, roi) == pytest.approx(0.75)
    with pytest.raises(FilterMetricError, match="shapes differ"):
        roi_containment(mask, roi[:4])


def test_context_bbox_is_square_clipped_and_deterministic() -> None:
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[0:4, 0:6] = 255

    first = context_bbox(mask, ratio=2.5)
    second = context_bbox(mask, ratio=2.5)

    assert first == second == (0, 0, 15, 15)


def test_crop_phash_distance_detects_identical_and_changed_context() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:] = 255
    changed = image.copy()
    changed[8:24, 8:24] = [255, 0, 0]
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255

    baseline_hash = crop_phash(image, mask, ratio=1.5, hash_size=8)
    repeated_hash = crop_phash(image, mask, ratio=1.5, hash_size=8)
    changed_hash = crop_phash(changed, mask, ratio=1.5, hash_size=8)

    assert phash_distance(baseline_hash, repeated_hash) == 0
    assert phash_distance(baseline_hash, changed_hash) > 0


def test_seam_score_penalizes_outer_gradient_discontinuity() -> None:
    background = np.full((64, 64, 3), 100, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:40, 24:40] = 255
    smooth = background.copy()
    smooth[24:40, 24:40] = 120
    hard_seam = smooth.copy()
    hard_seam[21:24, 21:43] = 255
    hard_seam[40:43, 21:43] = 0

    smooth_score = seam_score(
        smooth,
        background,
        mask,
        band_px=5,
        histogram_bins=32,
    )
    hard_score = seam_score(
        hard_seam,
        background,
        mask,
        band_px=5,
        histogram_bins=32,
    )

    assert smooth_score > 0.95
    assert hard_score < smooth_score
