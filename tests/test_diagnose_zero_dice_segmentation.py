from __future__ import annotations

import numpy as np
import pytest

from scripts.diagnose_zero_dice_segmentation import (
    INFORMATIVE_PIXEL_AUROC,
    DiagnosisError,
    classify,
    probability_statistics,
    samples_from_manifest,
)

_RECORD = {
    "sample_id": "pcb1/test/000",
    "object_name": "pcb1",
    "kind": "real",
    "source_name": "real_test",
    "root": "visa_raw",
    "image_path": "pcb1/Data/Images/Anomaly/000.JPG",
    "image_sha256": "a" * 64,
    "mask_path": "pcb1/Data/Masks/Anomaly/000.png",
    "mask_sha256": "b" * 64,
    "has_defect": True,
    "manifest_refs": ["pcb1/Data/Images/Anomaly/000.JPG"],
    "defect_type": None,
}


def test_underconfident_peak_inside_ground_truth_is_named_separately() -> None:
    """pcb1/copypaste_only: the most confident pixel on the whole test set is a defect."""
    verdict = classify(
        dice=0.0,
        max_probability=0.4530,
        pixel_auroc=0.9015,
        threshold=0.5,
        peak_inside_ground_truth=True,
    )
    assert verdict == "underconfident_peak_inside_ground_truth"


def test_peak_outside_ground_truth_splits_on_ranking_quality() -> None:
    """Both peak below the threshold, so only the ranking separates them."""
    informative = classify(
        dice=0.0,
        max_probability=0.3157,
        pixel_auroc=INFORMATIVE_PIXEL_AUROC,
        threshold=0.5,
        peak_inside_ground_truth=False,
    )
    near_random = classify(
        dice=0.0,
        max_probability=0.3014,
        pixel_auroc=INFORMATIVE_PIXEL_AUROC - 0.01,
        threshold=0.5,
        peak_inside_ground_truth=False,
    )
    assert informative == "underconfident_peak_outside_ground_truth_ranking_informative"
    assert near_random == "underconfident_peak_outside_ground_truth_ranking_near_random"


def test_positive_pixels_that_never_overlap_are_not_a_confidence_problem() -> None:
    verdict = classify(
        dice=0.0,
        max_probability=0.99,
        pixel_auroc=0.9,
        threshold=0.5,
        peak_inside_ground_truth=False,
    )
    assert verdict == "positive_pixels_never_overlap_ground_truth"


def test_non_zero_dice_is_not_diagnosed() -> None:
    verdict = classify(
        dice=0.01,
        max_probability=0.1,
        pixel_auroc=0.1,
        threshold=0.5,
        peak_inside_ground_truth=None,
    )
    assert verdict == "non_zero_dice"


def test_probability_statistics_locate_the_peak_relative_to_ground_truth() -> None:
    probabilities = np.zeros((2, 2, 2), dtype=np.float32)
    masks = np.zeros((2, 2, 2), dtype=bool)
    masks[0, 0, 0] = True
    probabilities[0, 0, 0] = 0.45  # the global peak sits inside the defect
    probabilities[0, 1, 1] = 0.30
    probabilities[1, 0, 0] = 0.20

    statistics = probability_statistics(probabilities, masks, threshold=0.5)

    assert statistics["max_probability"] == pytest.approx(0.45)
    assert statistics["pixels_at_or_above_threshold"] == 0
    assert statistics["images_with_any_pixel_at_or_above_threshold"] == 0
    assert statistics["images_total"] == 2
    assert statistics["defect_images"] == 1
    assert statistics["max_probability_inside_ground_truth"] == pytest.approx(0.45)
    assert statistics["max_probability_outside_ground_truth"] == pytest.approx(0.30)


def test_probability_statistics_count_positive_pixels_when_threshold_is_reached() -> None:
    probabilities = np.full((1, 2, 2), 0.6, dtype=np.float32)
    masks = np.zeros((1, 2, 2), dtype=bool)

    statistics = probability_statistics(probabilities, masks, threshold=0.5)

    assert statistics["pixels_at_or_above_threshold"] == 4
    assert statistics["images_with_any_pixel_at_or_above_threshold"] == 1
    assert statistics["max_probability_inside_ground_truth"] is None


def test_manifest_records_must_be_real() -> None:
    assert len(samples_from_manifest([_RECORD])) == 1

    synthetic = {**_RECORD, "kind": "synthetic"}
    with pytest.raises(DiagnosisError, match="non-real record"):
        samples_from_manifest([synthetic])
