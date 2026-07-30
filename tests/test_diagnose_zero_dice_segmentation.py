from __future__ import annotations

import numpy as np
import pytest

from scripts.diagnose_zero_dice_segmentation import (
    INFORMATIVE_PIXEL_AUROC,
    DiagnosisError,
    classify,
    probability_statistics,
    render_markdown,
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


def _payload_run(
    *,
    group: str,
    seed: int,
    dice: float,
    max_probability: float,
    positive_pixels: int,
    pixel_auroc: float = 0.90,
) -> dict[str, object]:
    return {
        "run_name": f"m18_{group}_pcb1_seed{seed}",
        "object": "pcb1",
        "canonical_group": group,
        "threshold": 0.5,
        "verdict": "non_zero_dice" if dice > 0.0 else "positive_pixels_never_overlap_ground_truth",
        "published_metrics": {"dice": dice, "pixel_auroc": pixel_auroc},
        "recomputed_metrics": {"dice": dice, "pixel_auroc": pixel_auroc},
        "metric_deltas": {"dice": 0.0},
        "reproduced_published_metrics": True,
        "defect_exposure": {"real_share": 1.0},
        "probability": {
            "max_probability": max_probability,
            "pixels_at_or_above_threshold": positive_pixels,
            "max_probability_inside_ground_truth": max_probability / 2,
            "max_probability_outside_ground_truth": max_probability,
            "mean_probability_inside_ground_truth": 0.1,
            "mean_probability_outside_ground_truth": 0.1,
            "peak_inside_ground_truth": False,
            "images_with_any_pixel_at_or_above_threshold": 1 if positive_pixels else 0,
            "images_total": 10,
            "defect_images": 5,
            "percentiles": {},
        },
    }


def _payload(runs: list[dict[str, object]]) -> dict[str, object]:
    zero = [run for run in runs if run["published_metrics"]["dice"] == 0.0]
    return {
        "threshold": 0.5,
        "runs_total": len(runs),
        "zero_dice_runs": len(zero),
        "max_metric_delta": 1e-05,
        "runs": runs,
    }


def test_report_claims_the_ceiling_explains_everything_only_when_it_does() -> None:
    runs = [
        _payload_run(group="copypaste_only", seed=42, dice=0.0, max_probability=0.45, positive_pixels=0),
        _payload_run(group="diffusion_only", seed=43, dice=0.0, max_probability=0.27, positive_pixels=0),
        _payload_run(group="real_only", seed=42, dice=0.38, max_probability=0.96, positive_pixels=18_466),
    ]
    report = render_markdown(_payload(runs))
    assert "最高預測機率完全決定 Dice 是否退化" in report
    assert "2 / 2 個零 Dice run 連一個正像素都沒有" in report
    assert "與模型排序能力（pixel AUROC）無關" in report


def test_a_zero_dice_run_that_has_positive_pixels_breaks_the_ceiling_claim() -> None:
    # Regression: the summary used to assert "all below threshold" and merely fill in the
    # maximum, so a run whose positive pixels simply miss the target read as a ceiling case.
    runs = [
        _payload_run(group="copypaste_only", seed=42, dice=0.0, max_probability=0.45, positive_pixels=0),
        _payload_run(group="filtered_syn", seed=43, dice=0.0, max_probability=0.8187, positive_pixels=8_660),
        _payload_run(group="real_only", seed=42, dice=0.38, max_probability=0.96, positive_pixels=18_466),
    ]
    report = render_markdown(_payload(runs))
    assert "機率天花板只解釋其中一種" in report
    assert "1 / 2 個零 Dice run 連一個正像素都沒有" in report
    assert "完全沒有落在真實瑕疵上" in report
    assert "seed 43" in report
    # The refuted claim must not survive anywhere in the report.
    assert "完全由**機率天花板**決定" not in report
    # The reported ceiling must describe the ceiling-limited runs only, not the outlier.
    assert "0.8187`）" not in report.split("## 零 Dice run 的判定")[0].split("- 另外")[0]


def test_seed_is_recoverable_for_every_row() -> None:
    runs = [
        _payload_run(group="copypaste_only", seed=seed, dice=0.0, max_probability=0.4, positive_pixels=0)
        for seed in (42, 43, 44)
    ]
    runs.append(
        _payload_run(group="real_only", seed=42, dice=0.4, max_probability=0.9, positive_pixels=100)
    )
    report = render_markdown(_payload(runs))
    for seed in (42, 43, 44):
        assert f"/ seed {seed}" in report
