from copy import deepcopy

import pytest

from src.filtering.rules import (
    RejectReason,
    minimum_phash_distance,
    rejection_reasons,
)


def config() -> dict[str, object]:
    return {
        "rules": {
            "roi": {"minimum_containment": 1.0},
            "area": {
                "maximum_abs_zscore": 2.5,
                "enforce_real_quantiles": True,
            },
            "aspect": {"maximum_abs_zscore": 2.5},
            "phash": {"minimum_hamming_distance": 5},
            "dinov2": {"tau_copy": 0.98},
            "seam": {"minimum_score": 0.85},
        }
    }


def passing_scores() -> dict[str, float | int | None]:
    return {
        "roi_containment": 1.0,
        "area_ratio": 0.02,
        "area_p05": 0.01,
        "area_p95": 0.08,
        "area_zscore": 0.1,
        "aspect_zscore": -0.2,
        "phash_min_dist": None,
        "nn_score": 0.8,
        "tau_low": 0.6,
        "outlier_score": 1.0,
        "tau_outlier": 2.0,
        "seam_score": 0.9,
    }


def test_rejection_reasons_preserve_locked_funnel_order() -> None:
    scores = passing_scores()
    scores.update(
        {
            "roi_containment": 0.9,
            "area_ratio": 0.1,
            "area_zscore": 3.0,
            "aspect_zscore": -3.0,
            "phash_min_dist": 2,
            "nn_score": 0.99,
            "outlier_score": 3.0,
            "seam_score": 0.5,
        }
    )

    assert rejection_reasons(scores, config=config()) == [
        RejectReason.ROI_OVERFLOW,
        RejectReason.AREA_OUT_OF_RANGE,
        RejectReason.ASPECT_OUT_OF_RANGE,
        RejectReason.PHASH_DUPLICATE,
        RejectReason.NN_TOO_HIGH_COPY,
        RejectReason.EMBEDDING_OUTLIER,
        RejectReason.SEAM_POOR,
    ]


def test_disabled_rule_removes_only_its_reason() -> None:
    scores = passing_scores()
    scores["seam_score"] = 0.5

    assert rejection_reasons(scores, config=config(), disabled={"seam"}) == []
    with pytest.raises(ValueError, match="Unknown disabled"):
        rejection_reasons(scores, config=config(), disabled={"unknown"})


def test_area_quantile_guard_can_be_disabled_without_disabling_zscore() -> None:
    scores = passing_scores()
    scores["area_ratio"] = 0.09
    assert rejection_reasons(scores, config=config()) == [RejectReason.AREA_OUT_OF_RANGE]

    adjusted = deepcopy(config())
    adjusted["rules"]["area"]["enforce_real_quantiles"] = False
    assert rejection_reasons(scores, config=adjusted) == []


def test_minimum_phash_distance_handles_first_and_nearest() -> None:
    assert minimum_phash_distance("0000000000000000", []) is None
    assert minimum_phash_distance(
        "0000000000000000",
        ["0000000000000000", "ffffffffffffffff"],
    ) == 0
