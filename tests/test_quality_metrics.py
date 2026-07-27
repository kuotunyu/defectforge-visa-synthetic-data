import numpy as np
import pytest
from scipy import linalg

from src.evaluation.quality_metrics import (
    QualityMetricError,
    evaluate_quality,
    low_rank_fid,
    mutual_nearest_neighbor_score,
    nearest_neighbor_scores,
    polynomial_kid,
    polynomial_kid_biased,
)


def test_correspondence_metrics_match_exact_mutual_pairs() -> None:
    real = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    generated = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    assert nearest_neighbor_scores(generated, real).tolist() == pytest.approx(
        [0.9938837, 0.9938837, 0.9701425]
    )
    assert mutual_nearest_neighbor_score(generated, real) == 1.0


def test_mnn_exposes_mode_collapse() -> None:
    real = np.eye(3)
    generated = np.asarray([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]])
    assert mutual_nearest_neighbor_score(generated, real) == pytest.approx(1 / 3)


def test_kid_is_symmetric_and_near_self_distribution() -> None:
    rng = np.random.default_rng(42)
    first = rng.normal(size=(40, 16))
    second = rng.normal(loc=2.0, size=(40, 16))
    self_kid = polynomial_kid(first, first)
    shifted = polynomial_kid(first, second)
    assert polynomial_kid(second, first) == pytest.approx(shifted)
    assert shifted > self_kid


def test_biased_kid_is_an_exact_finite_sample_identity_check() -> None:
    rng = np.random.default_rng(43)
    first = rng.normal(size=(3, 16))
    second = rng.normal(loc=2.0, size=(3, 16))
    assert polynomial_kid_biased(first, first) == pytest.approx(0.0, abs=1e-12)
    assert polynomial_kid_biased(first, second) > 0.0


def test_low_rank_fid_matches_standard_formula_in_small_dimension() -> None:
    rng = np.random.default_rng(7)
    first = rng.normal(size=(9, 5))
    second = rng.normal(loc=0.5, size=(11, 5))
    first_covariance = np.cov(first, rowvar=False)
    second_covariance = np.cov(second, rowvar=False)
    covariance_mean = linalg.sqrtm(first_covariance @ second_covariance)
    expected = float(
        np.square(first.mean(axis=0) - second.mean(axis=0)).sum()
        + np.trace(first_covariance)
        + np.trace(second_covariance)
        - 2 * np.trace(covariance_mean).real
    )
    assert low_rank_fid(first, second) == pytest.approx(expected, rel=1e-7, abs=1e-7)
    assert low_rank_fid(first, first) == pytest.approx(0.0, abs=1e-8)


def test_evaluate_quality_rejects_cross_feature_count_mismatch() -> None:
    with pytest.raises(QualityMetricError, match="counts differ"):
        evaluate_quality(
            np.ones((3, 4)),
            np.ones((2, 4)),
            np.ones((2, 8)),
            np.ones((2, 8)),
        )
