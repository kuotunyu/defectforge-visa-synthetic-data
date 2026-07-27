import numpy as np
import pytest

from src.filtering.embeddings import (
    EmbeddingError,
    calibrate_references,
    l2_normalize,
    semantic_scores,
)


def test_calibration_uses_leave_one_out_and_centroid_distance() -> None:
    references = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )
    calibration = calibrate_references(
        references,
        tau_low_quantile=0.05,
        outlier_quantile=0.95,
    )
    nearest, outlier = semantic_scores(references[:1], calibration)
    assert nearest[0] == pytest.approx(1.0)
    assert outlier[0] >= 0
    assert -1 <= calibration.tau_low <= 1
    assert calibration.tau_outlier >= 0


def test_l2_normalize_rejects_zero_and_invalid_shapes() -> None:
    with pytest.raises(EmbeddingError, match="zero vector"):
        l2_normalize(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(EmbeddingError, match="matrix"):
        l2_normalize(np.ones(3, dtype=np.float32))


def test_semantic_scores_reject_dimension_mismatch() -> None:
    calibration = calibrate_references(
        np.eye(3, dtype=np.float32),
        tau_low_quantile=0.05,
        outlier_quantile=0.95,
    )
    with pytest.raises(EmbeddingError, match="dimensions"):
        semantic_scores(np.ones((1, 4), dtype=np.float32), calibration)
