"""Pure, deterministic M14 embedding and image-feature quality metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class QualityMetricError(ValueError):
    """A quality metric received an invalid or undersized feature matrix."""


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """One real/generated group comparison."""

    n_real: int
    n_generated: int
    nn_mean: float
    nn_median: float
    nn_p05: float
    nn_p95: float
    mnn_score: float
    kid: float
    fid: float


def _finite_matrix(values: np.ndarray, *, name: str, minimum: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) < minimum or not np.isfinite(array).all():
        raise QualityMetricError(
            f"{name} must be a finite matrix with at least {minimum} rows"
        )
    return array


def l2_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize rows for cosine correspondence metrics."""

    array = _finite_matrix(values, name="embeddings")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise QualityMetricError("Embedding contains a zero vector")
    return array / norms


def nearest_neighbor_scores(
    generated: np.ndarray,
    real: np.ndarray,
) -> np.ndarray:
    """Return each generated embedding's maximum cosine to a real embedding."""

    generated_normalized = l2_normalize(generated)
    real_normalized = l2_normalize(real)
    if generated_normalized.shape[1] != real_normalized.shape[1]:
        raise QualityMetricError("Generated and real embedding dimensions differ")
    return (generated_normalized @ real_normalized.T).max(axis=1)


def mutual_nearest_neighbor_score(
    generated: np.ndarray,
    real: np.ndarray,
) -> float:
    """Return the fraction of real references in a mutual-NN pair."""

    generated_normalized = l2_normalize(generated)
    real_normalized = l2_normalize(real)
    if generated_normalized.shape[1] != real_normalized.shape[1]:
        raise QualityMetricError("Generated and real embedding dimensions differ")
    similarities = generated_normalized @ real_normalized.T
    nearest_real_for_generated = np.argmax(similarities, axis=1)
    nearest_generated_for_real = np.argmax(similarities, axis=0)
    mutual = [
        nearest_real_for_generated[generated_index] == real_index
        for real_index, generated_index in enumerate(nearest_generated_for_real)
    ]
    return float(np.mean(mutual))


def polynomial_kid(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return the deterministic unbiased degree-3 polynomial MMD estimate."""

    first = _finite_matrix(first, name="first features", minimum=2)
    second = _finite_matrix(second, name="second features", minimum=2)
    if first.shape[1] != second.shape[1]:
        raise QualityMetricError("KID feature dimensions differ")
    dimension = first.shape[1]
    first_kernel = (first @ first.T / dimension + 1.0) ** 3
    second_kernel = (second @ second.T / dimension + 1.0) ** 3
    cross_kernel = (first @ second.T / dimension + 1.0) ** 3
    first_term = (
        first_kernel.sum() - np.trace(first_kernel)
    ) / (len(first) * (len(first) - 1))
    second_term = (
        second_kernel.sum() - np.trace(second_kernel)
    ) / (len(second) * (len(second) - 1))
    return float(first_term + second_term - 2.0 * cross_kernel.mean())


def polynomial_kid_biased(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return the biased polynomial MMD estimate used by identity sanity checks.

    Unlike unbiased KID, this estimator includes each kernel diagonal.  It is
    therefore exactly zero when a finite feature set is compared with itself,
    which makes it suitable for checking the implementation and feature
    alignment.  Formal generated-vs-real rows continue to use unbiased KID.
    """

    first = _finite_matrix(first, name="first features")
    second = _finite_matrix(second, name="second features")
    if first.shape[1] != second.shape[1]:
        raise QualityMetricError("KID feature dimensions differ")
    dimension = first.shape[1]
    first_kernel = (first @ first.T / dimension + 1.0) ** 3
    second_kernel = (second @ second.T / dimension + 1.0) ** 3
    cross_kernel = (first @ second.T / dimension + 1.0) ** 3
    return float(
        first_kernel.mean() + second_kernel.mean() - 2.0 * cross_kernel.mean()
    )


def low_rank_fid(first: np.ndarray, second: np.ndarray) -> float:
    """Compute exact FID from clean-fid features without 2048x2048 sqrtm.

    If centered feature factors are ``A`` and ``B``, then
    ``tr(sqrt(C1 C2))`` equals the nuclear norm of ``A.T @ B``. This is
    mathematically equivalent to the usual covariance formula and is stable
    when the real group has only a handful of examples.
    """

    first = _finite_matrix(first, name="first features", minimum=2)
    second = _finite_matrix(second, name="second features", minimum=2)
    if first.shape[1] != second.shape[1]:
        raise QualityMetricError("FID feature dimensions differ")
    first_mean = first.mean(axis=0)
    second_mean = second.mean(axis=0)
    first_factor = (first - first_mean).T / math.sqrt(len(first) - 1)
    second_factor = (second - second_mean).T / math.sqrt(len(second) - 1)
    cross = first_factor.T @ second_factor
    covariance_overlap = float(np.linalg.svd(cross, compute_uv=False).sum())
    value = float(
        np.square(first_mean - second_mean).sum()
        + np.square(first_factor).sum()
        + np.square(second_factor).sum()
        - 2.0 * covariance_overlap
    )
    if value < -1e-7:
        raise QualityMetricError(f"FID became materially negative: {value}")
    return max(0.0, value)


def evaluate_quality(
    generated_embeddings: np.ndarray,
    real_embeddings: np.ndarray,
    generated_features: np.ndarray,
    real_features: np.ndarray,
) -> QualityMetrics:
    """Compute all four locked M14 metrics for one non-empty group."""

    if len(generated_embeddings) != len(generated_features):
        raise QualityMetricError("Generated embedding/feature counts differ")
    if len(real_embeddings) != len(real_features):
        raise QualityMetricError("Real embedding/feature counts differ")
    nn_scores = nearest_neighbor_scores(generated_embeddings, real_embeddings)
    return QualityMetrics(
        n_real=len(real_embeddings),
        n_generated=len(generated_embeddings),
        nn_mean=float(nn_scores.mean()),
        nn_median=float(np.median(nn_scores)),
        nn_p05=float(np.quantile(nn_scores, 0.05)),
        nn_p95=float(np.quantile(nn_scores, 0.95)),
        mnn_score=mutual_nearest_neighbor_score(
            generated_embeddings,
            real_embeddings,
        ),
        kid=polynomial_kid(generated_features, real_features),
        fid=low_rank_fid(generated_features, real_features),
    )
