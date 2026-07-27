"""Pure M13 filtering metrics with no model or filesystem side effects."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import imagehash
import numpy as np
from PIL import Image


class FilterMetricError(ValueError):
    """A filtering metric received invalid or degenerate input."""


@dataclass(frozen=True, slots=True)
class GeometryReference:
    """Frozen real-mask reference distribution for one object."""

    area_values: tuple[float, ...]
    area_p05: float
    area_p95: float
    aspect_values: tuple[float, ...]


def _finite_positive(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2:
        raise FilterMetricError(f"{name} requires at least two reference values")
    if not np.isfinite(array).all() or np.any(array <= 0):
        raise FilterMetricError(f"{name} must contain finite positive values")
    return array


def fitted_zscore(
    value: float,
    reference: Sequence[float],
    *,
    log_transform: bool,
) -> float:
    """Score a positive value against a frozen reference distribution."""

    if not math.isfinite(value) or value <= 0:
        raise FilterMetricError("Scored value must be finite and positive")
    values = _finite_positive(reference, name="reference")
    if log_transform:
        values = np.log(values)
        value = math.log(value)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if std <= np.finfo(np.float64).eps:
        return 0.0 if math.isclose(value, mean) else math.copysign(math.inf, value - mean)
    return float((value - mean) / std)


def geometry_scores(
    mask: np.ndarray,
    reference: GeometryReference,
) -> dict[str, float]:
    """Return area ratio, aspect ratio, and real-reference z-scores."""

    if mask.ndim != 2:
        raise FilterMetricError(f"Expected 2D mask, observed {mask.shape}")
    binary = mask > 0
    if not np.any(binary):
        raise FilterMetricError("Mask is empty")
    ys, xs = np.nonzero(binary)
    width = int(xs.max()) - int(xs.min()) + 1
    height = int(ys.max()) - int(ys.min()) + 1
    area_ratio = float(binary.mean())
    aspect_ratio = float(width / height)
    return {
        "area_ratio": area_ratio,
        "area_zscore": fitted_zscore(
            area_ratio,
            reference.area_values,
            log_transform=True,
        ),
        "aspect_ratio": aspect_ratio,
        "aspect_zscore": fitted_zscore(
            aspect_ratio,
            reference.aspect_values,
            log_transform=True,
        ),
    }


def roi_containment(mask: np.ndarray, legal_roi: np.ndarray) -> float:
    """Return the fraction of positive mask pixels inside the legal ROI."""

    if mask.shape != legal_roi.shape or mask.ndim != 2:
        raise FilterMetricError("Mask and legal ROI shapes differ")
    binary = mask > 0
    area = int(binary.sum())
    if area == 0:
        raise FilterMetricError("Mask is empty")
    return float(np.logical_and(binary, legal_roi > 0).sum() / area)


def context_bbox(
    mask: np.ndarray,
    *,
    ratio: float,
) -> tuple[int, int, int, int]:
    """Return a clipped square context bbox as ``x0, y0, x1, y1``."""

    if mask.ndim != 2 or not np.any(mask > 0):
        raise FilterMetricError("Expected a non-empty 2D mask")
    if not math.isfinite(ratio) or ratio < 1:
        raise FilterMetricError("Context ratio must be finite and >=1")
    height, width = mask.shape
    ys, xs = np.nonzero(mask > 0)
    box_width = int(xs.max()) - int(xs.min()) + 1
    box_height = int(ys.max()) - int(ys.min()) + 1
    side = min(max(width, height), max(box_width, box_height, round(max(box_width, box_height) * ratio)))
    center_x = (int(xs.min()) + int(xs.max()) + 1) / 2
    center_y = (int(ys.min()) + int(ys.max()) + 1) / 2
    x0 = round(center_x - side / 2)
    y0 = round(center_y - side / 2)
    x0 = min(max(0, x0), max(0, width - side))
    y0 = min(max(0, y0), max(0, height - side))
    return x0, y0, min(width, x0 + side), min(height, y0 + side)


def crop_phash(image: np.ndarray, mask: np.ndarray, *, ratio: float, hash_size: int) -> str:
    """Compute pHash on the same local context used by semantic metrics."""

    if image.ndim != 3 or image.shape[:2] != mask.shape:
        raise FilterMetricError("Image and mask shapes differ")
    if hash_size < 2:
        raise FilterMetricError("pHash size must be >=2")
    x0, y0, x1, y1 = context_bbox(mask, ratio=ratio)
    crop = Image.fromarray(image[y0:y1, x0:x1].astype(np.uint8), mode="RGB")
    return str(imagehash.phash(crop, hash_size=hash_size))


def phash_distance(first: str, second: str) -> int:
    """Return Hamming distance between two serialized pHashes."""

    try:
        return int(imagehash.hex_to_hash(first) - imagehash.hex_to_hash(second))
    except (TypeError, ValueError) as error:
        raise FilterMetricError("Invalid pHash") from error


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )


def seam_score(
    synthetic: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    *,
    band_px: int,
    histogram_bins: int,
) -> float:
    """Compare outer-boundary gradient distributions with normalized Wasserstein-1."""

    if (
        synthetic.shape != background.shape
        or synthetic.ndim != 3
        or synthetic.shape[:2] != mask.shape
    ):
        raise FilterMetricError("Synthetic/background/mask shapes differ")
    if band_px < 1 or histogram_bins < 2:
        raise FilterMetricError("Invalid seam metric settings")
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        raise FilterMetricError("Mask is empty")
    side = band_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    band = (cv2.dilate(binary, kernel) > 0) & ~(binary > 0)
    if not np.any(band):
        raise FilterMetricError("Mask has no outer boundary band")
    maximum = 1020.0
    synthetic_values = np.clip(_gradient_magnitude(synthetic)[band], 0, maximum)
    background_values = np.clip(_gradient_magnitude(background)[band], 0, maximum)
    histogram_range = (0.0, maximum)
    synthetic_hist, edges = np.histogram(
        synthetic_values,
        bins=histogram_bins,
        range=histogram_range,
        density=False,
    )
    background_hist, _ = np.histogram(
        background_values,
        bins=histogram_bins,
        range=histogram_range,
        density=False,
    )
    synthetic_cdf = np.cumsum(synthetic_hist, dtype=np.float64) / synthetic_hist.sum()
    background_cdf = np.cumsum(background_hist, dtype=np.float64) / background_hist.sum()
    bin_width = float(edges[1] - edges[0])
    wasserstein = float(np.abs(synthetic_cdf - background_cdf).sum() * bin_width)
    return float(np.clip(1.0 - wasserstein / maximum, 0.0, 1.0))
