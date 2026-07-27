"""Pinned DINOv2 crop embeddings and deterministic real-reference calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src.filtering.metrics import context_bbox


class EmbeddingError(RuntimeError):
    """DINOv2 extraction or calibration could not satisfy its invariants."""


@dataclass(frozen=True, slots=True)
class DinoCalibration:
    """Frozen thresholds and normalized reference embeddings for one object."""

    references: np.ndarray
    centroid: np.ndarray
    tau_low: float
    tau_outlier: float


def l2_normalize(values: np.ndarray) -> np.ndarray:
    """L2-normalize a two-dimensional embedding matrix."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise EmbeddingError("Embeddings must be a non-empty finite matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise EmbeddingError("Embedding contains a zero vector")
    return array / norms


def calibrate_references(
    embeddings: np.ndarray,
    *,
    tau_low_quantile: float,
    outlier_quantile: float,
) -> DinoCalibration:
    """Fit leave-one-out similarity and centroid-distance thresholds."""

    references = l2_normalize(embeddings)
    if len(references) < 3:
        raise EmbeddingError("DINOv2 calibration needs at least three real crops")
    for name, value in {
        "tau_low_quantile": tau_low_quantile,
        "outlier_quantile": outlier_quantile,
    }.items():
        if not math.isfinite(value) or not 0 < value < 1:
            raise EmbeddingError(f"{name} must be in (0, 1)")
    similarities = references @ references.T
    np.fill_diagonal(similarities, -np.inf)
    leave_one_out = similarities.max(axis=1)
    centroid = references.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    outlier = 1.0 - references @ centroid
    return DinoCalibration(
        references=references,
        centroid=centroid.astype(np.float32),
        tau_low=float(np.quantile(leave_one_out, tau_low_quantile)),
        tau_outlier=float(np.quantile(outlier, outlier_quantile)),
    )


def semantic_scores(
    embeddings: np.ndarray,
    calibration: DinoCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-real cosine similarity and centroid outlier distance."""

    generated = l2_normalize(embeddings)
    if generated.shape[1] != calibration.references.shape[1]:
        raise EmbeddingError("Generated and reference dimensions differ")
    nn_score = (generated @ calibration.references.T).max(axis=1)
    outlier_score = 1.0 - generated @ calibration.centroid
    return nn_score.astype(np.float32), outlier_score.astype(np.float32)


def crop_image(image_path: Path, mask: np.ndarray, *, ratio: float) -> Image.Image:
    """Load one image and return the mask-centered context crop."""

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        if (image.height, image.width) != mask.shape:
            raise EmbeddingError(
                f"Image/mask shape mismatch: {image_path} "
                f"{(image.height, image.width)} != {mask.shape}"
            )
        x0, y0, x1, y1 = context_bbox(mask, ratio=ratio)
        return image.crop((x0, y0, x1, y1))


class DinoEmbedder:
    """Small explicit-lifetime wrapper around the pinned DINOv2 encoder."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        batch_size: int,
        device: str = "cuda",
    ) -> None:
        if batch_size < 1:
            raise EmbeddingError("batch_size must be positive")
        if device == "cuda" and not torch.cuda.is_available():
            raise EmbeddingError("DINOv2 filtering requires an available CUDA device")
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision).to(self.device).eval()
        observed = str(getattr(self.model.config, "_commit_hash", None) or "unavailable")
        if observed != revision:
            self.close()
            raise EmbeddingError(
                f"DINOv2 revision mismatch: expected {revision}, observed {observed}"
            )
        self.revision = observed

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Encode PIL crops in deterministic batches using the CLS token."""

        if not images:
            raise EmbeddingError("No crops to embed")
        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                inputs: dict[str, Any] = self.processor(images=list(batch), return_tensors="pt")
                inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
                output = self.model(**inputs).last_hidden_state[:, 0, :]
                batches.append(output.float().cpu().numpy())
        return np.concatenate(batches, axis=0).astype(np.float32)

    def close(self) -> None:
        """Release model memory before another project needs the shared GPU."""

        model = getattr(self, "model", None)
        if model is not None:
            del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
