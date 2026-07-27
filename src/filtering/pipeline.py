"""End-to-end deterministic M13 scoring pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from src.common.imaging import detect_legal_roi
from src.common.paths import Paths
from src.filtering.dataset import FilterSample, filtered_record
from src.filtering.embeddings import (
    DinoCalibration,
    DinoEmbedder,
    calibrate_references,
    crop_image,
    semantic_scores,
)
from src.filtering.metrics import (
    GeometryReference,
    crop_phash,
    geometry_scores,
    phash_distance,
    roi_containment,
    seam_score,
)
from src.filtering.rules import rejection_reasons
from src.synthetic.copy_paste import object_roi_config
from src.synthetic.mask_placement import legal_roi

LOGGER = logging.getLogger("filter_pipeline")
PIPELINE_VERSION = "0.2.0"


class FilterPipelineError(RuntimeError):
    """The end-to-end filtering pipeline could not satisfy an invariant."""


@dataclass(frozen=True, slots=True)
class FilterRun:
    """Complete decisions plus the thresholds needed to reproduce them."""

    records: list[dict[str, Any]]
    thresholds: dict[str, dict[str, float]]
    counts: dict[str, Any]
    model_revision: str | None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise FilterPipelineError(f"Expected YAML mapping in {path}")
    return value


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L")) > 0


def geometry_references(path: Path) -> dict[str, GeometryReference]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    references: dict[str, GeometryReference] = {}
    for object_name, section in payload["objects"].items():
        samples = section["samples"]
        area = tuple(float(sample["area_ratio"]) for sample in samples)
        aspect = tuple(float(sample["aspect_ratio"]) for sample in samples)
        references[object_name] = GeometryReference(
            area_values=area,
            area_p05=float(section.get("area_ratio_p05", np.quantile(area, 0.05))),
            area_p95=float(section.get("area_ratio_p95", np.quantile(area, 0.95))),
            aspect_values=aspect,
        )
    return references


def _real_crops(
    paths: Paths,
    stats_path: Path,
    *,
    ratio: float,
) -> dict[str, list[Image.Image]]:
    with stats_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result: dict[str, list[Image.Image]] = {}
    for object_name, section in payload["objects"].items():
        crops: list[Image.Image] = []
        for sample in section["samples"]:
            full_mask = load_mask(paths.visa_raw / sample["mask_path"])
            x, y, width, height = (int(value) for value in sample["bbox"])
            component = np.zeros_like(full_mask)
            component[y : y + height, x : x + width] = full_mask[
                y : y + height,
                x : x + width,
            ]
            crops.append(
                crop_image(
                    paths.visa_raw / sample["image_path"],
                    component,
                    ratio=ratio,
                )
            )
        result[object_name] = crops
    return result


def _embed_sample_crops(
    embedder: DinoEmbedder,
    samples: Sequence[FilterSample],
    *,
    ratio: float,
    chunk_size: int = 128,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in range(0, len(samples), chunk_size):
        chunk_samples = samples[start : start + chunk_size]
        crops = [
            crop_image(sample.image_path, load_mask(sample.mask_path), ratio=ratio)
            for sample in chunk_samples
        ]
        try:
            batches.append(embedder.embed(crops))
        finally:
            for crop in crops:
                crop.close()
        LOGGER.info("DINOv2 crops: %d/%d", min(start + len(crops), len(samples)), len(samples))
    return np.concatenate(batches, axis=0)


def embedding_cache_key(
    samples: Sequence[FilterSample],
    *,
    model_id: str,
    revision: str,
    crop_ratio: float,
) -> str:
    payload = {
        "algorithm": "dinov2_cls_l2_v1",
        "model_id": model_id,
        "revision": revision,
        "crop_ratio": crop_ratio,
        "samples": [
            {
                "input": sample.input_name,
                "sample_id": sample.record["sample_id"],
                "image_size": sample.image_path.stat().st_size,
                "mask_size": sample.mask_path.stat().st_size,
            }
            for sample in samples
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _load_roi_score_maps(paths: Paths, *, revision: str) -> dict[str, np.ndarray]:
    """Load frozen M9 score maps without triggering a second model process."""

    result: dict[str, np.ndarray] = {}
    for cache_file in sorted(paths.cache.glob("m9_dinov2_scores_*.npz")):
        with np.load(cache_file) as cached:
            if str(cached["model_revision"].item()) != revision:
                continue
            image_paths = [str(value) for value in cached["image_paths"].tolist()]
            scores = cached["scores"].astype(np.float32)
        for image_path, score in zip(image_paths, scores, strict=True):
            previous = result.get(image_path)
            if previous is not None and not np.array_equal(previous, score):
                raise FilterPipelineError(f"Conflicting M9 ROI caches for {image_path}")
            result[image_path] = score
    return result


def _legal_roi_for_sample(
    sample: FilterSample,
    background: np.ndarray,
    *,
    paths: Paths,
    stage_a_config: dict[str, Any],
    placement_config: dict[str, Any],
    roi_score_maps: Mapping[str, np.ndarray],
) -> np.ndarray:
    object_name = str(sample.record["object"])
    background_path = str(sample.record["source"]["background_image"])
    if sample.record["generator"] in {"stageB_sd2", "stageB_sdxl"}:
        score = roi_score_maps.get(background_path)
        if score is None:
            raise FilterPipelineError(f"Missing frozen M9 ROI score for {background_path}")
        return legal_roi(
            background,
            score,
            config=placement_config,
            object_name=object_name,
            method="intersect",
        )
    return detect_legal_roi(
        background,
        **object_roi_config(stage_a_config, object_name),
    )


def _thresholds(
    filter_config: dict[str, Any],
    references: Mapping[str, GeometryReference],
    calibrations: Mapping[str, DinoCalibration] | None,
) -> dict[str, dict[str, float]]:
    rules = filter_config["rules"]
    result: dict[str, dict[str, float]] = {}
    for object_name, reference in references.items():
        semantic = None if calibrations is None else calibrations[object_name]
        result[object_name] = {
            "minimum_containment": float(rules["roi"]["minimum_containment"]),
            "area_p05": reference.area_p05,
            "area_p95": reference.area_p95,
            "maximum_area_abs_zscore": float(rules["area"]["maximum_abs_zscore"]),
            "maximum_aspect_abs_zscore": float(rules["aspect"]["maximum_abs_zscore"]),
            "minimum_phash_distance": float(rules["phash"]["minimum_hamming_distance"]),
            "tau_low": 0.0 if semantic is None else semantic.tau_low,
            "tau_copy": float(rules["dinov2"]["tau_copy"]),
            "tau_outlier": 0.0 if semantic is None else semantic.tau_outlier,
            "minimum_seam_score": float(rules["seam"]["minimum_score"]),
        }
    return result


def run_filter_pipeline(
    paths: Paths,
    samples: Sequence[FilterSample],
    *,
    filter_config: dict[str, Any],
    stage_a_config: dict[str, Any],
    placement_config: dict[str, Any],
    stats_path: Path,
    disabled: set[str] | None = None,
) -> FilterRun:
    """Score every sample, then apply pHash against prior finally-accepted crops."""

    disabled = disabled or set()
    references = geometry_references(stats_path)
    semantic_config = filter_config["rules"]["dinov2"]
    roi_score_maps = (
        {}
        if "roi" in disabled
        else _load_roi_score_maps(paths, revision=str(semantic_config["revision"]))
    )
    score_rows: list[dict[str, float | int | str | None]] = []
    # Keep CPU RAM friendly while this workstation runs three projects at once.
    # Legal ROIs remain cached (small boolean arrays), but full RGB backgrounds do not.
    backgrounds: OrderedDict[str, np.ndarray] = OrderedDict()
    legal_rois: dict[tuple[str, str], np.ndarray] = {}

    for index, sample in enumerate(samples):
        object_name = str(sample.record["object"])
        mask = load_mask(sample.mask_path)
        synthetic = load_rgb(sample.image_path)
        background_path = str(sample.record["source"]["background_image"])
        background = backgrounds.get(background_path)
        if background is None:
            background = load_rgb(paths.visa_raw / background_path)
            backgrounds[background_path] = background
            if len(backgrounds) > 32:
                backgrounds.popitem(last=False)
        else:
            backgrounds.move_to_end(background_path)
        if synthetic.shape != background.shape or mask.shape != background.shape[:2]:
            raise FilterPipelineError(f"Shape mismatch for {sample.key}")
        geometry = geometry_scores(mask, references[object_name])
        roi_key = (sample.record["generator"], background_path)
        containment = 1.0
        if "roi" not in disabled:
            selected_roi = legal_rois.get(roi_key)
            if selected_roi is None:
                selected_roi = _legal_roi_for_sample(
                    sample,
                    background,
                    paths=paths,
                    stage_a_config=stage_a_config,
                    placement_config=placement_config,
                    roi_score_maps=roi_score_maps,
                )
                legal_rois[roi_key] = selected_roi
            containment = roi_containment(mask, selected_roi)
        seam = (
            1.0
            if "seam" in disabled
            else seam_score(
                synthetic,
                background,
                mask,
                band_px=int(filter_config["rules"]["seam"]["band_px"]),
                histogram_bins=int(filter_config["rules"]["seam"]["histogram_bins"]),
            )
        )
        phash = crop_phash(
            synthetic,
            mask,
            ratio=float(semantic_config["crop_ratio"]),
            hash_size=int(filter_config["rules"]["phash"]["hash_size"]),
        )
        score_rows.append(
            {
                **geometry,
                "area_p05": references[object_name].area_p05,
                "area_p95": references[object_name].area_p95,
                "roi_containment": containment,
                "phash": phash,
                "phash_min_dist": None,
                "nn_score": 0.0,
                "tau_low": 0.0,
                "outlier_score": 0.0,
                "tau_outlier": 0.0,
                "seam_score": seam,
            }
        )
        if (index + 1) % 100 == 0:
            LOGGER.info("CPU metrics: %d/%d", index + 1, len(samples))

    calibrations: dict[str, DinoCalibration] | None = None
    model_revision: str | None = None
    if "dinov2" not in disabled:
        model_id = str(semantic_config["model_id"])
        revision = str(semantic_config["revision"])
        crop_ratio = float(semantic_config["crop_ratio"])
        cache_key = embedding_cache_key(
            samples,
            model_id=model_id,
            revision=revision,
            crop_ratio=crop_ratio,
        )
        cache_file = paths.cache / f"m13_dinov2_generated_{cache_key}.npz"
        real_crops = _real_crops(paths, stats_path, ratio=crop_ratio)
        try:
            with DinoEmbedder(
                model_id=model_id,
                revision=revision,
                batch_size=int(semantic_config["batch_size"]),
            ) as embedder:
                model_revision = embedder.revision
                calibrations = {
                    object_name: calibrate_references(
                        embedder.embed(crops),
                        tau_low_quantile=float(semantic_config["tau_low_quantile"]),
                        outlier_quantile=float(semantic_config["outlier_quantile"]),
                    )
                    for object_name, crops in real_crops.items()
                }
                if cache_file.is_file():
                    with np.load(cache_file) as cached:
                        generated_embeddings = cached["embeddings"].astype(np.float32)
                        cached_revision = str(cached["model_revision"].item())
                    if generated_embeddings.shape != (len(samples), 768):
                        raise FilterPipelineError(
                            f"Unexpected generated embedding cache shape: "
                            f"{generated_embeddings.shape}"
                        )
                    if cached_revision != revision:
                        raise FilterPipelineError("Generated embedding cache revision changed")
                    LOGGER.info("Loaded %d generated embeddings", len(samples))
                else:
                    generated_embeddings = _embed_sample_crops(
                        embedder,
                        samples,
                        ratio=crop_ratio,
                    )
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        cache_file,
                        embeddings=generated_embeddings,
                        model_revision=np.asarray(model_revision),
                    )
        finally:
            for crops in real_crops.values():
                for crop in crops:
                    crop.close()
        for object_name in references:
            indices = [
                index
                for index, sample in enumerate(samples)
                if sample.record["object"] == object_name
            ]
            nearest, outlier = semantic_scores(
                generated_embeddings[indices],
                calibrations[object_name],
            )
            for local_index, sample_index in enumerate(indices):
                score_rows[sample_index]["nn_score"] = float(nearest[local_index])
                score_rows[sample_index]["tau_low"] = calibrations[object_name].tau_low
                score_rows[sample_index]["outlier_score"] = float(outlier[local_index])
                score_rows[sample_index]["tau_outlier"] = calibrations[
                    object_name
                ].tau_outlier

    thresholds = _thresholds(filter_config, references, calibrations)
    accepted_hashes: dict[str, list[str]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for sample, scores in zip(samples, score_rows, strict=True):
        object_name = str(sample.record["object"])
        current_hash = str(scores.pop("phash"))
        distances = [
            phash_distance(current_hash, accepted_hash)
            for accepted_hash in accepted_hashes[object_name]
        ]
        scores["phash_min_dist"] = min(distances) if distances else None
        reasons = rejection_reasons(scores, config=filter_config, disabled=disabled)
        if not reasons:
            accepted_hashes[object_name].append(current_hash)
        reason_counts.update(reasons)
        records.append(
            filtered_record(
                sample,
                scores=scores,
                reject_reasons=reasons,
                thresholds=thresholds[object_name],
                pipeline_version=PIPELINE_VERSION,
            )
        )
    accepted_count = sum(bool(record["filter"]["passed"]) for record in records)
    return FilterRun(
        records=records,
        thresholds=thresholds,
        counts={
            "total": len(records),
            "accepted": accepted_count,
            "rejected": len(records) - accepted_count,
            "reject_reasons": dict(sorted(reason_counts.items())),
            "by_input": dict(sorted(Counter(sample.input_name for sample in samples).items())),
            "by_object": dict(
                sorted(Counter(str(sample.record["object"]) for sample in samples).items())
            ),
        },
        model_revision=model_revision,
    )
