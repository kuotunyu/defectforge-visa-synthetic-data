"""End-to-end M14 feature extraction, sanity gates, and grouped metrics."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cleanfid import fid as clean_fid
from PIL import Image

from src.common.integrity import sha256_file
from src.common.paths import Paths
from src.evaluation.quality_data import CropEntry, SourceAudit
from src.evaluation.quality_metrics import (
    QualityMetrics,
    evaluate_quality,
    low_rank_fid,
    mutual_nearest_neighbor_score,
    nearest_neighbor_scores,
    polynomial_kid_biased,
)
from src.filtering.dataset import FilterSample
from src.filtering.embeddings import DinoEmbedder
from src.filtering.pipeline import embedding_cache_key

LOGGER = logging.getLogger("quality_pipeline")
PIPELINE_VERSION = "0.1.0"


class QualityPipelineError(RuntimeError):
    """M14 feature extraction or sanity validation failed."""


@dataclass(frozen=True, slots=True)
class QualityRun:
    """All M14 output data before rendering."""

    rows: list[dict[str, Any]]
    sanity: list[dict[str, Any]]
    sanity_passed: bool
    source_audit: SourceAudit
    model_revision: str
    clean_fid_version: str
    generated_dino_cache: str
    reference_dino_cache: str
    clean_feature_cache: str
    representatives: dict[str, str]
    nn_distributions: dict[str, list[float]]


def _manifest_digest(crop_root: Path) -> str:
    return sha256_file(crop_root / "manifest.json")


def _feature_cache_key(
    crop_root: Path,
    *,
    package_version: str,
    mode: str,
) -> str:
    payload = {
        "algorithm": "clean-fid-inception-features-v1",
        "manifest_sha256": _manifest_digest(crop_root),
        "package_version": package_version,
        "mode": mode,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]


def extract_clean_features(
    paths: Paths,
    crop_root: Path,
    entries: Sequence[CropEntry],
    *,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, Path, str]:
    """Load or compute clean-fid Inception features in manifest order."""

    observed_version = importlib.metadata.version("clean-fid")
    expected_version = str(config["package_version"])
    if observed_version != expected_version:
        raise QualityPipelineError(
            f"clean-fid version changed: expected {expected_version}, observed {observed_version}"
        )
    mode = str(config["mode"])
    cache_key = _feature_cache_key(
        crop_root,
        package_version=observed_version,
        mode=mode,
    )
    cache_path = paths.cache / f"m14_cleanfid_features_{cache_key}.npz"
    relative_paths = [entry.relative_path for entry in entries]
    if cache_path.is_file():
        with np.load(cache_path) as cached:
            cached_paths = [str(value) for value in cached["relative_paths"].tolist()]
            features = cached["features"].astype(np.float64)
            cached_version = str(cached["package_version"].item())
        if cached_paths != relative_paths:
            raise QualityPipelineError("clean-fid cache path order changed")
        if cached_version != observed_version or features.shape != (len(entries), 2048):
            raise QualityPipelineError(
                f"Unexpected clean-fid cache contents: {features.shape}, {cached_version}"
            )
        return features, cache_path, observed_version

    if not torch.cuda.is_available():
        raise QualityPipelineError("M14 clean-fid extraction requires CUDA")
    device = torch.device("cuda")
    model = clean_fid.build_feature_extractor(
        mode,
        device=device,
        use_dataparallel=bool(config["use_dataparallel"]),
    )
    try:
        features = clean_fid.get_files_features(
            [str(crop_root / relative_path) for relative_path in relative_paths],
            model=model,
            num_workers=int(config["num_workers"]),
            batch_size=int(config["batch_size"]),
            device=device,
            mode=mode,
            description="M14 clean-fid",
            verbose=True,
        ).astype(np.float64)
    finally:
        del model
        torch.cuda.empty_cache()
    if features.shape != (len(entries), 2048) or not np.isfinite(features).all():
        raise QualityPipelineError(f"Unexpected clean-fid features: {features.shape}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        relative_paths=np.asarray(relative_paths),
        features=features.astype(np.float32),
        package_version=np.asarray(observed_version),
        mode=np.asarray(mode),
    )
    return features, cache_path, observed_version


def _embed_files(
    embedder: DinoEmbedder,
    paths: Sequence[Path],
    *,
    chunk_size: int = 128,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in range(0, len(paths), chunk_size):
        chunk = paths[start : start + chunk_size]
        images: list[Image.Image] = []
        try:
            for path in chunk:
                with Image.open(path) as handle:
                    images.append(handle.convert("RGB"))
            batches.append(embedder.embed(images))
        finally:
            for image in images:
                image.close()
        LOGGER.info("M14 DINO references: %d/%d", min(start + len(chunk), len(paths)), len(paths))
    return np.concatenate(batches, axis=0)


def load_dino_embeddings(
    paths: Paths,
    crop_root: Path,
    entries: Sequence[CropEntry],
    samples: Sequence[FilterSample],
    *,
    config: Mapping[str, Any],
    crop_ratio: float,
) -> tuple[np.ndarray, np.ndarray, Path, Path, str]:
    """Return generated and real/noise embeddings without recomputing M13 outputs."""

    model_id = str(config["model_id"])
    revision = str(config["revision"])
    generated_key = embedding_cache_key(
        samples,
        model_id=model_id,
        revision=revision,
        crop_ratio=crop_ratio,
    )
    generated_cache = paths.cache / f"m13_dinov2_generated_{generated_key}.npz"
    if not generated_cache.is_file():
        raise QualityPipelineError(f"Missing full M13 embedding cache: {generated_cache}")
    with np.load(generated_cache) as cached:
        generated = cached["embeddings"].astype(np.float32)
        generated_revision = str(cached["model_revision"].item())
    if generated.shape != (len(samples), 768) or generated_revision != revision:
        raise QualityPipelineError("M13 generated embedding cache changed")

    reference_entries = [entry for entry in entries if entry.kind in {"real", "noise"}]
    payload = {
        "algorithm": "dinov2-cls-m14-reference-v1",
        "manifest_sha256": _manifest_digest(crop_root),
        "model_id": model_id,
        "revision": revision,
        "keys": [entry.key for entry in reference_entries],
    }
    reference_key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    reference_cache = paths.cache / f"m14_dinov2_references_{reference_key}.npz"
    model_revision = revision
    if reference_cache.is_file():
        with np.load(reference_cache) as cached:
            reference = cached["embeddings"].astype(np.float32)
            cached_keys = [str(value) for value in cached["keys"].tolist()]
            model_revision = str(cached["model_revision"].item())
        if (
            reference.shape != (len(reference_entries), 768)
            or cached_keys != [entry.key for entry in reference_entries]
            or model_revision != revision
        ):
            raise QualityPipelineError("M14 DINO reference cache changed")
    else:
        with DinoEmbedder(
            model_id=model_id,
            revision=revision,
            batch_size=int(config["batch_size"]),
        ) as embedder:
            model_revision = embedder.revision
            reference = _embed_files(
                embedder,
                [crop_root / entry.relative_path for entry in reference_entries],
            )
        np.savez_compressed(
            reference_cache,
            embeddings=reference,
            keys=np.asarray([entry.key for entry in reference_entries]),
            model_revision=np.asarray(model_revision),
        )
    return generated, reference, generated_cache, reference_cache, model_revision


def align_decisions(
    samples: Sequence[FilterSample],
    decisions: Sequence[Mapping[str, Any]],
) -> None:
    """Prove the M13 embedding order is the published decision order."""

    if len(samples) != len(decisions):
        raise QualityPipelineError("M13 source and decision counts differ")
    for index, (sample, decision) in enumerate(zip(samples, decisions, strict=True)):
        identity = (sample.input_name, str(sample.record["sample_id"]))
        decision_identity = (
            str(decision["filter"]["input_name"]),
            str(decision["sample_id"]),
        )
        if identity != decision_identity:
            raise QualityPipelineError(
                f"M13 decision order differs at {index}: {identity} != {decision_identity}"
            )


def _entry_indices(
    entries: Sequence[CropEntry],
    *,
    kind: str,
    object_name: str,
    defect_type: str | None,
) -> list[int]:
    return [
        index
        for index, entry in enumerate(entries)
        if entry.kind == kind
        and entry.object_name == object_name
        and (defect_type is None or entry.defect_type == defect_type)
    ]


def generated_group_indices(
    samples: Sequence[FilterSample],
    decisions: Sequence[Mapping[str, Any]],
    *,
    input_name: str,
    object_name: str,
    defect_type: str | None,
    passed_only: bool,
) -> list[int]:
    """Return deterministic source-order indices for one report group."""

    return [
        index
        for index, (sample, decision) in enumerate(zip(samples, decisions, strict=True))
        if sample.input_name == input_name
        and sample.record["object"] == object_name
        and (defect_type is None or sample.record["defect_type"] == defect_type)
        and (not passed_only or bool(decision["filter"]["passed"]))
    ]


def _metric_row(
    metrics: QualityMetrics | None,
    *,
    view: str,
    input_name: str,
    object_name: str,
    defect_type: str,
    real_scope: str,
    n_real: int,
    n_generated: int,
) -> dict[str, Any]:
    base = {
        "view": view,
        "input_name": input_name,
        "object": object_name,
        "defect_type": defect_type,
        "real_scope": real_scope,
        "n_real": n_real,
        "n_generated": n_generated,
        "status": "empty" if metrics is None else "ok",
    }
    if metrics is None:
        return {
            **base,
            "nn_mean": None,
            "nn_median": None,
            "nn_p05": None,
            "nn_p95": None,
            "mnn_score": None,
            "kid": None,
            "fid": None,
        }
    values = asdict(metrics)
    values.pop("n_real")
    values.pop("n_generated")
    return {**base, **values}


def _sanity_checks(
    entries: Sequence[CropEntry],
    clean_features: np.ndarray,
    reference_embeddings: np.ndarray,
    *,
    thresholds: Mapping[str, Mapping[str, float]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reference_entries = [entry for entry in entries if entry.kind in {"real", "noise"}]
    reference_position = {
        entry.key: index for index, entry in enumerate(reference_entries)
    }
    checks: list[dict[str, Any]] = []
    object_types = sorted(
        {(entry.object_name, entry.defect_type) for entry in entries if entry.kind == "real"}
    )
    for object_name, defect_type in object_types:
        real_entry_indices = _entry_indices(
            entries,
            kind="real",
            object_name=object_name,
            defect_type=defect_type,
        )
        noise_entry_indices = _entry_indices(
            entries,
            kind="noise",
            object_name=object_name,
            defect_type=defect_type,
        )
        real_reference_indices = [
            reference_position[entries[index].key] for index in real_entry_indices
        ]
        noise_reference_indices = [
            reference_position[entries[index].key] for index in noise_entry_indices
        ]
        real_embeddings = reference_embeddings[real_reference_indices]
        noise_embeddings = reference_embeddings[noise_reference_indices]
        real_features = clean_features[real_entry_indices]
        noise_features = clean_features[noise_entry_indices]
        self_nn = nearest_neighbor_scores(real_embeddings, real_embeddings)
        self_mnn = mutual_nearest_neighbor_score(real_embeddings, real_embeddings)
        # Unbiased KID is intended for independent samples.  On the same finite
        # set its excluded diagonals yield a negative value whose magnitude can
        # be large for tiny real groups (n=3 here).  The biased estimator is the
        # correct identity check because X versus X is exactly zero.
        self_kid = polynomial_kid_biased(real_features, real_features)
        self_fid = low_rank_fid(real_features, real_features)
        noise_nn = nearest_neighbor_scores(noise_embeddings, real_embeddings)
        noise_kid = polynomial_kid_biased(noise_features, real_features)
        noise_fid = low_rank_fid(noise_features, real_features)
        tau_low = float(thresholds[object_name]["tau_low"])
        passed = bool(
            float(self_nn.min()) >= float(config["self_nn_minimum"])
            and self_mnn >= float(config["self_mnn_minimum"])
            and self_fid <= float(config["self_fid_maximum"])
            and abs(self_kid) <= float(config["self_kid_abs_maximum"])
            and float(noise_nn.mean()) <= tau_low - float(config["noise_nn_margin"])
            and noise_kid >= self_kid + float(config["noise_kid_minimum_gap"])
            and noise_fid > self_fid
        )
        checks.append(
            {
                "object": object_name,
                "defect_type": defect_type,
                "n": len(real_entry_indices),
                "self_nn_min": float(self_nn.min()),
                "self_mnn": self_mnn,
                "self_kid": self_kid,
                "self_fid": self_fid,
                "noise_nn_mean": float(noise_nn.mean()),
                "tau_low": tau_low,
                "noise_kid": noise_kid,
                "noise_fid": noise_fid,
                "passed": passed,
            }
        )
    return checks


def evaluate_groups(
    paths: Paths,
    entries: Sequence[CropEntry],
    samples: Sequence[FilterSample],
    decisions: Sequence[Mapping[str, Any]],
    generated_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    clean_features: np.ndarray,
    *,
    input_names: Sequence[str],
    views: Sequence[str],
    defect_types: Mapping[str, Any],
    thresholds: Mapping[str, Mapping[str, float]],
    sanity_config: Mapping[str, Any],
    source_audit: SourceAudit,
    model_revision: str,
    clean_fid_version: str,
    generated_dino_cache: Path,
    reference_dino_cache: Path,
    clean_feature_cache: Path,
) -> QualityRun:
    """Compute sanity gates and every per-source/type metric row."""

    align_decisions(samples, decisions)
    generated_entries = [entry for entry in entries if entry.kind == "generated"]
    if len(generated_entries) != len(samples):
        raise QualityPipelineError("Generated crop count differs from M13 inputs")
    clean_position = {entry.key: index for index, entry in enumerate(entries)}
    reference_entries = [entry for entry in entries if entry.kind in {"real", "noise"}]
    reference_position = {
        entry.key: index for index, entry in enumerate(reference_entries)
    }

    rows: list[dict[str, Any]] = []
    distributions: dict[str, list[float]] = {}
    representatives: dict[str, str] = {}
    for view in views:
        if view not in {"unfiltered", "filtered"}:
            raise QualityPipelineError(f"Unknown quality view: {view}")
        passed_only = view == "filtered"
        for input_name in input_names:
            for object_name, object_record in sorted(defect_types["objects"].items()):
                real_type_ids = {
                    _type_from_trigger(record["trigger_token"])
                    for record in object_record["types"]
                }
                generated_type_ids = sorted(
                    {
                        str(sample.record["defect_type"])
                        for sample in samples
                        if sample.input_name == input_name
                        and sample.record["object"] == object_name
                    }
                )
                for defect_type_value in [*generated_type_ids, None]:
                    label = "__all__" if defect_type_value is None else defect_type_value
                    real_defect_type = (
                        defect_type_value
                        if defect_type_value in real_type_ids
                        else None
                    )
                    real_scope = (
                        f"type:{real_defect_type}"
                        if real_defect_type is not None
                        else "object_all"
                    )
                    generated_indices = generated_group_indices(
                        samples,
                        decisions,
                        input_name=input_name,
                        object_name=object_name,
                        defect_type=defect_type_value,
                        passed_only=passed_only,
                    )
                    real_entry_indices = _entry_indices(
                        entries,
                        kind="real",
                        object_name=object_name,
                        defect_type=real_defect_type,
                    )
                    real_reference_indices = [
                        reference_position[entries[index].key]
                        for index in real_entry_indices
                    ]
                    generated_clean_indices = [
                        clean_position[generated_entries[index].key]
                        for index in generated_indices
                    ]
                    metrics = None
                    if generated_indices:
                        group_nn = nearest_neighbor_scores(
                            generated_embeddings[generated_indices],
                            reference_embeddings[real_reference_indices],
                        )
                        distribution_key = f"{view}|{input_name}|{object_name}|{label}"
                        distributions[distribution_key] = [float(value) for value in group_nn]
                        median_position = int(np.argsort(group_nn)[len(group_nn) // 2])
                        representatives[distribution_key] = generated_entries[
                            generated_indices[median_position]
                        ].relative_path
                        metrics = evaluate_quality(
                            generated_embeddings[generated_indices],
                            reference_embeddings[real_reference_indices],
                            clean_features[generated_clean_indices],
                            clean_features[real_entry_indices],
                        )
                    rows.append(
                        _metric_row(
                            metrics,
                            view=view,
                            input_name=input_name,
                            object_name=object_name,
                            defect_type=label,
                            real_scope=real_scope,
                            n_real=len(real_entry_indices),
                            n_generated=len(generated_indices),
                        )
                    )

    sanity = _sanity_checks(
        entries,
        clean_features,
        reference_embeddings,
        thresholds=thresholds,
        config=sanity_config,
    )
    sanity_passed = source_audit.blocklist_hits == 0 and all(
        bool(check["passed"]) for check in sanity
    )
    return QualityRun(
        rows=rows,
        sanity=sanity,
        sanity_passed=sanity_passed,
        source_audit=source_audit,
        model_revision=model_revision,
        clean_fid_version=clean_fid_version,
        generated_dino_cache=str(generated_dino_cache),
        reference_dino_cache=str(reference_dino_cache),
        clean_feature_cache=str(clean_feature_cache),
        representatives=representatives,
        nn_distributions=distributions,
    )


def _type_from_trigger(trigger: str) -> str:
    value = trigger.removeprefix("<").removesuffix(">").rsplit("-", 1)[-1]
    if not value.startswith("type"):
        raise QualityPipelineError(f"Invalid trigger token: {trigger}")
    return value
