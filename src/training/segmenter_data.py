"""Leakage-safe image/mask expansion for M18 segmentation experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.common.integrity import sha256_file, verify_frozen_manifest
from src.common.paths import Paths
from src.training.classifier_data import (
    ClassificationSample,
    build_classification_group,
    resolve_sample_path,
)

SEGMENTER_DATA_VERSION = "0.1.0"


class SegmentationDataError(RuntimeError):
    """Raised when an M18 group violates the frozen image/mask contract."""


@dataclass(frozen=True, slots=True)
class SegmentationSample:
    """One portable image/mask record for a segmentation run."""

    sample_id: str
    object_name: str
    kind: str
    source_name: str
    root: str
    image_path: str
    image_sha256: str
    mask_path: str | None
    mask_sha256: str | None
    has_defect: bool
    manifest_refs: tuple[str, ...]
    defect_type: str | None = None


@dataclass(frozen=True, slots=True)
class SegmentationGroup:
    """Expanded train/validation/test records for one M18 run."""

    requested_group: str
    canonical_group: str
    object_name: str
    mode: str
    standard_augmentation: bool
    train: tuple[SegmentationSample, ...]
    validation: tuple[SegmentationSample, ...]
    test: tuple[SegmentationSample, ...]
    manifest_sha256: str
    selection_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "pipeline_version": SEGMENTER_DATA_VERSION,
            "requested_group": self.requested_group,
            "canonical_group": self.canonical_group,
            "object": self.object_name,
            "mode": self.mode,
            "standard_augmentation": self.standard_augmentation,
            "split_manifest_sha256": self.manifest_sha256,
            "selection_sha256": self.selection_sha256,
            "counts": {
                "train": summarize_samples(self.train),
                "validation": summarize_samples(self.validation),
                "test": summarize_samples(self.test),
            },
            "train": [asdict(sample) for sample in self.train],
            "validation": [asdict(sample) for sample in self.validation],
            "test": [asdict(sample) for sample in self.test],
        }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SegmentationDataError(message)


def _resolve_group(
    groups: Mapping[str, Any],
    requested_group: str,
) -> tuple[str, Mapping[str, Any]]:
    require(requested_group in groups, f"Unknown segmentation group: {requested_group}")
    seen: set[str] = set()
    current = requested_group
    while True:
        require(current not in seen, f"Segmentation group alias cycle: {current}")
        seen.add(current)
        raw = groups[current]
        require(isinstance(raw, dict), f"Invalid segmentation group: {current}")
        alias = raw.get("alias_of")
        if alias is None:
            return current, raw
        current = str(alias)
        require(current in groups, f"Unknown segmentation group alias: {current}")


def _manifest_by_image(paths: Paths) -> tuple[dict[str, Mapping[str, Any]], str]:
    manifest, manifest_sha256 = verify_frozen_manifest(paths.splits)
    rows = manifest.get("images")
    require(isinstance(rows, list), "Frozen manifest images are missing")
    by_image = {str(row["image_path"]): row for row in rows}
    require(len(by_image) == len(rows), "Frozen manifest contains duplicate image paths")
    return by_image, manifest_sha256


def _synthetic_mask_index(
    paths: Paths,
    view: str,
) -> dict[str, tuple[str, str]]:
    root = paths.synthetic / view
    metadata_path = root / "metadata.jsonl"
    require(metadata_path.is_file(), f"Synthetic metadata is missing: {view}")
    index: dict[str, tuple[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            require(isinstance(raw, dict), f"Invalid metadata line {view}:{line_number}")
            image_path = str(raw["image_path"]).replace("\\", "/")
            mask_path = str(raw["mask_path"]).replace("\\", "/")
            require(image_path not in index, f"Duplicate synthetic image path: {view}/{image_path}")
            resolved_mask = (root / mask_path).resolve(strict=True)
            index[image_path] = (mask_path, sha256_file(resolved_mask))
    return index


def _from_classification_sample(
    paths: Paths,
    sample: ClassificationSample,
    manifest_by_image: Mapping[str, Mapping[str, Any]],
    synthetic_indexes: dict[str, dict[str, tuple[str, str]]],
) -> SegmentationSample:
    if sample.kind == "real":
        record = manifest_by_image.get(sample.relative_path)
        require(record is not None, f"Real sample is absent from manifest: {sample.relative_path}")
        if sample.label == 0:
            mask_path = None
            mask_sha256 = None
        else:
            require(record.get("mask_path"), f"Real anomaly has no mask: {sample.relative_path}")
            mask_path = str(record["mask_path"])
            mask_sha256 = str(record["mask_sha256"])
        return SegmentationSample(
            sample_id=sample.sample_id,
            object_name=sample.object_name,
            kind=sample.kind,
            source_name=sample.source_name,
            root=sample.root,
            image_path=sample.relative_path,
            image_sha256=sample.sha256,
            mask_path=mask_path,
            mask_sha256=mask_sha256,
            has_defect=sample.label == 1,
            manifest_refs=sample.manifest_refs,
            defect_type=sample.defect_type,
        )

    require(sample.root.startswith("synthetic/"), f"Unknown synthetic root: {sample.root}")
    view = sample.root.removeprefix("synthetic/")
    if view not in synthetic_indexes:
        synthetic_indexes[view] = _synthetic_mask_index(paths, view)
    mask_record = synthetic_indexes[view].get(sample.relative_path)
    require(mask_record is not None, f"Synthetic mask record is missing: {view}/{sample.relative_path}")
    mask_path, mask_sha256 = mask_record
    return SegmentationSample(
        sample_id=sample.sample_id,
        object_name=sample.object_name,
        kind=sample.kind,
        source_name=sample.source_name,
        root=sample.root,
        image_path=sample.relative_path,
        image_sha256=sample.sha256,
        mask_path=mask_path,
        mask_sha256=mask_sha256,
        has_defect=True,
        manifest_refs=sample.manifest_refs,
        defect_type=sample.defect_type,
    )


def resolve_image_path(paths: Paths, sample: SegmentationSample) -> Path:
    classification_sample = ClassificationSample(
        sample_id=sample.sample_id,
        object_name=sample.object_name,
        label=int(sample.has_defect),
        kind=sample.kind,
        source_name=sample.source_name,
        root=sample.root,
        relative_path=sample.image_path,
        sha256=sample.image_sha256,
        manifest_refs=sample.manifest_refs,
        defect_type=sample.defect_type,
    )
    return resolve_sample_path(paths, classification_sample)


def resolve_mask_path(paths: Paths, sample: SegmentationSample) -> Path | None:
    if sample.mask_path is None:
        return None
    if sample.root == "visa_raw":
        root = paths.visa_raw
    elif sample.root.startswith("synthetic/"):
        root = paths.synthetic / sample.root.removeprefix("synthetic/")
    else:
        raise SegmentationDataError(f"Unknown sample root: {sample.root}")
    return (root / sample.mask_path).resolve(strict=True)


def _validate_mask(paths: Paths, sample: SegmentationSample) -> None:
    image_path = resolve_image_path(paths, sample)
    require(sha256_file(image_path) == sample.image_sha256, f"Image hash changed: {sample.sample_id}")
    mask_path = resolve_mask_path(paths, sample)
    if not sample.has_defect:
        require(mask_path is None and sample.mask_sha256 is None, "Normal sample has a stored mask")
        return
    require(mask_path is not None and sample.mask_sha256 is not None, "Defect mask is missing")
    require(sha256_file(mask_path) == sample.mask_sha256, f"Mask hash changed: {sample.sample_id}")
    with Image.open(image_path) as image_handle, Image.open(mask_path) as mask_handle:
        require(image_handle.size == mask_handle.size, f"Image/mask size mismatch: {sample.sample_id}")
        mask = np.asarray(mask_handle.convert("L"))
    values = {int(value) for value in np.unique(mask)}
    if sample.kind == "synthetic":
        require(
            values <= {0, 1} or values <= {0, 255},
            f"Synthetic mask is not binary: {sample.sample_id} {sorted(values)}",
        )
    require(any(value > 0 for value in values), f"Defect mask is empty: {sample.sample_id}")


def validate_group(paths: Paths, group: SegmentationGroup) -> dict[str, Any]:
    seen: set[str] = set()
    for partition in (group.train, group.validation, group.test):
        for sample in partition:
            require(sample.image_sha256 not in seen, f"Duplicate image content: {sample.sample_id}")
            seen.add(sample.image_sha256)
            _validate_mask(paths, sample)

    if group.mode == "development":
        require(group.validation, "Development run has no validation set")
        require(not group.test, "Development run must not load the test set")
        require(
            all(sample.kind == "real" for sample in group.validation),
            "Validation contains synthetic samples",
        )
    else:
        require(not group.validation, "Final run retained a validation holdout")
        require(group.test, "Final run has no frozen real test set")
        require(all(sample.kind == "real" for sample in group.test), "Test contains synthetic data")

    if group.canonical_group == "procedural_only":
        require(
            not any(sample.kind == "real" and sample.has_defect for sample in group.train),
            "procedural_only contains real defect pixels",
        )
        require(
            any(sample.kind == "synthetic" for sample in group.train),
            "procedural_only contains no procedural samples",
        )
    return {
        "status": "passed",
        "unique_images": len(seen),
        "validation_is_real_only": all(sample.kind == "real" for sample in group.validation),
        "test_is_real_only": all(sample.kind == "real" for sample in group.test),
        "real_defect_train_images": sum(
            sample.kind == "real" and sample.has_defect for sample in group.train
        ),
    }


def build_segmentation_group(
    paths: Paths,
    config: Mapping[str, Any],
    *,
    group_name: str,
    object_name: str,
    seed: int,
    mode: str,
) -> SegmentationGroup:
    groups = config.get("groups")
    require(isinstance(groups, dict), "Segmenter group config is missing")
    canonical_group, group_config = _resolve_group(groups, group_name)
    if mode == "development":
        require(canonical_group == "real_only", "Only real_only may be used for development")

    classifier_group_config = dict(group_config)
    remove_real_defects = classifier_group_config.get("real_bad") == "none"
    if remove_real_defects:
        classifier_group_config["real_bad"] = "seed10"
    classifier_config = {"groups": {canonical_group: classifier_group_config}}
    classification = build_classification_group(
        paths,
        classifier_config,
        group_name=canonical_group,
        object_name=object_name,
        seed=seed,
        mode=mode,
    )
    classification_train = classification.train
    if remove_real_defects:
        classification_train = tuple(
            sample
            for sample in classification_train
            if sample.kind == "synthetic" or sample.label == 0
        )

    manifest_by_image, manifest_sha256 = _manifest_by_image(paths)
    require(
        manifest_sha256 == classification.manifest_sha256,
        "Classifier and segmenter manifest hashes differ",
    )
    synthetic_indexes: dict[str, dict[str, tuple[str, str]]] = {}

    def convert(samples: Sequence[ClassificationSample]) -> tuple[SegmentationSample, ...]:
        return tuple(
            _from_classification_sample(
                paths,
                sample,
                manifest_by_image,
                synthetic_indexes,
            )
            for sample in samples
        )

    group = SegmentationGroup(
        requested_group=group_name,
        canonical_group=canonical_group,
        object_name=object_name,
        mode=mode,
        standard_augmentation=bool(group_config.get("standard_augmentation", False)),
        train=convert(classification_train),
        validation=convert(classification.validation),
        test=convert(classification.test),
        manifest_sha256=classification.manifest_sha256,
        selection_sha256=classification.selection_sha256,
    )
    validate_group(paths, group)
    return group


def summarize_samples(samples: Sequence[SegmentationSample]) -> dict[str, Any]:
    return {
        "total": len(samples),
        "defect": sum(sample.has_defect for sample in samples),
        "normal": sum(not sample.has_defect for sample in samples),
        "real": sum(sample.kind == "real" for sample in samples),
        "synthetic": sum(sample.kind == "synthetic" for sample in samples),
        "sources": {
            source: sum(sample.source_name == source for sample in samples)
            for source in sorted({sample.source_name for sample in samples})
        },
    }


def group_payload_sha256(group: SegmentationGroup) -> str:
    payload = json.dumps(
        group.payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
