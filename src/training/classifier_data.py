"""Leakage-safe dataset expansion for M16 classification experiments."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.common.integrity import (
    IntegrityError,
    load_json,
    read_checksum_file,
    sha256_file,
    verify_frozen_manifest,
)
from src.common.paths import Paths

CLASSIFIER_DATA_VERSION = "0.1.0"


class ClassificationDataError(RuntimeError):
    """Raised when an M16 group cannot satisfy the frozen data contract."""


@dataclass(frozen=True, slots=True)
class ClassificationSample:
    """One portable run-manifest entry without an absolute machine path."""

    sample_id: str
    object_name: str
    label: int
    kind: str
    source_name: str
    root: str
    relative_path: str
    sha256: str
    manifest_refs: tuple[str, ...]
    defect_type: str | None = None


@dataclass(frozen=True, slots=True)
class ClassificationGroup:
    """Expanded train/validation/test lists for one run."""

    requested_group: str
    canonical_group: str
    object_name: str
    mode: str
    standard_augmentation: bool
    train: tuple[ClassificationSample, ...]
    validation: tuple[ClassificationSample, ...]
    test: tuple[ClassificationSample, ...]
    manifest_sha256: str
    selection_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "pipeline_version": CLASSIFIER_DATA_VERSION,
            "requested_group": self.requested_group,
            "canonical_group": self.canonical_group,
            "object": self.object_name,
            "mode": self.mode,
            "standard_augmentation": self.standard_augmentation,
            "manifest_sha256": self.manifest_sha256,
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
        raise ClassificationDataError(message)


def _load_selection(paths: Paths, manifest_sha256: str) -> tuple[dict[str, Any], str]:
    expected, filename = read_checksum_file(paths.splits / "FEWSHOT_SELECTION.sha256")
    selection_path = paths.splits / filename
    actual = sha256_file(selection_path)
    require(expected == actual, "Few-shot selection checksum changed")
    selection = load_json(selection_path)
    require(
        selection.get("manifest_sha256") == manifest_sha256,
        "Few-shot selection points to another manifest",
    )
    return selection, actual


def _manifest_index(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, set[str]]]:
    images = manifest.get("images")
    require(isinstance(images, list), "Manifest images are missing")
    by_path: dict[str, Mapping[str, Any]] = {}
    group_sets: dict[str, set[str]] = defaultdict(set)
    for raw in images:
        require(isinstance(raw, dict), "Manifest image record is invalid")
        relative = str(raw["image_path"])
        require(relative not in by_path, f"Duplicate manifest path: {relative}")
        by_path[relative] = raw
        group_sets[f"{raw['object']}:{raw['group_id']}"].add(str(raw["set"]))
    crossing = sorted(key for key, sets in group_sets.items() if len(sets) != 1)
    require(not crossing, f"pHash groups cross partitions: {crossing[:3]}")
    return by_path, group_sets


def _real_sample(record: Mapping[str, Any]) -> ClassificationSample:
    relative = str(record["image_path"])
    label = 1 if record["label"] == "bad" else 0
    return ClassificationSample(
        sample_id=f"real::{relative}",
        object_name=str(record["object"]),
        label=label,
        kind="real",
        source_name=f"real_{record['set']}",
        root="visa_raw",
        relative_path=relative,
        sha256=str(record["sha256"]),
        manifest_refs=(relative,),
    )


def _selection_paths(
    selection: Mapping[str, Any],
    object_name: str,
    section: str,
) -> set[str]:
    object_record = selection["objects"][object_name]
    if section == "fewshot_seed":
        rows = object_record[section]
    else:
        validation = object_record["validation"]
        rows = [*validation["good"], *validation["bad"]]
    return {str(row["image_path"]) for row in rows}


def _resolve_group(
    groups: Mapping[str, Any],
    requested_group: str,
) -> tuple[str, Mapping[str, Any]]:
    require(requested_group in groups, f"Unknown classification group: {requested_group}")
    seen: set[str] = set()
    current = requested_group
    while True:
        require(current not in seen, f"Classification group alias cycle: {current}")
        seen.add(current)
        raw = groups[current]
        require(isinstance(raw, dict), f"Invalid group config: {current}")
        alias = raw.get("alias_of")
        if alias is None:
            return current, raw
        current = str(alias)
        require(current in groups, f"Unknown classification group alias: {current}")


def _synthetic_input_name(record: Mapping[str, Any]) -> str:
    filter_record = record.get("filter")
    if isinstance(filter_record, dict) and filter_record.get("input_name"):
        return str(filter_record["input_name"])
    generator = str(record["generator"])
    bucket = record.get("bucket")
    if bucket in (None, ""):
        return generator
    if generator == "stageA_procedural" and bucket == "no_real_stats":
        return "stageA_procedural/no_real_stats"
    return f"{generator}/{bucket}"


def _load_synthetic_records(
    paths: Paths,
    *,
    view: str,
    object_name: str,
    inputs: set[str] | None,
) -> list[tuple[Mapping[str, Any], Path, str]]:
    root = (paths.synthetic / view).resolve(strict=False)
    metadata_path = root / "metadata.jsonl"
    require(metadata_path.is_file(), f"Synthetic metadata is missing: {view}")
    records: list[tuple[Mapping[str, Any], Path, str]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            require(isinstance(raw, dict), f"Invalid metadata line {view}:{line_number}")
            if raw.get("object") != object_name:
                continue
            input_name = _synthetic_input_name(raw)
            if inputs is not None and input_name not in inputs:
                continue
            image_path = (root / str(raw["image_path"])).resolve(strict=True)
            records.append((raw, image_path, input_name))
    return records


def _stable_seed(seed: int, *values: str) -> int:
    payload = "\0".join([str(seed), *values]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stratified_select(
    records: Sequence[tuple[Mapping[str, Any], Path, str]],
    *,
    count: int,
    seed: int,
    group_name: str,
    object_name: str,
) -> list[tuple[Mapping[str, Any], Path, str]]:
    """Select exact deterministic quotas across source and defect type."""
    require(count > 0, "Synthetic count must be positive")
    require(len(records) >= count, f"{group_name}/{object_name} has {len(records)} < {count}")
    strata: dict[tuple[str, str], list[tuple[Mapping[str, Any], Path, str]]] = defaultdict(list)
    for record in records:
        raw, _, input_name = record
        strata[(input_name, str(raw["defect_type"]))].append(record)
    for items in strata.values():
        items.sort(key=lambda item: str(item[0]["sample_id"]))

    total = sum(len(items) for items in strata.values())
    ideals = {key: count * len(items) / total for key, items in strata.items()}
    quotas = {key: min(len(strata[key]), math.floor(value)) for key, value in ideals.items()}
    remaining = count - sum(quotas.values())
    order = sorted(
        strata,
        key=lambda key: (
            -(ideals[key] - math.floor(ideals[key])),
            key,
        ),
    )
    while remaining:
        progress = False
        for key in order:
            if quotas[key] < len(strata[key]):
                quotas[key] += 1
                remaining -= 1
                progress = True
                if remaining == 0:
                    break
        require(progress, "Could not allocate all synthetic quotas")

    selected: list[tuple[Mapping[str, Any], Path, str]] = []
    for key in sorted(strata):
        rng = random.Random(_stable_seed(seed, group_name, object_name, *key))
        indices = sorted(rng.sample(range(len(strata[key])), quotas[key]))
        selected.extend(strata[key][index] for index in indices)
    selected.sort(key=lambda item: (item[2], str(item[0]["defect_type"]), str(item[0]["sample_id"])))
    require(len(selected) == count, "Synthetic quota total mismatch")
    return selected


def _synthetic_sample(
    paths: Paths,
    view: str,
    raw: Mapping[str, Any],
    image_path: Path,
    input_name: str,
    manifest_index: Mapping[str, Mapping[str, Any]],
    blocked: set[str],
) -> ClassificationSample:
    source = raw.get("source")
    require(isinstance(source, dict), f"Synthetic provenance is missing: {raw.get('sample_id')}")
    manifest_refs = tuple(
        str(source[field])
        for field in ("background_image", "defect_source_image")
        if source.get(field) is not None
    )
    require(manifest_refs, f"Synthetic record has no manifest provenance: {raw.get('sample_id')}")
    for relative in manifest_refs:
        manifest_record = manifest_index.get(relative)
        require(manifest_record is not None, f"Synthetic source is absent from manifest: {relative}")
        require(
            manifest_record["set"] == "train",
            f"Synthetic record points to non-train source: {relative}",
        )
    background = str(source["background_image"])
    background_record = manifest_index[background]
    require(
        source.get("background_sha256") == background_record["sha256"],
        f"Synthetic background hash changed: {raw.get('sample_id')}",
    )
    observed = sha256_file(image_path)
    require(observed not in blocked, f"Synthetic image hits test blocklist: {image_path}")
    return ClassificationSample(
        sample_id=f"synthetic::{view}::{raw['sample_id']}",
        object_name=str(raw["object"]),
        label=1,
        kind="synthetic",
        source_name=input_name,
        root=f"synthetic/{view}",
        relative_path=str(raw["image_path"]).replace("\\", "/"),
        sha256=observed,
        manifest_refs=manifest_refs,
        defect_type=str(raw["defect_type"]),
    )


def _real_training_records(
    manifest_records: Sequence[Mapping[str, Any]],
    *,
    object_name: str,
    real_bad: str,
    seed_paths: set[str],
    validation_paths: set[str],
    mode: str,
) -> list[ClassificationSample]:
    records = [
        record
        for record in manifest_records
        if record["object"] == object_name and record["set"] == "train"
    ]
    selected: list[Mapping[str, Any]] = []
    for record in records:
        relative = str(record["image_path"])
        if mode == "development" and relative in validation_paths:
            continue
        include_bad = (
            (real_bad == "seed10" and relative in seed_paths)
            or (real_bad == "pool20" and bool(record["in_fewshot_pool"]))
            or real_bad == "full"
        )
        if record["label"] == "good" or include_bad:
            selected.append(record)
    require(real_bad in {"seed10", "pool20", "full"}, f"Invalid real_bad policy: {real_bad}")
    return [_real_sample(record) for record in sorted(selected, key=lambda item: item["image_path"])]


def _real_evaluation_records(
    manifest_records: Sequence[Mapping[str, Any]],
    *,
    object_name: str,
    selected_paths: set[str] | None,
    partition: str,
) -> list[ClassificationSample]:
    records = []
    for record in manifest_records:
        if record["object"] != object_name:
            continue
        if partition == "test" and record["set"] != "test":
            continue
        if partition == "validation" and (
            record["set"] != "train"
            or str(record["image_path"]) not in (selected_paths or set())
        ):
            continue
        records.append(_real_sample(record))
    return sorted(records, key=lambda sample: sample.relative_path)


def resolve_sample_path(paths: Paths, sample: ClassificationSample) -> Path:
    if sample.root == "visa_raw":
        root = paths.visa_raw
    elif sample.root.startswith("synthetic/"):
        root = paths.synthetic / sample.root.removeprefix("synthetic/")
    else:
        raise ClassificationDataError(f"Unknown sample root: {sample.root}")
    return (root / sample.relative_path).resolve(strict=True)


def validate_group(paths: Paths, group: ClassificationGroup) -> dict[str, Any]:
    blocklist = load_json(paths.splits / "test_blocklist.json")
    blocked = {str(value) for value in blocklist["sha256"]}
    train_hashes: set[str] = set()
    for sample in (*group.train, *group.validation):
        path = resolve_sample_path(paths, sample)
        observed = sha256_file(path)
        require(observed == sample.sha256, f"Sample hash changed: {sample.sample_id}")
        require(observed not in blocked, f"Training-side blocklist hit: {sample.sample_id}")
        require(sample.sha256 not in train_hashes, f"Duplicate training-side content: {sample.sample_id}")
        train_hashes.add(sample.sha256)

    test_hashes: set[str] = set()
    for sample in group.test:
        require(sample.kind == "real", "Synthetic record entered the test set")
        observed = sha256_file(resolve_sample_path(paths, sample))
        require(observed == sample.sha256, f"Test sample hash changed: {sample.sample_id}")
        require(observed in blocked, f"Test sample is absent from frozen blocklist: {sample.sample_id}")
        require(observed not in train_hashes, f"Train/test hash overlap: {sample.sample_id}")
        test_hashes.add(observed)

    if group.mode == "development":
        require(group.validation, "Development run has no validation set")
        require(not group.test, "Development run must not load the test set")
        require(
            all(sample.kind == "real" for sample in group.validation),
            "Validation contains synthetic data",
        )
    else:
        require(not group.validation, "Final refit must not retain validation holdout")
        require(group.test, "Final refit has no frozen test set")

    return {
        "blocklist_hits": 0,
        "train_hashes": len(train_hashes),
        "test_hashes": len(test_hashes),
        "train_test_overlap": 0,
        "validation_is_real_only": all(
            sample.kind == "real" for sample in group.validation
        ),
    }


def build_classification_group(
    paths: Paths,
    config: Mapping[str, Any],
    *,
    group_name: str,
    object_name: str,
    seed: int,
    mode: str,
) -> ClassificationGroup:
    require(object_name in paths.objects, f"Unsupported object: {object_name}")
    require(mode in {"development", "final"}, f"Unsupported run mode: {mode}")
    groups = config.get("groups")
    require(isinstance(groups, dict), "Classifier group config is missing")
    canonical_group, group_config = _resolve_group(groups, group_name)
    if mode == "development":
        require(
            canonical_group == "real_only",
            "Only Real-only may be used for hyperparameter development",
        )

    manifest, manifest_sha256 = verify_frozen_manifest(paths.splits)
    selection, selection_sha256 = _load_selection(paths, manifest_sha256)
    manifest_records = manifest.get("images")
    require(isinstance(manifest_records, list), "Manifest images are missing")
    manifest_index, _ = _manifest_index(manifest)
    seed_paths = _selection_paths(selection, object_name, "fewshot_seed")
    validation_paths = _selection_paths(selection, object_name, "validation")
    real_bad = str(group_config.get("real_bad", "seed10"))
    train = _real_training_records(
        manifest_records,
        object_name=object_name,
        real_bad=real_bad,
        seed_paths=seed_paths,
        validation_paths=validation_paths,
        mode=mode,
    )

    synthetic_config = group_config.get("synthetic")
    if synthetic_config is not None:
        require(isinstance(synthetic_config, dict), "Synthetic group config is invalid")
        view = str(synthetic_config["view"])
        inputs_value = synthetic_config.get("inputs")
        inputs = None if inputs_value is None else {str(value) for value in inputs_value}
        count = int(synthetic_config["count"])
        raw_records = _load_synthetic_records(
            paths,
            view=view,
            object_name=object_name,
            inputs=inputs,
        )
        selected = stratified_select(
            raw_records,
            count=count,
            seed=seed,
            group_name=canonical_group,
            object_name=object_name,
        )
        blocklist = load_json(paths.splits / "test_blocklist.json")
        blocked = {str(value) for value in blocklist["sha256"]}
        train.extend(
            _synthetic_sample(
                paths,
                view,
                raw,
                image_path,
                input_name,
                manifest_index,
                blocked,
            )
            for raw, image_path, input_name in selected
        )

    validation = (
        _real_evaluation_records(
            manifest_records,
            object_name=object_name,
            selected_paths=validation_paths,
            partition="validation",
        )
        if mode == "development"
        else []
    )
    test = (
        _real_evaluation_records(
            manifest_records,
            object_name=object_name,
            selected_paths=None,
            partition="test",
        )
        if mode == "final"
        else []
    )
    group = ClassificationGroup(
        requested_group=group_name,
        canonical_group=canonical_group,
        object_name=object_name,
        mode=mode,
        standard_augmentation=bool(group_config.get("standard_augmentation", False)),
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
        manifest_sha256=manifest_sha256,
        selection_sha256=selection_sha256,
    )
    validate_group(paths, group)
    return group


def summarize_samples(samples: Sequence[ClassificationSample]) -> dict[str, Any]:
    labels: dict[str, int] = defaultdict(int)
    kinds: dict[str, int] = defaultdict(int)
    sources: dict[str, int] = defaultdict(int)
    for sample in samples:
        labels["bad" if sample.label else "good"] += 1
        kinds[sample.kind] += 1
        sources[sample.source_name] += 1
    return {
        "total": len(samples),
        "labels": dict(sorted(labels.items())),
        "kinds": dict(sorted(kinds.items())),
        "sources": dict(sorted(sources.items())),
    }


def group_payload_sha256(group: ClassificationGroup) -> str:
    payload = json.dumps(
        group.payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_group_does_not_hit_blocklist(paths: Paths, group: ClassificationGroup) -> None:
    """Compatibility wrapper with the shared guard's failure type."""
    try:
        validate_group(paths, group)
    except ClassificationDataError as exc:
        raise IntegrityError(str(exc)) from exc
