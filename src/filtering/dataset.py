"""Read M13 inputs and publish lossless filtered/unfiltered views."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.paths import Paths
from src.synthetic.metadata import validate_metadata

SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


class FilterDatasetError(RuntimeError):
    """The filter input or output violated its provenance contract."""


@dataclass(frozen=True, slots=True)
class FilterSample:
    """One synthetic record together with its immutable source root."""

    input_name: str
    source_root: Path
    record: dict[str, Any]

    @property
    def key(self) -> str:
        prefix = SAFE_NAME.sub("_", self.input_name).strip("_")
        sample_id = SAFE_NAME.sub("_", str(self.record["sample_id"])).strip("_")
        return f"{prefix}__{sample_id}"

    @property
    def image_path(self) -> Path:
        return (self.source_root / self.record["image_path"]).resolve(strict=True)

    @property
    def mask_path(self) -> Path:
        return (self.source_root / self.record["mask_path"]).resolve(strict=True)


def input_root(paths: Paths, input_name: str) -> Path:
    """Resolve a configured input below the synthetic root without path escape."""

    candidate = (paths.synthetic / input_name).resolve(strict=True)
    try:
        candidate.relative_to(paths.synthetic.resolve(strict=True))
    except ValueError as error:
        raise FilterDatasetError(f"Input escapes synthetic root: {input_name}") from error
    if not candidate.is_dir():
        raise FilterDatasetError(f"Filter input is not a directory: {candidate}")
    return candidate


def read_filter_inputs(
    paths: Paths,
    input_names: Sequence[str],
    *,
    objects: set[str] | None = None,
    limit_per_input: int | None = None,
) -> list[FilterSample]:
    """Read configured JSONL inputs in deterministic order."""

    if limit_per_input is not None and limit_per_input < 1:
        raise FilterDatasetError("limit_per_input must be positive")
    samples: list[FilterSample] = []
    seen: set[tuple[str, str]] = set()
    for input_name in input_names:
        root = input_root(paths, input_name)
        metadata = root / "metadata.jsonl"
        if not metadata.is_file():
            raise FilterDatasetError(f"Missing metadata: {metadata}")
        retained = 0
        with metadata.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                    validate_metadata(record)
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise FilterDatasetError(
                        f"Invalid metadata at {metadata}:{line_number}: {error}"
                    ) from error
                if objects is not None and record["object"] not in objects:
                    continue
                identity = (input_name, str(record["sample_id"]))
                if identity in seen:
                    raise FilterDatasetError(f"Duplicate input identity: {identity}")
                seen.add(identity)
                sample = FilterSample(input_name=input_name, source_root=root, record=record)
                if not sample.image_path.is_file() or not sample.mask_path.is_file():
                    raise FilterDatasetError(f"Missing image or mask for {identity}")
                samples.append(sample)
                retained += 1
                if limit_per_input is not None and retained >= limit_per_input:
                    break
    return samples


def filtered_record(
    sample: FilterSample,
    *,
    scores: Mapping[str, float | int | str | None],
    reject_reasons: Sequence[str],
    thresholds: Mapping[str, float],
    pipeline_version: str,
) -> dict[str, Any]:
    """Create a schema-compatible record containing a complete filter decision."""

    record = deepcopy(sample.record)
    suffix = sample.image_path.suffix.lower()
    mask_suffix = sample.mask_path.suffix.lower()
    record["image_path"] = f"images/{sample.key}{suffix}"
    record["mask_path"] = f"masks/{sample.key}{mask_suffix}"
    record["filter"] = {
        "schema_version": 1,
        "pipeline_version": pipeline_version,
        "passed": not reject_reasons,
        "reject_reasons": list(reject_reasons),
        "scores": dict(scores),
        "thresholds": dict(thresholds),
        "input_name": sample.input_name,
        "input_image_path": sample.record["image_path"],
        "input_mask_path": sample.record["mask_path"],
    }
    validate_metadata(record)
    return record


def _link(source: Path, destination: Path, *, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise FilterDatasetError(f"Refusing to replace unrelated output: {destination}")
    if mode != "hardlink":
        raise FilterDatasetError(f"Unsupported link mode: {mode}")
    try:
        os.link(source, destination)
    except OSError as error:
        raise FilterDatasetError(
            f"Could not hardlink {source} -> {destination}: {error}"
        ) from error


def _write_jsonl_atomic(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FilterDatasetError(f"Temporary metadata already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_views(
    paths: Paths,
    samples: Sequence[FilterSample],
    records: Sequence[dict[str, Any]],
    *,
    filtered_name: str,
    unfiltered_name: str,
    link_mode: str,
) -> tuple[Path, Path]:
    """Publish all decisions and the accepted subset using hardlinked payloads."""

    if len(samples) != len(records):
        raise FilterDatasetError("Sample and record counts differ")
    filtered_root = (paths.synthetic / filtered_name).resolve(strict=False)
    unfiltered_root = (paths.synthetic / unfiltered_name).resolve(strict=False)
    synthetic_root = paths.synthetic.resolve(strict=True)
    for root in (filtered_root, unfiltered_root):
        try:
            root.relative_to(synthetic_root)
        except ValueError as error:
            raise FilterDatasetError(f"Output escapes synthetic root: {root}") from error
    if filtered_root == unfiltered_root:
        raise FilterDatasetError("Filtered and unfiltered roots must differ")

    accepted: list[dict[str, Any]] = []
    for sample, record in zip(samples, records, strict=True):
        for root in (unfiltered_root,):
            _link(sample.image_path, root / record["image_path"], mode=link_mode)
            _link(sample.mask_path, root / record["mask_path"], mode=link_mode)
        if bool(record["filter"]["passed"]):
            _link(sample.image_path, filtered_root / record["image_path"], mode=link_mode)
            _link(sample.mask_path, filtered_root / record["mask_path"], mode=link_mode)
            accepted.append(record)

    _write_jsonl_atomic(unfiltered_root / "metadata.jsonl", records)
    _write_jsonl_atomic(filtered_root / "metadata.jsonl", accepted)
    return filtered_root, unfiltered_root
