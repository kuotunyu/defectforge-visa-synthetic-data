"""Validate and atomically assemble the local Hugging Face release bundles."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_publish import scan_text  # isort: skip
from src.common.integrity import sha256_file  # isort: skip
from src.common.paths import Paths, load_paths  # isort: skip

OBJECTS = ("pcb1", "capsules")
DATASET_SOURCES = (
    ("data/filtered", "filtered", 1770),
    ("data/unfiltered", "unfiltered", 3000),
    (
        "data/ablations/stageA_procedural_norealstats",
        "stageA_procedural_norealstats",
        1000,
    ),
    ("data/diagnostics/stageB_sd2/original", "stageB_sd2/original", 1000),
    ("data/diagnostics/stageB_sdxl/original", "stageB_sdxl/original", 500),
    ("data/diagnostics/stageB_sdxl/searched", "stageB_sdxl/searched", 500),
)
SPLIT_FILES = (
    "split_manifest.json",
    "MANIFEST.sha256",
    "defect_types.json",
    "DEFECT_TYPES.sha256",
    "fewshot_selection.json",
    "FEWSHOT_SELECTION.sha256",
    "test_blocklist.json",
)
MODEL_PAYLOAD_FILES = (
    "trainer_state.json",
    "unet_adapter/adapter_config.json",
    "unet_adapter/adapter_model.safetensors",
    "text_token_adapter/adapter_config.json",
    "text_token_adapter/adapter_model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)
SDXL_EXTRA_FILES = (
    "text_token_adapter_2/adapter_config.json",
    "text_token_adapter_2/adapter_model.safetensors",
    "tokenizer_2/tokenizer.json",
    "tokenizer_2/tokenizer_config.json",
)


class ReleasePackagingError(RuntimeError):
    """Raised when a local release bundle cannot be proven safe and complete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleasePackagingError(message)


@dataclass(frozen=True, slots=True)
class PayloadFile:
    source: Path
    destination: PurePosixPath
    bytes: int
    sha256: str

    def manifest_row(self) -> dict[str, Any]:
        return {
            "path": self.destination.as_posix(),
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


def load_json_object(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"Expected a JSON object: {path}")
    return payload


def safe_relative_path(value: Any, *, field: str) -> PurePosixPath:
    require(isinstance(value, str) and bool(value), f"Invalid metadata field: {field}")
    path = PurePosixPath(value)
    require(not path.is_absolute(), f"Absolute metadata path in {field}: {value}")
    require(".." not in path.parts, f"Parent traversal in {field}: {value}")
    require("\\" not in value and ":" not in value, f"Non-portable metadata path in {field}")
    return path


def read_metadata(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"Missing metadata.jsonl: {path}")
    text = path.read_text(encoding="utf-8")
    require(not scan_text(Path("metadata.jsonl"), text), f"Unsafe text in {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        require(bool(line.strip()), f"Blank metadata line {line_number}: {path}")
        payload = json.loads(line)
        require(isinstance(payload, dict), f"Invalid metadata line {line_number}: {path}")
        rows.append(payload)
    require(bool(rows), f"Empty metadata file: {path}")
    return rows


def inventory_dataset_source(
    source_root: Path,
    *,
    destination_root: PurePosixPath,
    expected_samples: int,
    blocked_sha256: set[str],
) -> tuple[list[PayloadFile], dict[str, Any]]:
    require(source_root.is_dir(), f"Missing dataset source: {source_root}")
    require(not source_root.is_symlink(), f"Dataset source is a symlink: {source_root}")
    image_root = source_root / "images"
    mask_root = source_root / "masks"
    images = sorted(image_root.glob("*.png"))
    masks = sorted(mask_root.glob("*.png"))
    require(len(images) == expected_samples, f"Unexpected image count: {source_root}")
    require(len(masks) == expected_samples, f"Unexpected mask count: {source_root}")
    require(
        {path.name for path in images} == {path.name for path in masks},
        f"Image/mask names differ: {source_root}",
    )
    require(
        not [path for path in image_root.iterdir() if not path.is_file() or path.suffix != ".png"],
        f"Unexpected image payload: {source_root}",
    )
    require(
        not [path for path in mask_root.iterdir() if not path.is_file() or path.suffix != ".png"],
        f"Unexpected mask payload: {source_root}",
    )
    metadata_path = source_root / "metadata.jsonl"
    metadata = read_metadata(metadata_path)
    require(len(metadata) == expected_samples, f"Unexpected metadata count: {source_root}")
    observed_paths: set[tuple[str, str]] = set()
    observed_ids: set[str] = set()
    for row in metadata:
        image_path = safe_relative_path(row.get("image_path"), field="image_path")
        mask_path = safe_relative_path(row.get("mask_path"), field="mask_path")
        require(image_path.parts[0] == "images", "metadata image_path left images/")
        require(mask_path.parts[0] == "masks", "metadata mask_path left masks/")
        require((source_root / Path(*image_path.parts)).is_file(), "Metadata image is missing")
        require((source_root / Path(*mask_path.parts)).is_file(), "Metadata mask is missing")
        key = (image_path.as_posix(), mask_path.as_posix())
        require(key not in observed_paths, f"Duplicate metadata paths: {key}")
        observed_paths.add(key)
        sample_id = str(row.get("sample_id", ""))
        require(bool(sample_id) and sample_id not in observed_ids, "Duplicate or empty sample_id")
        observed_ids.add(sample_id)
        source = row.get("source")
        require(isinstance(source, Mapping), "Metadata source is not an object")
        safe_relative_path(source.get("background_image"), field="source.background_image")
        for field in ("defect_source_image", "defect_source_mask"):
            if source.get(field) is not None:
                safe_relative_path(source[field], field=f"source.{field}")

    files: list[PayloadFile] = []
    blocklist_hits: list[str] = []
    for directory, paths in (("images", images), ("masks", masks)):
        for path in paths:
            require(not path.is_symlink(), f"Dataset payload is a symlink: {path}")
            digest = sha256_file(path)
            if digest in blocked_sha256:
                blocklist_hits.append(f"{directory}/{path.name}")
            files.append(
                PayloadFile(
                    source=path,
                    destination=destination_root / directory / path.name,
                    bytes=path.stat().st_size,
                    sha256=digest,
                )
            )
    require(not blocklist_hits, f"Frozen test blocklist hits: {blocklist_hits[:3]}")
    files.append(
        PayloadFile(
            source=metadata_path,
            destination=destination_root / "metadata.jsonl",
            bytes=metadata_path.stat().st_size,
            sha256=sha256_file(metadata_path),
        )
    )
    return files, {
        "destination": destination_root.as_posix(),
        "samples": expected_samples,
        "files": len(files),
        "bytes": sum(item.bytes for item in files),
        "test_blocklist_hits": 0,
    }


def inventory_model_family(
    *,
    family: str,
    roots: Mapping[str, Path],
    validation: Mapping[str, Any],
) -> tuple[list[PayloadFile], dict[str, Any]]:
    require(validation.get("status") == "passed", f"{family} validation did not pass")
    validation_objects = validation.get("objects")
    require(isinstance(validation_objects, Mapping), f"{family} validation has no objects")
    relative_files = (*MODEL_PAYLOAD_FILES, *(SDXL_EXTRA_FILES if family == "lora_sdxl" else ()))
    files: list[PayloadFile] = []
    summaries: dict[str, Any] = {}
    for object_name in OBJECTS:
        source_root = roots[object_name]
        require(source_root.is_dir(), f"Missing {family}/{object_name} final bundle")
        evidence = validation_objects.get(object_name)
        require(isinstance(evidence, Mapping), f"Missing {family}/{object_name} validation")
        adapter_hashes = evidence.get("adapter_hashes")
        require(isinstance(adapter_hashes, Mapping), "Missing validated adapter hashes")
        object_files: list[PayloadFile] = []
        for relative in relative_files:
            source = source_root / relative
            require(source.is_file() and not source.is_symlink(), f"Missing model file: {source}")
            text = source.read_text(encoding="utf-8") if source.suffix == ".json" else None
            if text is not None:
                require(not scan_text(Path(relative), text), f"Unsafe model JSON: {source}")
            digest = sha256_file(source)
            adapter_key = {
                "unet_adapter/adapter_model.safetensors": "unet_adapter_sha256",
                "text_token_adapter/adapter_model.safetensors": "text_token_adapter_sha256",
                "text_token_adapter_2/adapter_model.safetensors": (
                    "text_token_adapter_2_sha256"
                ),
            }.get(relative)
            if adapter_key is not None:
                require(
                    adapter_hashes.get(adapter_key) == digest,
                    f"Validated adapter hash changed: {family}/{object_name}/{relative}",
                )
            item = PayloadFile(
                source=source,
                destination=PurePosixPath(family) / object_name / relative,
                bytes=source.stat().st_size,
                sha256=digest,
            )
            files.append(item)
            object_files.append(item)
        summaries[object_name] = {
            "files": len(object_files),
            "bytes": sum(item.bytes for item in object_files),
            "adapter_hashes": dict(adapter_hashes),
        }
    return files, summaries


def materialize(files: Sequence[PayloadFile], root: Path) -> dict[str, int]:
    linked = 0
    copied = 0
    destinations: set[str] = set()
    for item in files:
        relative = item.destination.as_posix()
        require(relative not in destinations, f"Duplicate release destination: {relative}")
        destinations.add(relative)
        destination = root / Path(*item.destination.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(item.source, destination)
        except OSError:
            shutil.copy2(item.source, destination)
            copied += 1
            require(
                sha256_file(destination) == item.sha256,
                f"Copied file changed: {relative}",
            )
        else:
            linked += 1
            require(os.path.samefile(item.source, destination), f"Hardlink changed: {relative}")
        require(destination.stat().st_size == item.bytes, f"File size changed: {relative}")
    return {"hardlinked": linked, "copied": copied}


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify_release_manifest(root: Path, files: Sequence[PayloadFile]) -> str:
    manifest_path = root / "release_manifest.json"
    manifest = load_json_object(manifest_path)
    require(manifest.get("status") == "passed", f"Release manifest did not pass: {root}")
    observed = manifest.get("files")
    expected = [item.manifest_row() for item in files]
    require(observed == expected, f"Release manifest is stale: {root}")
    return sha256_file(manifest_path)


def build_inventory(paths: Paths) -> tuple[list[PayloadFile], list[PayloadFile], dict[str, Any]]:
    blocklist = load_json_object(paths.splits / "test_blocklist.json")
    blocked_sha256 = {str(value) for value in blocklist.get("sha256", [])}
    require(bool(blocked_sha256), "Frozen test blocklist is empty")
    dataset_files: list[PayloadFile] = []
    dataset_sources: list[dict[str, Any]] = []
    for destination, source, expected_samples in DATASET_SOURCES:
        files, summary = inventory_dataset_source(
            paths.synthetic / Path(*PurePosixPath(source).parts),
            destination_root=PurePosixPath(destination),
            expected_samples=expected_samples,
            blocked_sha256=blocked_sha256,
        )
        dataset_files.extend(files)
        dataset_sources.append(summary)
    dataset_card = paths.project_root / "hf_cards" / "dataset" / "README.md"
    dataset_files.append(
        PayloadFile(
            source=dataset_card,
            destination=PurePosixPath("README.md"),
            bytes=dataset_card.stat().st_size,
            sha256=sha256_file(dataset_card),
        )
    )
    for name in SPLIT_FILES:
        source = paths.splits / name
        require(source.is_file(), f"Missing publish split file: {name}")
        dataset_files.append(
            PayloadFile(
                source=source,
                destination=PurePosixPath("splits") / name,
                bytes=source.stat().st_size,
                sha256=sha256_file(source),
            )
        )

    sd2_validation = load_json_object(paths.reports / "lora_sd2_validation.json")
    sdxl_validation = load_json_object(paths.reports / "lora_sdxl_import_validation.json")
    sd2_roots = {
        object_name: paths.runs / "lora_sd2" / object_name / "seed_42" / "final"
        for object_name in OBJECTS
    }
    sdxl_roots = {
        object_name: paths.project_root
        / "results"
        / "colab"
        / "lora_sdxl"
        / object_name
        / "final"
        for object_name in OBJECTS
    }
    sd2_files, sd2_summary = inventory_model_family(
        family="lora_sd2",
        roots=sd2_roots,
        validation=sd2_validation,
    )
    sdxl_files, sdxl_summary = inventory_model_family(
        family="lora_sdxl",
        roots=sdxl_roots,
        validation=sdxl_validation,
    )
    model_card = paths.project_root / "hf_cards" / "model" / "README.md"
    model_files = [*sd2_files, *sdxl_files]
    model_files.append(
        PayloadFile(
            source=model_card,
            destination=PurePosixPath("README.md"),
            bytes=model_card.stat().st_size,
            sha256=sha256_file(model_card),
        )
    )
    summary = {
        "schema_version": 1,
        "status": "passed",
        "publishes_or_uploads": False,
        "dataset": {
            "unique_generated_samples": sum(item[2] for item in DATASET_SOURCES[1:]),
            "published_sample_views": sum(item[2] for item in DATASET_SOURCES),
            "view_files": len(dataset_files),
            "bytes": sum(item.bytes for item in dataset_files),
            "sources": dataset_sources,
            "test_blocklist_hits": 0,
        },
        "model": {
            "files": len(model_files),
            "bytes": sum(item.bytes for item in model_files),
            "families": {"lora_sd2": sd2_summary, "lora_sdxl": sdxl_summary},
        },
    }
    return dataset_files, model_files, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--validation-out",
        type=Path,
        default=Path("reports/hf_package_validation.json"),
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Create local bundles; omitted means validation and inventory only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_paths(args.paths)
    output_root = (
        args.output_root.resolve(strict=False)
        if args.output_root is not None
        else paths.data_root / "publish"
    )
    dataset_files, model_files, summary = build_inventory(paths)
    summary["mode"] = "build" if args.build else "dry_run"
    summary["output_root"] = str(output_root)
    if args.build:
        require(not output_root.exists(), f"Release output already exists: {output_root}")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        staging = output_root.parent / f".{output_root.name}.working-{uuid.uuid4().hex}"
        require(not staging.exists(), f"Release staging already exists: {staging}")
        try:
            staging.mkdir()
            dataset_root = staging / "hf_dataset"
            model_root = staging / "hf_model"
            dataset_stats = materialize(dataset_files, dataset_root)
            model_stats = materialize(model_files, model_root)
            dataset_manifest = {
                "schema_version": 1,
                "status": "passed",
                "files": [item.manifest_row() for item in dataset_files],
            }
            model_manifest = {
                "schema_version": 1,
                "status": "passed",
                "files": [item.manifest_row() for item in model_files],
            }
            atomic_write_json(dataset_root / "release_manifest.json", dataset_manifest)
            atomic_write_json(model_root / "release_manifest.json", model_manifest)
            os.replace(staging, output_root)
        except BaseException:
            if staging.exists() and staging.parent == output_root.parent:
                shutil.rmtree(staging)
            raise
        summary["materialization"] = {
            "dataset": dataset_stats,
            "model": model_stats,
        }
    if output_root.exists():
        summary["existing_bundle_verified"] = True
        summary["release_manifest_sha256"] = {
            "dataset": verify_release_manifest(
                output_root / "hf_dataset",
                dataset_files,
            ),
            "model": verify_release_manifest(
                output_root / "hf_model",
                model_files,
            ),
        }
    else:
        summary["existing_bundle_verified"] = False
    atomic_write_json(args.validation_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
