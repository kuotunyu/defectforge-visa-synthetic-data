"""Independently validate a Stage A copy-paste dataset and its provenance."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.imaging import detect_legal_roi, mask_bbox
from src.common.integrity import (
    IntegrityError,
    load_json,
    sha256_file,
    verify_frozen_manifest,
)
from src.common.paths import load_paths
from src.synthetic.copy_paste import (
    ValidationError,
    allocate_quotas,
    load_config,
    object_roi_config,
)
from src.synthetic.metadata import MetadataError, validate_metadata

LOGGER = logging.getLogger("validate_synthetic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument("--config", default="configs/stage_a.yaml", type=Path)
    parser.add_argument("--out-name", default="stageA_copypaste")
    parser.add_argument("--n", type=int, default=500, help="Expected samples per object.")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise MetadataError("record must be a JSON object")
                validate_metadata(record)
            except (json.JSONDecodeError, MetadataError) as error:
                raise ValidationError(f"{path}:{line_number}: {error}") from error
            records.append(record)
    return records


def safe_output_path(root: Path, relative: str, expected_parent: str) -> Path:
    path = (root / relative).resolve(strict=True)
    expected_root = (root / expected_parent).resolve(strict=True)
    if not path.is_relative_to(expected_root) or path.suffix.lower() != ".png":
        raise ValidationError(f"Unsafe or misplaced output path: {relative!r}")
    return path


def validate_tree_inventory(
    output_root: Path,
    records: list[dict[str, Any]],
) -> None:
    for key, directory in (("image_path", "images"), ("mask_path", "masks")):
        expected = {Path(record[key]).as_posix() for record in records}
        observed = {
            path.relative_to(output_root).as_posix()
            for path in (output_root / directory).glob("*.png")
        }
        if observed != expected:
            raise ValidationError(
                f"{directory} inventory mismatch: "
                f"missing={len(expected - observed)}, extra={len(observed - expected)}"
            )


def expected_type_counts(
    defect_types: dict[str, Any],
    objects: tuple[str, ...],
    n: int,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for object_name in objects:
        type_sizes = {
            int(record["cluster_id"]): int(record["n_components"])
            for record in defect_types["objects"][object_name]["types"]
        }
        minimum = 50 if n >= 50 * len(type_sizes) else 1
        result[object_name] = {
            f"type{cluster_id}": quota
            for cluster_id, quota in allocate_quotas(type_sizes, n, minimum=minimum).items()
        }
    return result


def source_component_keys(
    defect_types: dict[str, Any],
    objects: tuple[str, ...],
) -> set[tuple[str, str, str, str, int]]:
    return {
        (
            object_name,
            f"type{int(type_record['cluster_id'])}",
            str(component["image_path"]),
            str(component["mask_path"]),
            int(component["component_id"]),
        )
        for object_name in objects
        for type_record in defect_types["objects"][object_name]["types"]
        for component in type_record["components"]
    }


def validate_dataset(
    *,
    paths_config: Path,
    stage_config: Path,
    out_name: str,
    n: int,
) -> dict[str, Any]:
    if n < 1:
        raise ValidationError("n must be positive")
    paths = load_paths(paths_config)
    config = load_config(stage_config)
    output_root = paths.synthetic / out_name
    records = read_records(output_root / "metadata.jsonl")
    objects = paths.objects
    expected_total = n * len(objects)
    if len(records) != expected_total:
        raise ValidationError(f"Expected {expected_total} records, observed {len(records)}")

    sample_ids = [str(record["sample_id"]) for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValidationError("Duplicate sample_id in metadata")
    image_paths = [str(record["image_path"]) for record in records]
    mask_paths = [str(record["mask_path"]) for record in records]
    if len(set(image_paths)) != len(image_paths) or len(set(mask_paths)) != len(mask_paths):
        raise ValidationError("Duplicate output path in metadata")
    validate_tree_inventory(output_root, records)

    manifest, manifest_sha256 = verify_frozen_manifest(paths.splits)
    defect_types_path = paths.splits / "defect_types.json"
    defect_types_sha256 = sha256_file(defect_types_path)
    defect_types = load_json(defect_types_path)
    if defect_types.get("manifest_sha256") != manifest_sha256:
        raise ValidationError("Defect types point to another frozen manifest")
    blocklist = set(load_json(paths.splits / "test_blocklist.json")["sha256"])
    backgrounds = {
        (str(record["object"]), str(record["image_path"])): record
        for record in manifest["images"]
        if record["set"] == "train" and record["label"] == "good"
    }
    source_keys = source_component_keys(defect_types, objects)
    expected_counts = expected_type_counts(defect_types, objects, n)
    observed_counts = {
        object_name: Counter(
            str(record["defect_type"]) for record in records if record["object"] == object_name
        )
        for object_name in objects
    }
    for object_name in objects:
        if observed_counts[object_name] != expected_counts[object_name]:
            raise ValidationError(
                f"{object_name} quota mismatch: "
                f"observed={observed_counts[object_name]}, "
                f"expected={expected_counts[object_name]}"
            )

    roi_cache: dict[tuple[str, str], np.ndarray] = {}
    source_sha_cache: dict[str, str] = {}
    output_sha: set[str] = set()
    blend_counts: Counter[str] = Counter()
    for index, record in enumerate(records, start=1):
        object_name = str(record["object"])
        if object_name not in objects or record["generator"] != "stageA_copypaste":
            raise ValidationError(f"Unexpected object/generator in {record['sample_id']}")
        source = record["source"]
        background_key = (object_name, str(source["background_image"]))
        background_record = backgrounds.get(background_key)
        if background_record is None:
            raise ValidationError(
                f"Background is not frozen train-good data: {source['background_image']}"
            )
        if source["background_sha256"] != background_record["sha256"]:
            raise ValidationError(f"Background manifest SHA mismatch in {record['sample_id']}")
        component_key = (
            object_name,
            str(record["defect_type"]),
            str(source["defect_source_image"]),
            str(source["defect_source_mask"]),
            int(source["defect_source_component_id"]),
        )
        if component_key not in source_keys:
            raise ValidationError(f"Unknown frozen defect component in {record['sample_id']}")

        for source_path in (
            str(source["background_image"]),
            str(source["defect_source_image"]),
            str(source["defect_source_mask"]),
        ):
            if source_path not in source_sha_cache:
                digest = sha256_file(paths.visa_raw / source_path)
                if digest in blocklist:
                    raise ValidationError(f"Source data hit test blocklist: {source_path}")
                source_sha_cache[source_path] = digest
        if source_sha_cache[str(source["background_image"])] != source["background_sha256"]:
            raise ValidationError(f"Background file changed for {record['sample_id']}")

        image_path = safe_output_path(output_root, str(record["image_path"]), "images")
        mask_path = safe_output_path(output_root, str(record["mask_path"]), "masks")
        with Image.open(image_path) as image_handle:
            image_size = image_handle.size
        with Image.open(mask_path) as mask_handle:
            mask = np.asarray(mask_handle.convert("L"))
            mask_size = mask_handle.size
        if image_size != mask_size or not np.any(mask):
            raise ValidationError(f"Invalid image/mask pair: {record['sample_id']}")
        unique_mask = {int(value) for value in np.unique(mask)}
        if not unique_mask.issubset({0, 255}):
            raise ValidationError(f"Mask is not binary: {record['sample_id']}")
        mask_bool = mask > 0
        placement = record["placement"]
        if list(mask_bbox(mask_bool)) != placement["mask_bbox"]:
            raise ValidationError(f"Mask bbox mismatch: {record['sample_id']}")
        area = int(mask_bool.sum())
        if area != int(placement["mask_area_px"]):
            raise ValidationError(f"Mask area mismatch: {record['sample_id']}")
        if not np.isclose(
            area / mask_bool.size,
            float(placement["mask_area_ratio"]),
            atol=1e-12,
        ):
            raise ValidationError(f"Mask area ratio mismatch: {record['sample_id']}")

        if background_key not in roi_cache:
            with Image.open(paths.visa_raw / background_key[1]) as background_handle:
                background = np.asarray(background_handle.convert("RGB"))
            roi_cache[background_key] = detect_legal_roi(
                background,
                **object_roi_config(config, object_name),
            )
        legal_roi = roi_cache[background_key]
        if image_size != (legal_roi.shape[1], legal_roi.shape[0]):
            raise ValidationError(f"Background/output size mismatch: {record['sample_id']}")
        if list(mask_bbox(legal_roi)) != placement["roi_bbox"]:
            raise ValidationError(f"ROI bbox mismatch: {record['sample_id']}")
        if not bool(np.all(legal_roi[mask_bool])):
            raise ValidationError(f"Mask escapes legal ROI: {record['sample_id']}")

        try:
            datetime.fromisoformat(str(record["created_at"]))
        except ValueError as error:
            raise ValidationError(f"Invalid created_at: {record['sample_id']}") from error
        for output_path in (image_path, mask_path):
            digest = sha256_file(output_path)
            if digest in blocklist:
                raise ValidationError(f"Output hit test blocklist: {output_path}")
            output_sha.add(digest)
        blend_counts[str(record["generation"]["blend"])] += 1
        if index % 100 == 0:
            LOGGER.info("Validated %d/%d records", index, len(records))

    _, final_manifest_sha256 = verify_frozen_manifest(paths.splits)
    if final_manifest_sha256 != manifest_sha256:
        raise ValidationError("Frozen manifest changed during validation")
    if sha256_file(defect_types_path) != defect_types_sha256:
        raise ValidationError("Defect types changed during validation")
    return {
        "output": str(output_root),
        "records": len(records),
        "objects": {
            object_name: dict(sorted(observed_counts[object_name].items()))
            for object_name in objects
        },
        "blends": dict(sorted(blend_counts.items())),
        "unique_source_files_hashed": len(source_sha_cache),
        "unique_background_rois_rebuilt": len(roi_cache),
        "unique_output_hashes": len(output_sha),
        "manifest_sha256": manifest_sha256,
        "defect_types_sha256": defect_types_sha256,
        "test_blocklist_hits": 0,
        "status": "passed",
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        summary = validate_dataset(
            paths_config=args.paths,
            stage_config=args.config,
            out_name=args.out_name,
            n=args.n,
        )
        payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if args.report is not None:
            report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    except (
        IntegrityError,
        MetadataError,
        OSError,
        ValidationError,
        json.JSONDecodeError,
    ) as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
