"""Independently validate M9 SDG-ready clean-image and mask placements."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.imaging import mask_bbox
from src.common.integrity import (
    IntegrityError,
    load_json,
    sha256_file,
    verify_frozen_manifest,
)
from src.common.paths import load_paths
from src.synthetic.copy_paste import ValidationError, allocate_quotas
from src.synthetic.mask_placement import (
    legal_roi,
    load_config,
    load_or_extract_dino_scores,
    placement_seed,
    shape_bounds,
    validate_placement_record,
    weighted_schedule,
)

LOGGER = logging.getLogger("validate_placements")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument("--config", default="configs/placement.yaml", type=Path)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--out-name", default="placements")
    parser.add_argument("--n-per-image", type=int, default=3)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--roi-method", choices=("otsu", "dinov2", "intersect"), default="intersect"
    )
    parser.add_argument("--limit-backgrounds", type=int)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValidationError("record must be an object")
                validate_placement_record(record)
            except (json.JSONDecodeError, ValidationError) as error:
                raise ValidationError(f"{path}:{line_number}: {error}") from error
            result.append(record)
    return result


def safe_mask_path(object_root: Path, relative: str) -> Path:
    path = (object_root / relative).resolve(strict=True)
    expected_root = (object_root / "masks").resolve(strict=True)
    if not path.is_relative_to(expected_root) or path.suffix.lower() != ".png":
        raise ValidationError(f"Unsafe placement mask path: {relative!r}")
    return path


def expected_backgrounds(
    manifest: dict[str, Any],
    objects: tuple[str, ...],
    limit: int | None,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for object_name in objects:
        records = sorted(
            (
                record
                for record in manifest["images"]
                if record["object"] == object_name
                and record["set"] == "train"
                and record["label"] == "good"
            ),
            key=lambda record: record["image_path"],
        )
        result[object_name] = records[:limit] if limit is not None else records
    return result


def expected_layout(
    defect_types: dict[str, Any],
    backgrounds: dict[str, list[dict[str, Any]]],
    objects: tuple[str, ...],
    n_per_image: int,
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    dict[str, dict[int, int]],
]:
    expected: dict[tuple[str, str, int], dict[str, Any]] = {}
    quotas: dict[str, dict[int, int]] = {}
    for object_name in objects:
        type_records = defect_types["objects"][object_name]["types"]
        sizes = {
            int(type_record["cluster_id"]): int(type_record["n_components"])
            for type_record in type_records
        }
        total = len(backgrounds[object_name]) * n_per_image
        minimum = 50 if total >= 50 * len(sizes) else 1
        quotas[object_name] = allocate_quotas(sizes, total, minimum=minimum)
        schedule = weighted_schedule(quotas[object_name])
        type_metadata = {
            int(type_record["cluster_id"]): type_record for type_record in type_records
        }
        for background_index, background in enumerate(backgrounds[object_name]):
            stem = Path(background["image_path"]).stem
            for variant_index in range(n_per_image):
                global_index = background_index * n_per_image + variant_index
                cluster_id = schedule[global_index]
                expected[(object_name, background["image_path"], variant_index)] = {
                    "placement_id": (
                        f"{object_name}__type{cluster_id}__{stem}__{variant_index:02d}"
                    ),
                    "cluster_id": cluster_id,
                    "trigger_token": type_metadata[cluster_id]["trigger_token"],
                    "global_index": global_index,
                }
    return expected, quotas


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


def validate_source_file(
    relative: str,
    *,
    paths: Any,
    manifest_by_image: dict[str, dict[str, Any]],
    blocklist: set[str],
    sha_cache: dict[str, str],
    is_mask: bool,
) -> None:
    if relative not in sha_cache:
        digest = sha256_file(paths.visa_raw / relative)
        if digest in blocklist:
            raise ValidationError(f"Input hit test blocklist: {relative}")
        sha_cache[relative] = digest
    if is_mask:
        candidates = [
            record for record in manifest_by_image.values() if record.get("mask_path") == relative
        ]
        if len(candidates) != 1 or sha_cache[relative] != candidates[0]["mask_sha256"]:
            raise ValidationError(f"Frozen source mask changed: {relative}")
    else:
        record = manifest_by_image.get(relative)
        if record is None or sha_cache[relative] != record["sha256"]:
            raise ValidationError(f"Frozen source image changed: {relative}")


def validate_dataset(
    *,
    paths_config: Path,
    placement_config: Path,
    objects: tuple[str, ...] | None,
    out_name: str,
    n_per_image: int,
    seed: int | None,
    roi_method: str,
    limit_backgrounds: int | None,
) -> dict[str, Any]:
    if n_per_image < 1:
        raise ValidationError("n-per-image must be positive")
    if limit_backgrounds is not None and limit_backgrounds < 1:
        raise ValidationError("limit-backgrounds must be positive")
    paths = load_paths(paths_config)
    config = load_config(placement_config)
    selected_seed = paths.seed if seed is None else seed
    selected_objects = tuple(objects or paths.objects)
    manifest, manifest_sha256 = verify_frozen_manifest(paths.splits)
    defect_types_path = paths.splits / "defect_types.json"
    defect_types_sha256 = sha256_file(defect_types_path)
    defect_types = load_json(defect_types_path)
    if defect_types["manifest_sha256"] != manifest_sha256:
        raise ValidationError("Defect types point to another manifest")
    backgrounds = expected_backgrounds(manifest, selected_objects, limit_backgrounds)
    expected, quotas = expected_layout(
        defect_types,
        backgrounds,
        selected_objects,
        n_per_image,
    )
    expected_sources = source_component_keys(defect_types, selected_objects)
    bounds = shape_bounds(paths, selected_objects, manifest_sha256=manifest_sha256)
    all_backgrounds = [
        record for object_name in selected_objects for record in backgrounds[object_name]
    ]
    if roi_method == "otsu":
        score_maps = {
            str(record["image_path"]): np.zeros((1, 1), dtype=np.float32)
            for record in all_backgrounds
        }
        dino_cache_file = None
        dino_revision = None
    else:
        score_maps, dino_cache_file, dino_revision = load_or_extract_dino_scores(
            paths,
            all_backgrounds,
            manifest_sha256=manifest_sha256,
            config=config,
        )

    output_root = paths.synthetic / out_name
    manifest_by_image = {str(record["image_path"]): record for record in manifest["images"]}
    background_by_key = {
        (str(record["object"]), str(record["image_path"])): record for record in all_backgrounds
    }
    blocklist = set(load_json(paths.splits / "test_blocklist.json")["sha256"])
    source_sha_cache: dict[str, str] = {}
    background_sha_cache: dict[str, str] = {}
    roi_cache: dict[tuple[str, str], np.ndarray] = {}
    mask_hashes: set[str] = set()
    observed_keys: set[tuple[str, str, int]] = set()
    observed_ids: set[str] = set()
    observed_quotas: dict[str, Counter[int]] = {}
    sibling_masks: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    total_records = 0

    for object_name in selected_objects:
        object_root = output_root / object_name
        records = read_records(object_root / "placements.jsonl")
        total_records += len(records)
        expected_count = len(backgrounds[object_name]) * n_per_image
        if len(records) != expected_count:
            raise ValidationError(
                f"{object_name}: expected {expected_count} records, observed {len(records)}"
            )
        expected_mask_paths = {str(record["mask_path"]) for record in records}
        observed_mask_paths = {
            path.relative_to(object_root).as_posix()
            for path in (object_root / "masks").glob("*.png")
        }
        if observed_mask_paths != expected_mask_paths:
            raise ValidationError(
                f"{object_name} mask inventory mismatch: "
                f"missing={len(expected_mask_paths - observed_mask_paths)}, "
                f"extra={len(observed_mask_paths - expected_mask_paths)}"
            )
        run_config = load_json(object_root / "run_config.json")
        if (
            run_config["manifest_sha256"] != manifest_sha256
            or run_config["defect_types_sha256"] != defect_types_sha256
            or int(run_config["seed"]) != selected_seed
            or int(run_config["n_per_image"]) != n_per_image
            or run_config["roi_method"] != roi_method
            or int(run_config["expected_records"]) != expected_count
        ):
            raise ValidationError(f"{object_name} run_config does not match this validation")

        observed_quotas[object_name] = Counter()
        for record_index, record in enumerate(records, start=1):
            if record["object"] != object_name or record["roi_method"] != roi_method:
                raise ValidationError(f"Object/ROI mismatch: {record['placement_id']}")
            placement_id = str(record["placement_id"])
            if placement_id in observed_ids:
                raise ValidationError(f"Duplicate placement_id: {placement_id}")
            observed_ids.add(placement_id)
            variant_index = int(record["variant_index"])
            key = (object_name, str(record["background_image"]), variant_index)
            layout = expected.get(key)
            if layout is None or key in observed_keys:
                raise ValidationError(f"Unexpected/duplicate placement slot: {placement_id}")
            observed_keys.add(key)
            cluster_id = int(str(record["defect_type"]).removeprefix("type"))
            if (
                placement_id != layout["placement_id"]
                or cluster_id != layout["cluster_id"]
                or record["trigger_token"] != layout["trigger_token"]
            ):
                raise ValidationError(f"Deterministic schedule mismatch: {placement_id}")
            expected_seed = placement_seed(
                selected_seed,
                object_name,
                layout["global_index"],
            )
            if int(record["seed"]) != expected_seed:
                raise ValidationError(f"Sample-local seed mismatch: {placement_id}")
            observed_quotas[object_name][cluster_id] += 1
            source_key = (
                object_name,
                str(record["defect_type"]),
                str(record["source_image"]),
                str(record["source_mask"]),
                int(record["source_component_id"]),
            )
            if source_key not in expected_sources:
                raise ValidationError(f"Source component is not frozen: {placement_id}")
            validate_source_file(
                str(record["source_image"]),
                paths=paths,
                manifest_by_image=manifest_by_image,
                blocklist=blocklist,
                sha_cache=source_sha_cache,
                is_mask=False,
            )
            validate_source_file(
                str(record["source_mask"]),
                paths=paths,
                manifest_by_image=manifest_by_image,
                blocklist=blocklist,
                sha_cache=source_sha_cache,
                is_mask=True,
            )

            background_key = (object_name, str(record["background_image"]))
            background_record = background_by_key.get(background_key)
            if (
                background_record is None
                or record["background_sha256"] != background_record["sha256"]
            ):
                raise ValidationError(f"Background provenance mismatch: {placement_id}")
            background_relative = background_key[1]
            if background_relative not in background_sha_cache:
                background_digest = sha256_file(paths.visa_raw / background_relative)
                if background_digest in blocklist:
                    raise ValidationError(f"Background hit test blocklist: {placement_id}")
                background_sha_cache[background_relative] = background_digest
            if background_sha_cache[background_relative] != record["background_sha256"]:
                raise ValidationError(f"Background file changed: {placement_id}")

            if background_key not in roi_cache:
                with Image.open(paths.visa_raw / background_relative) as image_handle:
                    image = np.asarray(image_handle.convert("RGB"))
                roi_cache[background_key] = legal_roi(
                    image,
                    score_maps[background_relative],
                    config=config,
                    object_name=object_name,
                    method=roi_method,
                )
            roi = roi_cache[background_key]
            mask_path = safe_mask_path(object_root, str(record["mask_path"]))
            with Image.open(mask_path) as mask_handle:
                mask_array = np.asarray(mask_handle.convert("L"))
                mask_size = mask_handle.size
            if mask_size != (roi.shape[1], roi.shape[0]):
                raise ValidationError(f"Mask/background size mismatch: {placement_id}")
            values = {int(value) for value in np.unique(mask_array)}
            if not values.issubset({0, 255}) or not np.any(mask_array):
                raise ValidationError(f"Mask is not non-empty binary PNG: {placement_id}")
            mask = mask_array > 0
            if not bool(np.all(roi[mask])):
                raise ValidationError(f"Mask escapes recomputed legal ROI: {placement_id}")
            if list(mask_bbox(roi)) != record["roi_bbox"]:
                raise ValidationError(f"ROI bbox mismatch: {placement_id}")
            observed_bbox = list(mask_bbox(mask))
            area = int(mask.sum())
            area_ratio = float(area / mask.size)
            aspect_ratio = float(observed_bbox[2] / observed_bbox[3])
            if (
                observed_bbox != record["mask_bbox"]
                or area != int(record["mask_area_px"])
                or not np.isclose(area_ratio, float(record["mask_area_ratio"]), atol=1e-12)
                or not np.isclose(aspect_ratio, float(record["aspect_ratio"]), atol=1e-12)
            ):
                raise ValidationError(f"Mask geometry metadata mismatch: {placement_id}")
            object_bounds = bounds[object_name]
            if not (
                object_bounds["area_ratio"][0] <= area_ratio <= object_bounds["area_ratio"][1]
                and object_bounds["aspect_ratio"][0]
                <= aspect_ratio
                <= object_bounds["aspect_ratio"][1]
            ):
                raise ValidationError(f"Mask is outside frozen real-stat bounds: {placement_id}")
            affine = record["affine"]
            if not (
                config["transform"]["rotation_deg"][0]
                <= float(affine["rotation_deg"])
                <= config["transform"]["rotation_deg"][1]
                and config["transform"]["scale"][0]
                <= float(affine["scale"])
                <= config["transform"]["scale"][1]
                and isinstance(affine["flip"], bool)
            ):
                raise ValidationError(f"Affine metadata is invalid: {placement_id}")
            digest = sha256_file(mask_path)
            if digest in blocklist:
                raise ValidationError(f"Output hit test blocklist: {placement_id}")
            mask_hashes.add(digest)
            sibling_masks[background_key].append(mask)
            if record_index % 100 == 0:
                LOGGER.info("%s: validated %d/%d", object_name, record_index, len(records))

        if observed_quotas[object_name] != Counter(quotas[object_name]):
            raise ValidationError(
                f"{object_name} type quota mismatch: {observed_quotas[object_name]}"
            )

    if observed_keys != set(expected):
        raise ValidationError(
            f"Placement slot inventory mismatch: missing={len(set(expected) - observed_keys)}"
        )
    for background_key, masks in sibling_masks.items():
        occupied = np.zeros(masks[0].shape, dtype=bool)
        for mask in masks:
            if np.any(occupied & mask):
                raise ValidationError(f"Sibling masks overlap: {background_key}")
            occupied |= mask

    _, final_manifest_sha256 = verify_frozen_manifest(paths.splits)
    if final_manifest_sha256 != manifest_sha256:
        raise ValidationError("Frozen manifest changed during M9 validation")
    if sha256_file(defect_types_path) != defect_types_sha256:
        raise ValidationError("Frozen defect types changed during M9 validation")
    return {
        "output": str(output_root),
        "objects": list(selected_objects),
        "records": total_records,
        "backgrounds": sum(len(records) for records in backgrounds.values()),
        "n_per_image": n_per_image,
        "seed": selected_seed,
        "roi_method": roi_method,
        "type_quotas": {
            object_name: {
                f"type{cluster_id}": count for cluster_id, count in sorted(object_quotas.items())
            }
            for object_name, object_quotas in quotas.items()
        },
        "area_aspect_outliers": 0,
        "overlapping_sibling_pairs": 0,
        "unique_background_rois_rebuilt": len(roi_cache),
        "unique_background_files_hashed": len(background_sha_cache),
        "unique_source_files_hashed": len(source_sha_cache),
        "unique_output_mask_hashes": len(mask_hashes),
        "dino_cache_file": str(dino_cache_file) if dino_cache_file else None,
        "dino_revision": dino_revision,
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
            placement_config=args.config,
            objects=tuple(args.objects) if args.objects else None,
            out_name=args.out_name,
            n_per_image=args.n_per_image,
            seed=args.seed,
            roi_method=args.roi_method,
            limit_backgrounds=args.limit_backgrounds,
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
        OSError,
        ValidationError,
        json.JSONDecodeError,
    ) as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
