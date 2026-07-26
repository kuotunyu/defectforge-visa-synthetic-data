"""Independently validate an M8 procedural Stage A dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_synthetic import (
    read_records,
    safe_output_path,
    validate_tree_inventory,
)
from src.common.imaging import detect_legal_roi, mask_bbox
from src.common.integrity import IntegrityError, load_json, sha256_file, verify_frozen_manifest
from src.common.paths import load_paths
from src.synthetic.copy_paste import ValidationError, load_config, object_roi_config
from src.synthetic.metadata import MetadataError
from src.synthetic.procedural import (
    even_schedule,
    install_forbidden_stats_guard,
    load_shape_bounds,
    parse_shapes,
)

LOGGER = logging.getLogger("validate_procedural")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument("--config", default="configs/stage_a.yaml", type=Path)
    parser.add_argument("--out-name")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--shapes")
    parser.add_argument("--no-real-stats", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def validate_dataset(
    *,
    paths_config: Path,
    stage_config: Path,
    output_name: str,
    n: int,
    shapes: tuple[str, ...],
    no_real_stats: bool,
) -> dict[str, Any]:
    paths = load_paths(paths_config)
    config = load_config(stage_config)
    manifest, manifest_sha256 = verify_frozen_manifest(paths.splits)
    if no_real_stats:
        install_forbidden_stats_guard(paths.reports / "real_mask_stats.json")
    bounds = load_shape_bounds(
        paths=paths,
        config=config,
        objects=paths.objects,
        no_real_stats=no_real_stats,
        manifest_sha256=manifest_sha256,
    )
    output_root = paths.synthetic / output_name
    records = read_records(output_root / "metadata.jsonl")
    expected_total = n * len(paths.objects)
    if len(records) != expected_total:
        raise ValidationError(f"Expected {expected_total} records, observed {len(records)}")
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValidationError("Duplicate sample_id in procedural metadata")
    validate_tree_inventory(output_root, records)

    blocklist = set(load_json(paths.splits / "test_blocklist.json")["sha256"])
    backgrounds = {
        (str(record["object"]), str(record["image_path"])): record
        for record in manifest["images"]
        if record["set"] == "train" and record["label"] == "good"
    }
    stats_mode = "no_real_stats" if no_real_stats else "real_stats"
    expected_counts = Counter(even_schedule(shapes, n))
    observed_counts: dict[str, Counter[str]] = {}
    for object_name in paths.objects:
        observed_counts[object_name] = Counter(
            str(record["defect_type"]) for record in records if record["object"] == object_name
        )
        if observed_counts[object_name] != expected_counts:
            raise ValidationError(
                f"{object_name} shape quota mismatch: "
                f"{observed_counts[object_name]} != {expected_counts}"
            )

    roi_cache: dict[tuple[str, str], np.ndarray] = {}
    background_sha_cache: dict[str, str] = {}
    output_sha: set[str] = set()
    outliers: dict[str, dict[str, int]] = {
        object_name: {"area_ratio": 0, "aspect_ratio": 0} for object_name in paths.objects
    }
    for index, record in enumerate(records, start=1):
        object_name = str(record["object"])
        if (
            object_name not in paths.objects
            or record["generator"] != "stageA_procedural"
            or record["bucket"] != stats_mode
        ):
            raise ValidationError(f"Unexpected object/generator/bucket: {record['sample_id']}")
        source = record["source"]
        if any(
            source[field] is not None
            for field in (
                "defect_source_image",
                "defect_source_mask",
                "defect_source_component_id",
            )
        ):
            raise ValidationError(f"Procedural record names a real defect: {record['sample_id']}")
        background_key = (object_name, str(source["background_image"]))
        background_record = backgrounds.get(background_key)
        if background_record is None:
            raise ValidationError(f"Background is not frozen train-good: {record['sample_id']}")
        if source["background_sha256"] != background_record["sha256"]:
            raise ValidationError(f"Background manifest SHA mismatch: {record['sample_id']}")
        background_relative = str(source["background_image"])
        if background_relative not in background_sha_cache:
            digest = sha256_file(paths.visa_raw / background_relative)
            if digest in blocklist:
                raise ValidationError(f"Background hit test blocklist: {background_relative}")
            background_sha_cache[background_relative] = digest
        if background_sha_cache[background_relative] != source["background_sha256"]:
            raise ValidationError(f"Background file changed: {record['sample_id']}")

        image_path = safe_output_path(output_root, str(record["image_path"]), "images")
        mask_path = safe_output_path(output_root, str(record["mask_path"]), "masks")
        with Image.open(image_path) as image_handle:
            image_size = image_handle.size
        with Image.open(mask_path) as mask_handle:
            mask = np.asarray(mask_handle.convert("L"))
            mask_size = mask_handle.size
        if image_size != mask_size or not np.any(mask):
            raise ValidationError(f"Invalid image/mask pair: {record['sample_id']}")
        if not {int(value) for value in np.unique(mask)}.issubset({0, 255}):
            raise ValidationError(f"Mask is not binary: {record['sample_id']}")
        mask_bool = mask > 0
        placement = record["placement"]
        observed_bbox = list(mask_bbox(mask_bool))
        if observed_bbox != placement["mask_bbox"]:
            raise ValidationError(f"Mask bbox mismatch: {record['sample_id']}")
        area = int(mask_bool.sum())
        area_ratio = float(area / mask_bool.size)
        aspect_ratio = float(observed_bbox[2] / observed_bbox[3])
        if area != int(placement["mask_area_px"]) or not np.isclose(
            area_ratio,
            float(placement["mask_area_ratio"]),
            atol=1e-12,
        ):
            raise ValidationError(f"Mask geometry mismatch: {record['sample_id']}")
        if (
            not bounds[object_name]["area_ratio"][0]
            <= area_ratio
            <= bounds[object_name]["area_ratio"][1]
        ):
            outliers[object_name]["area_ratio"] += 1
        if (
            not bounds[object_name]["aspect_ratio"][0]
            <= aspect_ratio
            <= bounds[object_name]["aspect_ratio"][1]
        ):
            outliers[object_name]["aspect_ratio"] += 1

        if background_key not in roi_cache:
            with Image.open(paths.visa_raw / background_relative) as background_handle:
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
        for output_path in (image_path, mask_path):
            digest = sha256_file(output_path)
            if digest in blocklist:
                raise ValidationError(f"Output hit test blocklist: {output_path}")
            output_sha.add(digest)
        if index % 100 == 0:
            LOGGER.info("Validated %d/%d records", index, len(records))

    outlier_rates = {
        object_name: {metric: count / n for metric, count in metric_counts.items()}
        for object_name, metric_counts in outliers.items()
    }
    for object_name, metric_rates in outlier_rates.items():
        if any(rate >= 0.1 for rate in metric_rates.values()):
            raise ValidationError(f"{object_name} procedural outlier rate >=10%: {metric_rates}")
    _, final_manifest_sha256 = verify_frozen_manifest(paths.splits)
    if final_manifest_sha256 != manifest_sha256:
        raise ValidationError("Frozen manifest changed during procedural validation")
    return {
        "output": str(output_root),
        "records": len(records),
        "stats_mode": stats_mode,
        "real_stats_access_guard": no_real_stats,
        "shape_counts": {
            object_name: dict(sorted(counts.items()))
            for object_name, counts in observed_counts.items()
        },
        "bounds": bounds,
        "outlier_rates": outlier_rates,
        "unique_background_rois_rebuilt": len(roi_cache),
        "unique_background_files_hashed": len(background_sha_cache),
        "unique_output_hashes": len(output_sha),
        "manifest_sha256": manifest_sha256,
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
        if args.n < 1:
            raise ValidationError("n must be positive")
        config = load_config(args.config)
        shapes = parse_shapes(args.shapes) if args.shapes else tuple(config["procedural"]["shapes"])
        output_name = args.out_name or (
            "stageA_procedural_norealstats" if args.no_real_stats else "stageA_procedural"
        )
        summary = validate_dataset(
            paths_config=args.paths,
            stage_config=args.config,
            output_name=output_name,
            n=args.n,
            shapes=shapes,
            no_real_stats=args.no_real_stats,
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
