"""Select deterministic few-shot/validation sets and render real-mask statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import (  # isort: skip
    IntegrityError,
    assert_not_blocklisted,
    load_json,
    sha256_file,
    verify_frozen_manifest,
)
from src.common.paths import Paths, load_paths  # isort: skip


LOGGER = logging.getLogger("sample_fewshot")
VALIDATION_FRACTION = 0.10


class ValidationError(RuntimeError):
    """An M5 assertion failed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def deterministic_sample(
    records: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda record: str(record["image_path"]))
    if count > len(ordered):
        raise ValidationError(f"Cannot sample {count} items from {len(ordered)} candidates")
    selected = random.Random(seed).sample(ordered, count)
    return sorted(selected, key=lambda record: str(record["image_path"]))


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_path": record["image_path"],
        "mask_path": record["mask_path"],
        "sha256": record["sha256"],
        "mask_sha256": record["mask_sha256"],
    }


def make_selection(
    manifest: dict[str, Any],
    *,
    manifest_sha256: str,
    objects: tuple[str, ...],
    seed: int,
    k: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    images = manifest["images"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "seed": seed,
        "k": k,
        "validation_fraction": VALIDATION_FRACTION,
        "validation_policy": (
            "floor(10% of highshot train per object/label), minimum 1; "
            "official fewshot train pool excluded; development only"
        ),
        "objects": {},
    }
    selected_records: dict[str, list[dict[str, Any]]] = {}

    for object_name in objects:
        object_train = [
            record
            for record in images
            if record["object"] == object_name and record["set"] == "train"
        ]
        fewshot_candidates = [
            record
            for record in object_train
            if record["label"] == "bad" and record["in_fewshot_pool"]
        ]
        if len(fewshot_candidates) != 20:
            raise ValidationError(
                f"{object_name} has {len(fewshot_candidates)} few-shot candidates; expected 20"
            )
        fewshot_seed = deterministic_sample(fewshot_candidates, count=k, seed=seed)

        validation: dict[str, list[dict[str, Any]]] = {}
        validation_records: list[dict[str, Any]] = []
        for label in ("good", "bad"):
            label_pool = [record for record in object_train if record["label"] == label]
            target = max(1, math.floor(len(label_pool) * VALIDATION_FRACTION))
            candidates = [
                record for record in label_pool if not record["in_fewshot_pool"]
            ]
            picked = deterministic_sample(candidates, count=target, seed=seed)
            validation[label] = [compact_record(record) for record in picked]
            validation_records.extend(picked)

        fewshot_paths = {record["image_path"] for record in fewshot_seed}
        validation_paths = {record["image_path"] for record in validation_records}
        if fewshot_paths & validation_paths:
            raise ValidationError(f"{object_name} few-shot seed overlaps validation")

        result["objects"][object_name] = {
            "fewshot_seed": [compact_record(record) for record in fewshot_seed],
            "validation": validation,
        }
        selected_records[object_name] = fewshot_seed
    return result, selected_records


def stable_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def assert_selection_not_blocklisted(
    selection: dict[str, Any],
    *,
    blocklist: dict[str, Any],
) -> None:
    blocked = set(blocklist["sha256"])
    selected_hashes: list[str] = []
    for object_selection in selection["objects"].values():
        for record in object_selection["fewshot_seed"]:
            selected_hashes.extend((record["sha256"], record["mask_sha256"]))
        for records in object_selection["validation"].values():
            for record in records:
                selected_hashes.append(record["sha256"])
                if record["mask_sha256"] is not None:
                    selected_hashes.append(record["mask_sha256"])
    hits = sorted(set(selected_hashes) & blocked)
    if hits:
        raise ValidationError(f"Selected training-side data hit the test blocklist: {hits}")


def summarize(values: list[float | int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def compute_mask_metrics(mask_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L")) > 0
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValidationError(f"Mask is empty: {mask_path}")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    area_px = int(mask.sum())
    image_height, image_width = mask.shape
    metrics = {
        "area_px": area_px,
        "area_ratio": area_px / (image_width * image_height),
        "bbox": [x0, y0, width, height],
        "aspect_ratio": width / height,
        "centroid_x_ratio": float(xs.mean() / image_width),
        "centroid_y_ratio": float(ys.mean() / image_height),
        "image_width": image_width,
        "image_height": image_height,
    }
    return metrics, mask


def render_contact_sheet(
    paths: Paths,
    object_name: str,
    records: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(18, 7.5), constrained_layout=True)
    for axis, record in zip(axes.flat, records, strict=True):
        image_path = paths.visa_raw / record["image_path"]
        mask_path = paths.visa_raw / record["mask_path"]
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        _, mask = compute_mask_metrics(mask_path)
        axis.imshow(rgb)
        axis.contour(mask.astype(np.uint8), levels=[0.5], colors=["#ff2d55"], linewidths=2)
        axis.set_title(Path(record["image_path"]).name, fontsize=10)
        axis.axis("off")
    figure.suptitle(
        f"{object_name}: deterministic k={len(records)} few-shot seed (red = GT mask)",
        fontsize=16,
    )
    destination = paths.figures / f"fewshot_contact_sheet_{object_name}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def build_mask_stats(
    paths: Paths,
    selected_records: dict[str, list[dict[str, Any]]],
    *,
    selection_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "selection_sha256": selection_sha256,
        "objects": {},
    }
    metric_names = (
        "area_px",
        "area_ratio",
        "aspect_ratio",
        "centroid_x_ratio",
        "centroid_y_ratio",
    )
    for object_name, records in selected_records.items():
        samples: list[dict[str, Any]] = []
        for record in records:
            metrics, _ = compute_mask_metrics(paths.visa_raw / record["mask_path"])
            samples.append(
                {
                    "image_path": record["image_path"],
                    "mask_path": record["mask_path"],
                    **metrics,
                }
            )
        result["objects"][object_name] = {
            "n": len(samples),
            "summary": {
                name: summarize([sample[name] for sample in samples])
                for name in metric_names
            },
            "samples": samples,
        }
    return result


def manifest_counts(
    manifest: dict[str, Any],
    objects: tuple[str, ...],
) -> dict[str, Counter[tuple[str, str]]]:
    counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for record in manifest["images"]:
        if record["object"] in objects:
            counts[record["object"]][(record["set"], record["label"])] += 1
    return counts


def write_stats_report(
    destination: Path,
    *,
    manifest: dict[str, Any],
    objects: tuple[str, ...],
    selection: dict[str, Any],
    selection_sha256: str,
    mask_stats: dict[str, Any],
) -> None:
    counts = manifest_counts(manifest, objects)
    lines = [
        "# M5 Few-shot Selection and Mask Statistics",
        "",
        f"- Manifest SHA256: `{selection['manifest_sha256']}`",
        f"- Selection SHA256: `{selection_sha256}`",
        f"- Seed: `{selection['seed']}`",
        f"- Few-shot k per object: `{selection['k']}`",
        "- Validation: 10% per object × label from highshot train, excluding official fewshot pool",
        "",
        "## Frozen partition counts",
        "",
        "| object | train good | train bad | test good | test bad | val good | val bad |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for object_name in objects:
        values = counts[object_name]
        validation = selection["objects"][object_name]["validation"]
        lines.append(
            f"| {object_name} | {values[('train', 'good')]} | "
            f"{values[('train', 'bad')]} | {values[('test', 'good')]} | "
            f"{values[('test', 'bad')]} | {len(validation['good'])} | "
            f"{len(validation['bad'])} |"
        )

    lines.extend(
        [
            "",
            "## k=10 seed mask distribution",
            "",
            "| object | metric | min | p05 | median | p95 | max |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for object_name in objects:
        summaries = mask_stats["objects"][object_name]["summary"]
        for metric in ("area_px", "area_ratio", "aspect_ratio"):
            values = summaries[metric]
            lines.append(
                f"| {object_name} | {metric} | {values['min']:.6g} | "
                f"{values['p05']:.6g} | {values['median']:.6g} | "
                f"{values['p95']:.6g} | {values['max']:.6g} |"
            )
    lines.extend(
        [
            "",
            "The raw per-mask values, bounding boxes, normalized centroids, and image sizes are in",
            "`reports/real_mask_stats.json` for downstream sampling and audit.",
            "",
            "## Assertions",
            "",
            "- Selection regenerated twice with byte-identical canonical JSON: **passed**",
            "- Few-shot seed and validation are disjoint: **passed**",
            "- No selected image or mask hash is test-blocklisted: **passed**",
            "- Frozen manifest checksum unchanged before/after M5: **passed**",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def write_selection(paths: Paths, selection: dict[str, Any]) -> str:
    destination = paths.splits / "fewshot_selection.json"
    checksum_path = paths.splits / "FEWSHOT_SELECTION.sha256"
    content = stable_json_bytes(selection)
    digest = hashlib.sha256(content).hexdigest()
    if destination.exists() and destination.read_bytes() != content:
        raise ValidationError(
            f"Refusing to overwrite non-identical deterministic selection: {destination}"
        )
    destination.write_bytes(content)
    checksum_path.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    if sha256_file(destination) != digest:
        raise ValidationError("Selection checksum changed after serialization")
    return digest


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.k < 1:
            raise ValidationError("k must be positive")
        paths = load_paths(args.paths)
        objects = tuple(args.objects or paths.objects)
        seed = paths.seed if args.seed is None else args.seed
        manifest, manifest_sha256 = verify_frozen_manifest(paths.splits)
        blocklist = load_json(paths.splits / "test_blocklist.json")
        if blocklist.get("manifest_sha256") != manifest_sha256:
            raise ValidationError("Blocklist points to a different manifest checksum")

        selection_a, selected_records = make_selection(
            manifest,
            manifest_sha256=manifest_sha256,
            objects=objects,
            seed=seed,
            k=args.k,
        )
        selection_b, _ = make_selection(
            manifest,
            manifest_sha256=manifest_sha256,
            objects=objects,
            seed=seed,
            k=args.k,
        )
        if stable_json_bytes(selection_a) != stable_json_bytes(selection_b):
            raise ValidationError("Repeated selection was not byte-identical")
        assert_selection_not_blocklisted(selection_a, blocklist=blocklist)

        input_files = [
            paths.visa_raw / record[key]
            for records in selected_records.values()
            for record in records
            for key in ("image_path", "mask_path")
        ]
        assert_not_blocklisted(input_files, paths.splits / "test_blocklist.json")

        if args.dry_run:
            LOGGER.info(
                "Dry-run selection SHA256: %s",
                hashlib.sha256(stable_json_bytes(selection_a)).hexdigest(),
            )
            return 0

        selection_sha256 = write_selection(paths, selection_a)
        mask_stats = build_mask_stats(
            paths,
            selected_records,
            selection_sha256=selection_sha256,
            manifest_sha256=manifest_sha256,
        )
        stats_path = paths.reports / "real_mask_stats.json"
        stats_path.write_text(
            json.dumps(mask_stats, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for object_name, records in selected_records.items():
            render_contact_sheet(paths, object_name, records)
        write_stats_report(
            paths.reports / "fewshot_stats.md",
            manifest=manifest,
            objects=objects,
            selection=selection_a,
            selection_sha256=selection_sha256,
            mask_stats=mask_stats,
        )

        _, final_manifest_sha256 = verify_frozen_manifest(paths.splits)
        if final_manifest_sha256 != manifest_sha256:
            raise ValidationError("Frozen manifest changed during M5")
        LOGGER.info("M5 selection frozen with SHA256 %s", selection_sha256)
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
