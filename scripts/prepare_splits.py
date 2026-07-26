"""Prepare official VisA few-shot/high-shot layouts and validate their invariants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import Paths, load_paths  # isort: skip


LOGGER = logging.getLogger("prepare_splits")
SPLIT_TYPES = ("2cls_fewshot", "2cls_highshot")
EXPECTED_COUNTS = {
    "2cls_fewshot": {
        "pcb1": {
            ("train", "normal"): 201,
            ("train", "anomaly"): 20,
            ("test", "normal"): 803,
            ("test", "anomaly"): 80,
        },
        "capsules": {
            ("train", "normal"): 120,
            ("train", "anomaly"): 20,
            ("test", "normal"): 482,
            ("test", "anomaly"): 80,
        },
    },
    "2cls_highshot": {
        "pcb1": {
            ("train", "normal"): 602,
            ("train", "anomaly"): 60,
            ("test", "normal"): 402,
            ("test", "anomaly"): 40,
        },
        "capsules": {
            ("train", "normal"): 361,
            ("train", "anomaly"): 60,
            ("test", "normal"): 241,
            ("test", "anomaly"): 40,
        },
    },
}


class ValidationError(RuntimeError):
    """An M3 assertion failed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument(
        "--object",
        action="append",
        dest="objects",
        help="Object to validate; repeat for multiple objects (default: paths.yaml objects).",
    )
    parser.add_argument(
        "--split-type",
        choices=(*SPLIT_TYPES, "both"),
        default="both",
    )
    parser.add_argument(
        "--spot-diff-dir",
        type=Path,
        help="Official amazon-science/spot-diff checkout (auto-detected when omitted).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_spot_diff(paths: Paths, requested: Path | None, *, dry_run: bool) -> Path:
    if requested is not None:
        checkout = requested.resolve(strict=False)
    else:
        candidates = (
            paths.data_root / "upstream" / "spot-diff",
            paths.data_root / "tools" / "spot-diff",
        )
        checkout = next((item for item in candidates if item.is_dir()), candidates[-1])

    if checkout.is_dir():
        return checkout
    if dry_run:
        return checkout

    checkout.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Cloning official spot-diff into %s", checkout)
    subprocess.run(
        ["git", "clone", "https://github.com/amazon-science/spot-diff.git", str(checkout)],
        check=True,
    )
    return checkout


def upstream_files(checkout: Path, split_type: str) -> tuple[Path, Path]:
    script = checkout / "utils" / "prepare_data.py"
    split_csv = checkout / "split_csv" / f"{split_type}.csv"
    if not script.is_file():
        raise ValidationError(f"Missing official preparation script: {script}")
    if not split_csv.is_file():
        raise ValidationError(f"Missing official split CSV: {split_csv}")
    return script, split_csv


def read_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"object", "split", "label", "image", "mask"}
    if not rows or set(rows[0]) != required:
        raise ValidationError(f"Unexpected CSV schema in {path}")
    return rows


def validate_source_rows(
    rows_by_split: dict[str, list[dict[str, str]]],
    *,
    objects: tuple[str, ...],
    visa_raw: Path,
) -> dict[str, dict[str, dict[str, int]]]:
    observed: dict[str, dict[str, dict[str, int]]] = {}
    row_sets: dict[str, dict[str, dict[str, set[str]]]] = {}

    for split_type, rows in rows_by_split.items():
        observed[split_type] = {}
        row_sets[split_type] = {}
        for object_name in objects:
            selected = [row for row in rows if row["object"] == object_name]
            counts = Counter((row["split"], row["label"]) for row in selected)
            expected = EXPECTED_COUNTS[split_type][object_name]
            if counts != expected:
                raise ValidationError(
                    f"{split_type}/{object_name} count mismatch: "
                    f"observed={dict(counts)!r}, expected={expected!r}"
                )

            image_sets: dict[str, set[str]] = {"train": set(), "test": set()}
            for row in selected:
                if row["split"] not in image_sets:
                    raise ValidationError(f"Unexpected set in {split_type}: {row!r}")
                if row["label"] not in {"normal", "anomaly"}:
                    raise ValidationError(f"Unexpected label in {split_type}: {row!r}")
                image_path = visa_raw / Path(row["image"])
                if not image_path.is_file():
                    raise ValidationError(f"CSV image is missing: {image_path}")
                image_sets[row["split"]].add(row["image"])
                if row["label"] == "anomaly":
                    if not row["mask"]:
                        raise ValidationError(f"Anomaly row has no mask: {row!r}")
                    mask_path = visa_raw / Path(row["mask"])
                    if not mask_path.is_file():
                        raise ValidationError(f"CSV mask is missing: {mask_path}")

            if image_sets["train"] & image_sets["test"]:
                raise ValidationError(f"{split_type}/{object_name} train/test overlap")

            row_sets[split_type][object_name] = image_sets
            observed[split_type][object_name] = {
                f"{set_name}_{label}": counts[(set_name, label)]
                for set_name in ("train", "test")
                for label in ("normal", "anomaly")
            }

    if set(SPLIT_TYPES) <= rows_by_split.keys():
        for object_name in objects:
            fewshot_train = row_sets["2cls_fewshot"][object_name]["train"]
            highshot_train = row_sets["2cls_highshot"][object_name]["train"]
            if not fewshot_train <= highshot_train:
                extra = sorted(fewshot_train - highshot_train)[:5]
                raise ValidationError(
                    f"{object_name} fewshot train is not a highshot train subset: {extra}"
                )
    return observed


def validate_prepared_tree(
    split_root: Path,
    split_type: str,
    *,
    objects: tuple[str, ...],
) -> None:
    for object_name in objects:
        expected = EXPECTED_COUNTS[split_type][object_name]
        for set_name in ("train", "test"):
            for source_label, prepared_label in (("normal", "good"), ("anomaly", "bad")):
                image_dir = split_root / object_name / set_name / prepared_label
                images = {item.stem for item in image_dir.glob("*") if item.is_file()}
                wanted = expected[(set_name, source_label)]
                if len(images) != wanted:
                    raise ValidationError(
                        f"{image_dir} has {len(images)} files; expected {wanted}"
                    )
                if source_label != "anomaly":
                    continue
                mask_dir = (
                    split_root
                    / object_name
                    / "ground_truth"
                    / set_name
                    / prepared_label
                )
                masks = {item.stem for item in mask_dir.glob("*") if item.is_file()}
                if masks != images:
                    raise ValidationError(
                        f"{split_type}/{object_name}/{set_name} bad image/mask stems differ"
                    )


def git_commit(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_report(
    paths: Paths,
    *,
    checkout: Path,
    rows_observed: dict[str, dict[str, dict[str, int]]],
    file_hashes: dict[str, str],
) -> None:
    report = {
        "upstream": {
            "repository": "https://github.com/amazon-science/spot-diff",
            "commit": git_commit(checkout),
            "sha256": file_hashes,
        },
        "counts": rows_observed,
        "assertions": {
            "eight_expected_counts": "passed",
            "highshot_train_test_disjoint": "passed",
            "fewshot_train_subset_highshot_train": "passed",
            "all_anomaly_images_have_masks": "passed",
            "prepared_tree_matches_csv": "passed",
        },
    }
    destination = paths.reports / "split_preparation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        paths = load_paths(args.paths)
        objects = tuple(args.objects or paths.objects)
        unknown = sorted(set(objects) - EXPECTED_COUNTS["2cls_fewshot"].keys())
        if unknown:
            raise ValidationError(f"Objects have no locked M3 counts: {unknown}")
        split_types = SPLIT_TYPES if args.split_type == "both" else (args.split_type,)
        checkout = resolve_spot_diff(paths, args.spot_diff_dir, dry_run=args.dry_run)

        rows_by_split: dict[str, list[dict[str, str]]] = {}
        file_hashes: dict[str, str] = {}
        official_script: Path | None = None
        for split_type in split_types:
            official_script, split_csv = upstream_files(checkout, split_type)
            rows_by_split[split_type] = read_split_csv(split_csv)
            file_hashes[str(split_csv.relative_to(checkout)).replace("\\", "/")] = (
                sha256_file(split_csv)
            )

        if official_script is None:
            raise ValidationError("No split type selected")
        file_hashes["utils/prepare_data.py"] = sha256_file(official_script)
        observed = validate_source_rows(
            rows_by_split,
            objects=objects,
            visa_raw=paths.visa_raw,
        )

        save_roots = {paths.visa_fewshot.parent, paths.visa_highshot.parent}
        if len(save_roots) != 1:
            raise ValidationError(
                "visa_fewshot and visa_highshot must share an official save-folder parent"
            )
        save_root = save_roots.pop()

        for split_type in split_types:
            official_script, split_csv = upstream_files(checkout, split_type)
            command = [
                sys.executable,
                str(official_script),
                "--split-type",
                split_type,
                "--data-folder",
                str(paths.visa_raw),
                "--save-folder",
                str(save_root),
                "--split-file",
                str(split_csv),
            ]
            if args.dry_run:
                LOGGER.info("Would run: %s", subprocess.list2cmdline(command))
                continue
            LOGGER.info("Preparing %s with the official spot-diff utility", split_type)
            subprocess.run(command, check=True)
            split_root = (
                paths.visa_fewshot
                if split_type == "2cls_fewshot"
                else paths.visa_highshot
            )
            validate_prepared_tree(split_root, split_type, objects=objects)

        if args.dry_run:
            return 0
        write_report(
            paths,
            checkout=checkout,
            rows_observed=observed,
            file_hashes=file_hashes,
        )
        LOGGER.info("M3 split preparation and all assertions passed")
        return 0
    except (OSError, subprocess.CalledProcessError, ValidationError) as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
