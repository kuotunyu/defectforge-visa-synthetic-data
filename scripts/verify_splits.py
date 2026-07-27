"""Independently re-run the frozen ADR-007 split and blocklist assertions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import load_paths

OBJECTS = ("pcb1", "capsules")
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


class SplitVerificationError(RuntimeError):
    """Raised when frozen split evidence violates ADR-007."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitVerificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mapping(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing official split CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    require(
        set(reader.fieldnames or ()) == {"object", "split", "label", "image", "mask"},
        f"Unexpected split CSV schema: {path}",
    )
    require(bool(rows), f"Empty split CSV: {path}")
    return rows


def normalized_image_sets(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, set[str]]]:
    sets = {
        object_name: {"train": set(), "test": set()}
        for object_name in OBJECTS
    }
    for row in rows:
        object_name = row["object"]
        if object_name not in sets:
            continue
        set_name = row["split"]
        require(set_name in {"train", "test"}, f"Unexpected split name: {set_name}")
        path = row["image"].replace("\\", "/")
        require(path not in sets[object_name][set_name], f"Duplicate split row: {path}")
        sets[object_name][set_name].add(path)
    return sets


def validate_official_splits(
    rows_by_split: Mapping[str, Sequence[Mapping[str, str]]],
) -> dict[str, Any]:
    require(
        set(rows_by_split) == set(EXPECTED_COUNTS),
        "Both official fewshot and highshot CSVs are required",
    )
    counts: dict[str, dict[str, dict[str, int]]] = {}
    sets: dict[str, dict[str, dict[str, set[str]]]] = {}
    for split_type, expected_by_object in EXPECTED_COUNTS.items():
        rows = rows_by_split[split_type]
        sets[split_type] = normalized_image_sets(rows)
        counts[split_type] = {}
        for object_name in OBJECTS:
            observed = Counter(
                (row["split"], row["label"])
                for row in rows
                if row["object"] == object_name
            )
            require(
                observed == expected_by_object[object_name],
                f"ADR-007 count mismatch: {split_type}/{object_name}",
            )
            require(
                not (sets[split_type][object_name]["train"] & sets[split_type][object_name]["test"]),
                f"Official train/test overlap: {split_type}/{object_name}",
            )
            counts[split_type][object_name] = {
                f"{set_name}_{label}": observed[(set_name, label)]
                for set_name in ("train", "test")
                for label in ("normal", "anomaly")
            }
    for object_name in OBJECTS:
        fewshot_train = sets["2cls_fewshot"][object_name]["train"]
        highshot_train = sets["2cls_highshot"][object_name]["train"]
        require(
            fewshot_train < highshot_train,
            f"fewshot train is not a strict highshot train subset: {object_name}",
        )
    return {
        "counts": counts,
        "official_train_test_disjoint": True,
        "fewshot_train_strict_subset_highshot_train": True,
    }


def validate_manifest(
    *,
    manifest: Mapping[str, Any],
    blocklist: Mapping[str, Any],
    manifest_sha256: str,
    highshot_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    images = manifest.get("images")
    require(isinstance(images, list) and bool(images), "Manifest images are missing")
    expected_paths = {
        row["image"].replace("\\", "/")
        for row in highshot_rows
        if row["object"] in OBJECTS
    }
    manifest_paths = [str(row["image_path"]) for row in images]
    require(len(manifest_paths) == len(set(manifest_paths)), "Manifest image paths are duplicated")
    require(set(manifest_paths) == expected_paths, "Manifest is not the highshot CSV universe")
    require(
        all(row["split_type"] == "2cls_highshot" for row in images),
        "Manifest contains a non-highshot source",
    )

    group_sets: dict[tuple[str, int], set[str]] = defaultdict(set)
    train_hashes: set[str] = set()
    test_hashes: set[str] = set()
    blocked_expected: set[str] = set()
    test_images = 0
    test_masks = 0
    for row in images:
        group_sets[(str(row["object"]), int(row["group_id"]))].add(str(row["set"]))
        digest = str(row["sha256"])
        if row["set"] == "train":
            train_hashes.add(digest)
        elif row["set"] == "test":
            test_hashes.add(digest)
            blocked_expected.add(digest)
            test_images += 1
            if row.get("mask_sha256") is not None:
                blocked_expected.add(str(row["mask_sha256"]))
                test_masks += 1
        else:
            raise SplitVerificationError(f"Unexpected manifest set: {row['set']}")
    require(
        all(len(values) == 1 for values in group_sets.values()),
        "A pHash group crosses the frozen train/test boundary",
    )
    require(not (train_hashes & test_hashes), "Manifest train/test SHA256 overlap")

    blocked = blocklist.get("sha256")
    require(isinstance(blocked, list), "Blocklist SHA256 list is missing")
    require(len(blocked) == len(set(blocked)), "Blocklist contains duplicate SHA256")
    require(set(blocked) == blocked_expected, "Blocklist does not exactly cover frozen test")
    require(blocklist.get("manifest_sha256") == manifest_sha256, "Blocklist manifest hash changed")
    require(blocklist.get("image_count") == test_images, "Blocklist test image count changed")
    require(blocklist.get("mask_count") == test_masks, "Blocklist test mask count changed")
    require(
        blocklist.get("unique_sha256_count") == len(blocked_expected),
        "Blocklist unique count changed",
    )
    return {
        "manifest_images": len(images),
        "phash_groups": len(group_sets),
        "train_image_sha256": len(train_hashes),
        "test_image_sha256": len(test_hashes),
        "blocked_sha256": len(blocked_expected),
        "test_images": test_images,
        "test_masks": test_masks,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/split_verification.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_paths(args.paths)
    rows_by_split = {
        split_type: read_rows(paths.visa_raw / "split_csv" / f"{split_type}.csv")
        for split_type in EXPECTED_COUNTS
    }
    official = validate_official_splits(rows_by_split)

    manifest_path = paths.splits / "split_manifest.json"
    checksum_path = paths.splits / "MANIFEST.sha256"
    blocklist_path = paths.splits / "test_blocklist.json"
    manifest_sha256 = sha256_file(manifest_path)
    checksum_parts = checksum_path.read_text(encoding="utf-8").strip().split()
    require(
        checksum_parts == [manifest_sha256, "split_manifest.json"],
        "MANIFEST.sha256 does not match split_manifest.json",
    )
    manifest_result = validate_manifest(
        manifest=load_mapping(manifest_path),
        blocklist=load_mapping(blocklist_path),
        manifest_sha256=manifest_sha256,
        highshot_rows=rows_by_split["2cls_highshot"],
    )
    payload = {
        "status": "passed",
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "assertions": {
            "adr007_eight_counts": "passed",
            "official_train_test_disjoint": "passed",
            "fewshot_train_strict_subset_highshot_train": "passed",
            "manifest_phash_and_blocklist_integrity": "passed",
        },
        "official": official,
        "manifest": manifest_result,
    }
    atomic_write_json(args.output, payload)
    print(f"Verified frozen splits: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
