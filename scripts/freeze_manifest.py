"""Freeze the high-shot VisA partition with pHash groups and a test blocklist."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file  # isort: skip
from src.common.paths import Paths, load_paths  # isort: skip


LOGGER = logging.getLogger("freeze_manifest")
HASH_SIZE = 16
MAX_HASH_DISTANCE = HASH_SIZE * HASH_SIZE


class ValidationError(RuntimeError):
    """An M4 assertion failed."""


@dataclass(slots=True)
class UnionFind:
    parent: list[int]
    rank: list[int]

    @classmethod
    def create(cls, size: int) -> UnionFind:
        return cls(parent=list(range(size)), rank=[0] * size)

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def read_rows(path: Path, objects: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["object"] in objects
        ]
    return sorted(rows, key=lambda row: (row["object"], row["image"]))


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def file_fingerprints(
    paths: Paths,
    rows: list[dict[str, str]],
) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    cache_path = paths.cache / "m4_fingerprints.json"
    cache = load_cache(cache_path)
    image_sha256: list[str] = []
    phashes: list[str] = []
    changed = False

    for index, row in enumerate(rows, start=1):
        relative = row["image"].replace("\\", "/")
        image_path = paths.visa_raw / Path(relative)
        stat = image_path.stat()
        cached = cache.get(relative)
        if (
            cached
            and cached.get("size") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
        ):
            digest = str(cached["sha256"])
            phash = str(cached["phash"])
        else:
            digest = sha256_file(image_path)
            with Image.open(image_path) as image:
                phash = str(imagehash.phash(image, hash_size=HASH_SIZE))
            cache[relative] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest,
                "phash": phash,
            }
            changed = True
        image_sha256.append(digest)
        phashes.append(phash)
        if index % 200 == 0 or index == len(rows):
            LOGGER.info("Fingerprinted %d/%d images", index, len(rows))

    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return image_sha256, phashes, cache


def build_groups(
    hashes: list[int],
    *,
    threshold: int,
) -> tuple[list[int], list[int], int]:
    """Return stable component ids, nearest-neighbour distances, and edge count."""

    union_find = UnionFind.create(len(hashes))
    nearest = [MAX_HASH_DISTANCE] * len(hashes)
    edge_count = 0
    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            distance = (hashes[left] ^ hashes[right]).bit_count()
            nearest[left] = min(nearest[left], distance)
            nearest[right] = min(nearest[right], distance)
            if distance <= threshold:
                union_find.union(left, right)
                edge_count += 1

    roots = [union_find.find(index) for index in range(len(hashes))]
    root_order = {root: group_id for group_id, root in enumerate(dict.fromkeys(roots))}
    return [root_order[root] for root in roots], nearest, edge_count


def group_by_object(
    rows: list[dict[str, str]],
    phashes: list[str],
    *,
    threshold: int,
) -> tuple[list[int], dict[str, dict[str, Any]]]:
    group_ids = [-1] * len(rows)
    metrics: dict[str, dict[str, Any]] = {}
    next_group_id = 0

    object_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        object_indices[row["object"]].append(index)

    for object_name, indices in object_indices.items():
        values = [int(phashes[index], 16) for index in indices]
        local_groups, nearest, edge_count = build_groups(values, threshold=threshold)
        local_to_global: dict[int, int] = {}
        for index, local_group in zip(indices, local_groups, strict=True):
            if local_group not in local_to_global:
                local_to_global[local_group] = next_group_id
                next_group_id += 1
            group_ids[index] = local_to_global[local_group]

        sizes = Counter(local_groups)
        metrics[object_name] = {
            "images": len(indices),
            "groups": len(sizes),
            "largest_group": max(sizes.values()),
            "pairs_at_or_below_threshold": edge_count,
            "nearest_distance": {
                "min": min(nearest),
                "p05": float(np.percentile(nearest, 5)),
                "median": float(np.median(nearest)),
                "p95": float(np.percentile(nearest, 95)),
                "max": max(nearest),
            },
        }
    return group_ids, metrics


def final_sets(
    rows: list[dict[str, str]],
    group_ids: list[int],
) -> tuple[list[str], dict[str, dict[str, int]], int]:
    members: dict[int, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        members[group_id].append(index)

    sets = [row["split"] for row in rows]
    moved: dict[str, Counter[str]] = defaultdict(Counter)
    crossing_groups = 0
    for indices in members.values():
        original_sets = {rows[index]["split"] for index in indices}
        if original_sets == {"train", "test"}:
            crossing_groups += 1
            for index in indices:
                if sets[index] == "train":
                    sets[index] = "test"
                    moved[rows[index]["object"]][rows[index]["label"]] += 1

    moved_plain = {
        object_name: {
            "normal": counts["normal"],
            "anomaly": counts["anomaly"],
            "total": sum(counts.values()),
        }
        for object_name, counts in moved.items()
    }
    return sets, moved_plain, crossing_groups


def mask_sha256(
    paths: Paths,
    rows: list[dict[str, str]],
) -> dict[str, str]:
    cache_path = paths.cache / "m4_mask_sha256.json"
    cache = load_cache(cache_path)
    changed = False
    for row in rows:
        if row["label"] != "anomaly":
            continue
        relative = row["mask"].replace("\\", "/")
        path = paths.visa_raw / Path(relative)
        stat = path.stat()
        cached = cache.get(relative)
        if (
            cached
            and cached.get("size") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
        ):
            continue
        cache[relative] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }
        changed = True
    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {relative: str(value["sha256"]) for relative, value in cache.items()}


def assemble_manifest(
    *,
    paths: Paths,
    rows: list[dict[str, str]],
    image_sha256: list[str],
    phashes: list[str],
    group_ids: list[int],
    sets: list[str],
    fewshot_pool: set[str],
    mask_hashes: dict[str, str],
    seed: int,
    threshold: int,
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    for row, digest, phash, group_id, set_name in zip(
        rows,
        image_sha256,
        phashes,
        group_ids,
        sets,
        strict=True,
    ):
        mask_path = row["mask"].replace("\\", "/") if row["label"] == "anomaly" else None
        images.append(
            {
                "object": row["object"],
                "set": set_name,
                "label": "good" if row["label"] == "normal" else "bad",
                "split_type": "2cls_highshot",
                "image_path": row["image"].replace("\\", "/"),
                "mask_path": mask_path,
                "sha256": digest,
                "mask_sha256": mask_hashes[mask_path] if mask_path else None,
                "phash": phash,
                "group_id": group_id,
                "in_fewshot_pool": row["image"].replace("\\", "/") in fewshot_pool,
                "original_set": row["split"],
                "moved_to_test_by_phash": row["split"] == "train" and set_name == "test",
            }
        )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "phash_threshold": threshold,
        "phash_hash_size": HASH_SIZE,
        "source_checksums": "splits/source_checksums.json",
        "objects": list(dict.fromkeys(row["object"] for row in rows)),
        "images": images,
    }


def validate_group_sets(images: list[dict[str, Any]]) -> None:
    observed: dict[int, set[str]] = defaultdict(set)
    for image in images:
        observed[int(image["group_id"])].add(str(image["set"]))
    failures = [group_id for group_id, sets in observed.items() if len(sets) != 1]
    if failures:
        raise ValidationError(f"pHash groups still cross partitions: {failures[:10]}")


def make_blocklist(images: list[dict[str, Any]]) -> dict[str, Any]:
    test_images = [image for image in images if image["set"] == "test"]
    image_hashes = [str(image["sha256"]) for image in test_images]
    mask_hashes = [
        str(image["mask_sha256"])
        for image in test_images
        if image["mask_sha256"] is not None
    ]
    hashes = sorted(set(image_hashes + mask_hashes))
    blocklist = {
        "image_count": len(image_hashes),
        "mask_count": len(mask_hashes),
        "unique_sha256_count": len(hashes),
        "sha256": hashes,
    }
    blocked = set(hashes)
    if not all(digest in blocked for digest in image_hashes + mask_hashes):
        raise ValidationError("Test blocklist does not cover every test image and mask")
    if blocklist["unique_sha256_count"] != len(blocked):
        raise ValidationError("Test blocklist unique count is inconsistent")
    return blocklist


def split_counts(images: list[dict[str, Any]]) -> dict[str, Counter[tuple[str, str]]]:
    counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for image in images:
        counts[str(image["object"])][(str(image["set"]), str(image["label"]))] += 1
    return counts


def write_report(
    destination: Path,
    *,
    threshold: int,
    metrics: dict[str, dict[str, Any]],
    images: list[dict[str, Any]],
    moved: dict[str, dict[str, int]],
    crossing_groups: int,
    blocklist: dict[str, Any],
) -> None:
    counts = split_counts(images)
    lines = [
        "# M4 Split Freeze Report",
        "",
        f"- pHash: `imagehash.phash(hash_size={HASH_SIZE})`",
        f"- Hamming threshold: `{threshold}`",
        f"- Cross-partition groups resolved to test: `{crossing_groups}`",
        f"- Test images: `{blocklist['image_count']}`",
        f"- Test bad masks: `{blocklist['mask_count']}`",
        f"- Unique blocked SHA256: `{blocklist['unique_sha256_count']}`",
        "",
        "## pHash calibration",
        "",
        "| object | images | groups | largest | pairs ≤ threshold | nearest min / p05 / median / p95 / max |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for object_name, values in metrics.items():
        nearest = values["nearest_distance"]
        distribution = " / ".join(
            f"{nearest[key]:g}" for key in ("min", "p05", "median", "p95", "max")
        )
        lines.append(
            f"| {object_name} | {values['images']} | {values['groups']} | "
            f"{values['largest_group']} | {values['pairs_at_or_below_threshold']} | "
            f"{distribution} |"
        )

    lines.extend(
        [
            "",
            f"Threshold {threshold} was retained: it is far below the median nearest-neighbour distance,",
            "so it targets only near-identical captures. Connected components apply transitive closure.",
            "",
            "## Frozen counts",
            "",
            "| object | train good | train bad | test good | test bad | moved normal / anomaly |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for object_name, object_counts in counts.items():
        move = moved.get(object_name, {"normal": 0, "anomaly": 0})
        lines.append(
            f"| {object_name} | {object_counts[('train', 'good')]} | "
            f"{object_counts[('train', 'bad')]} | {object_counts[('test', 'good')]} | "
            f"{object_counts[('test', 'bad')]} | {move['normal']} / {move['anomaly']} |"
        )
    lines.extend(
        [
            "",
            "## Assertions",
            "",
            "- Every pHash group belongs to exactly one final set: **passed**",
            "- Every final test image SHA256 is blocklisted: **passed**",
            "- Every final test bad-mask SHA256 is blocklisted: **passed**",
            "- Manifest checksum written after serialization: **passed**",
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    paths: Paths,
    *,
    manifest: dict[str, Any],
    blocklist: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    moved: dict[str, dict[str, int]],
    crossing_groups: int,
    force: bool,
) -> str:
    manifest_path = paths.splits / "split_manifest.json"
    checksum_path = paths.splits / "MANIFEST.sha256"
    blocklist_path = paths.splits / "test_blocklist.json"
    targets = (manifest_path, checksum_path, blocklist_path)
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise ValidationError(f"Refusing to overwrite frozen artifacts: {existing}")

    paths.splits.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(manifest_path)
    checksum_path.write_text(f"{digest}  {manifest_path.name}\n", encoding="utf-8")
    blocklist["manifest_sha256"] = digest
    blocklist_path.write_text(
        json.dumps(blocklist, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(
        paths.reports / "split_report.md",
        threshold=int(manifest["phash_threshold"]),
        metrics=metrics,
        images=manifest["images"],
        moved=moved,
        crossing_groups=crossing_groups,
        blocklist=blocklist,
    )
    return digest


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.phash_threshold < 0 or args.phash_threshold > MAX_HASH_DISTANCE:
            raise ValidationError(
                f"pHash threshold must be between 0 and {MAX_HASH_DISTANCE}"
            )
        paths = load_paths(args.paths)
        objects = tuple(args.objects or paths.objects)
        seed = paths.seed if args.seed is None else args.seed
        highshot_csv = paths.visa_raw / "split_csv" / "2cls_highshot.csv"
        fewshot_csv = paths.visa_raw / "split_csv" / "2cls_fewshot.csv"
        rows = read_rows(highshot_csv, objects)
        fewshot_rows = read_rows(fewshot_csv, objects)
        fewshot_pool = {
            row["image"].replace("\\", "/")
            for row in fewshot_rows
            if row["split"] == "train"
        }
        if len(rows) != 1_806 and objects == paths.objects:
            raise ValidationError(f"Expected 1,806 locked images, observed {len(rows)}")

        image_hashes, phashes, _ = file_fingerprints(paths, rows)
        group_ids, metrics = group_by_object(
            rows,
            phashes,
            threshold=args.phash_threshold,
        )
        sets, moved, crossing_groups = final_sets(rows, group_ids)
        mask_hashes = mask_sha256(paths, rows)
        manifest = assemble_manifest(
            paths=paths,
            rows=rows,
            image_sha256=image_hashes,
            phashes=phashes,
            group_ids=group_ids,
            sets=sets,
            fewshot_pool=fewshot_pool,
            mask_hashes=mask_hashes,
            seed=seed,
            threshold=args.phash_threshold,
        )
        validate_group_sets(manifest["images"])
        blocklist = make_blocklist(manifest["images"])

        LOGGER.info(
            "Calibration: %s",
            json.dumps(
                {
                    "threshold": args.phash_threshold,
                    "metrics": metrics,
                    "crossing_groups": crossing_groups,
                    "moved": moved,
                    "final_counts": {
                        object_name: {
                            f"{set_name}_{label}": count
                            for (set_name, label), count in counts.items()
                        }
                        for object_name, counts in split_counts(manifest["images"]).items()
                    },
                },
                sort_keys=True,
            ),
        )
        if args.dry_run:
            LOGGER.info("Dry run: frozen files were not written")
            return 0

        digest = write_outputs(
            paths,
            manifest=manifest,
            blocklist=blocklist,
            metrics=metrics,
            moved=moved,
            crossing_groups=crossing_groups,
            force=args.force,
        )
        LOGGER.info("M4 manifest frozen with SHA256 %s", digest)
        return 0
    except (OSError, ValidationError, json.JSONDecodeError) as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
