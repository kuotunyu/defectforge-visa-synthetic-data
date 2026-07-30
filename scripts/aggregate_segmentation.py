"""Rebuild M20 segmentation.csv and its report only from raw M18 run artefacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_segmenter_runs import FORMAL_GROUPS, validate
from src.common.integrity import write_text_lf
from src.common.paths import load_paths

LOGICAL_GROUPS = (*FORMAL_GROUPS, "all_mixed")
# ADR-032 pre-registered a three-seed replication of all eight formal groups. Seed 42 is
# the pre-registered anchor; 43 and 44 give every group error bars, so no group could be
# chosen for replication after its result was known.
SEEDS = (42, 43, 44)
ANCHOR_SEED = 42
CSV_COLUMNS = (
    "logical_group",
    "canonical_group",
    "physical_run",
    "alias_of",
    "object",
    "seed",
    "run_name",
    "run_signature",
    "model",
    "model_revision",
    "input_size",
    "total_steps",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "train_total",
    "train_real",
    "train_synthetic",
    "train_real_defect",
    "train_synthetic_defect",
    "test_total",
    "dice",
    "miou",
    "pixel_auroc",
    "aupro",
    "peak_vram_gib",
    "training_seconds",
    "model_sha256",
    "data_manifest_sha256",
)
METRICS = ("dice", "miou", "pixel_auroc", "aupro")
MAX_RESULT_ARCHIVE_MEMBERS = 256
MAX_RESULT_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024**3


class SegmentationAggregationError(RuntimeError):
    """Raised when raw M18 artefacts cannot produce an exact M20 table."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SegmentationAggregationError(message)


def load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON mapping: {path}")
    return value


def _run_names(object_name: str, seeds: Sequence[int]) -> tuple[str, ...]:
    return tuple(
        f"m18_{group_name}_{object_name}_seed{seed}"
        for seed in seeds
        for group_name in FORMAL_GROUPS
    )


def _expected_archive_files(object_name: str, seeds: Sequence[int] = SEEDS) -> set[str]:
    expected = {
        f"m18_{object_name}_validation.json",
        f"segmentation_{object_name}.csv",
        "timings.json",
    }
    for run_name in _run_names(object_name, seeds):
        run = f"runs/{run_name}"
        expected.update(
            {
                f"{run}/training_report.json",
                f"{run}/run_config.json",
                f"{run}/data_manifest.json",
                f"{run}/final/config.json",
                f"{run}/final/model.safetensors",
            }
        )
    return expected


def _normalized_archive_path(info: zipfile.ZipInfo) -> str:
    name = info.filename
    require("\\" not in name, f"ZIP member uses a Windows separator: {name}")
    path = PurePosixPath(name)
    require(not path.is_absolute(), f"ZIP member is absolute: {name}")
    require(".." not in path.parts, f"ZIP member escapes its root: {name}")
    require(not any(":" in part for part in path.parts), f"ZIP member uses a drive/ADS: {name}")
    normalized = path.as_posix().rstrip("/")
    require(normalized not in {"", "."}, f"ZIP member has an empty path: {name}")
    return normalized


def _inspect_result_archive(
    archive_path: Path,
    *,
    object_name: str,
    seeds: Sequence[int] = SEEDS,
) -> tuple[list[zipfile.ZipInfo], int]:
    expected = _expected_archive_files(object_name, seeds)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            require(
                0 < len(members) <= MAX_RESULT_ARCHIVE_MEMBERS,
                f"Unexpected ZIP member count: {len(members)}",
            )
            files: dict[str, zipfile.ZipInfo] = {}
            casefolded: set[str] = set()
            total_uncompressed = 0
            for info in members:
                normalized = _normalized_archive_path(info)
                folded = normalized.casefold()
                require(folded not in casefolded, f"Duplicate ZIP path: {normalized}")
                casefolded.add(folded)
                unix_type = (info.external_attr >> 16) & 0o170000
                require(unix_type != 0o120000, f"ZIP member is a symlink: {normalized}")
                if info.is_dir():
                    require(
                        any(path.startswith(f"{normalized}/") for path in expected),
                        f"Unexpected ZIP directory: {normalized}",
                    )
                    continue
                require(normalized in expected, f"Unexpected ZIP file: {normalized}")
                files[normalized] = info
                total_uncompressed += info.file_size
            require(set(files) == expected, "M18 result ZIP inventory is incomplete")
            require(
                total_uncompressed <= MAX_RESULT_ARCHIVE_UNCOMPRESSED_BYTES,
                f"M18 result ZIP is too large when expanded: {total_uncompressed}",
            )
            require(archive.testzip() is None, f"M18 result ZIP CRC failed: {archive_path.name}")
            return list(files.values()), total_uncompressed
    except zipfile.BadZipFile as error:
        raise SegmentationAggregationError(
            f"Invalid M18 result ZIP: {archive_path}"
        ) from error


def _verify_returned_summary(
    root: Path,
    *,
    object_name: str,
    seeds: Sequence[int] = SEEDS,
) -> None:
    run_names = _run_names(object_name, seeds)
    validation = load_mapping(root / f"m18_{object_name}_validation.json")
    require(validation.get("status") == "passed", "Returned Colab validator did not pass")
    require(validation.get("object") == object_name, "Returned Colab object changed")
    require(validation.get("runs") == len(run_names), "Returned Colab run count changed")
    require(
        [int(seed) for seed in validation.get("seeds", [])] == list(seeds),
        "Returned Colab seed list changed",
    )
    require(validation.get("alias_reruns") == 0, "all_mixed was physically rerun")
    require(
        validation.get("all_mixed_alias_of") == "filtered_syn",
        "Returned all_mixed alias changed",
    )
    require(validation.get("fresh_model_reload") is True, "Colab skipped fresh model reload")
    timings = load_mapping(root / "timings.json")
    require(set(timings) == set(run_names), "Returned timings do not cover every seeded run")
    for run_name, value in timings.items():
        seconds = float(value)
        require(
            math.isfinite(seconds) and seconds > 0.0,
            f"Invalid returned timing: {run_name}",
        )


def import_result_archive(
    results_root: Path,
    *,
    object_name: str,
    seeds: Sequence[int] = SEEDS,
) -> Path:
    """Atomically and idempotently import one exact Colab result ZIP."""
    archive_path = results_root / f"m18_seg_results_{object_name}.zip"
    require(archive_path.is_file(), f"Missing M18 result ZIP: {archive_path}")
    archive_sha256 = sha256_file(archive_path)
    members, uncompressed_bytes = _inspect_result_archive(
        archive_path,
        object_name=object_name,
        seeds=seeds,
    )
    destination = results_root / object_name
    import_manifest_name = "import_manifest.json"
    if destination.exists():
        manifest = load_mapping(destination / import_manifest_name)
        require(manifest.get("archive_sha256") == archive_sha256, "Imported ZIP changed")
        require(manifest.get("object") == object_name, "Imported object changed")
        _verify_returned_summary(destination, object_name=object_name, seeds=seeds)
        return destination

    results_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".importing_{object_name}_", dir=results_root)
    ).resolve()
    require(temporary.parent == results_root.resolve(), "Unsafe M18 import directory")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in members:
                normalized = _normalized_archive_path(info)
                target = temporary.joinpath(*PurePosixPath(normalized).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        _verify_returned_summary(temporary, object_name=object_name, seeds=seeds)
        (temporary / import_manifest_name).write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "object": object_name,
                    "seeds": list(seeds),
                    "archive_file": archive_path.name,
                    "archive_sha256": archive_sha256,
                    "members": len(members),
                    "uncompressed_bytes": uncompressed_bytes,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists() and temporary.parent == results_root.resolve():
            shutil.rmtree(temporary)
        raise
    return destination


def _physical_row(run_dir: Path, group_name: str, object_name: str) -> dict[str, Any]:
    report = load_mapping(run_dir / "training_report.json")
    data = load_mapping(run_dir / "data_manifest.json")
    counts = report["train_counts"]
    metrics = report["metrics"]
    return {
        "logical_group": group_name,
        "canonical_group": group_name,
        "physical_run": "true",
        "alias_of": "",
        "object": object_name,
        "seed": report["seed"],
        "run_name": report["run_name"],
        "run_signature": report["run_signature"],
        "model": report["model"],
        "model_revision": report["model_revision"],
        "input_size": report["input_size"],
        "total_steps": report["executed_steps"],
        "batch_size": report["batch_size"],
        "learning_rate": report["learning_rate"],
        "weight_decay": report["weight_decay"],
        "train_total": counts["total"],
        "train_real": counts["real"],
        "train_synthetic": counts["synthetic"],
        "train_real_defect": sum(
            row["kind"] == "real" and row["has_defect"] for row in data["train"]
        ),
        "train_synthetic_defect": sum(
            row["kind"] == "synthetic" for row in data["train"]
        ),
        "test_total": report["evaluation_counts"]["total"],
        "dice": metrics["dice"],
        "miou": metrics["miou"],
        "pixel_auroc": metrics["pixel_auroc"],
        "aupro": metrics["aupro"],
        "peak_vram_gib": report["peak_vram_gib"],
        "training_seconds": report["training_seconds"],
        "model_sha256": report["model_sha256"],
        "data_manifest_sha256": report["data_manifest_sha256"],
    }


def logical_rows(
    run_root: Path,
    object_name: str,
    seeds: Sequence[int] = SEEDS,
) -> list[dict[str, Any]]:
    """Build one row per (group, seed), plus one `all_mixed` alias row per seed."""
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        seeded = [
            _physical_row(
                run_root / f"m18_{group_name}_{object_name}_seed{seed}",
                group_name,
                object_name,
            )
            for group_name in FORMAL_GROUPS
        ]
        for row in seeded:
            require(int(row["seed"]) == int(seed), f"Report seed changed: {row['run_name']}")
        filtered = next(row for row in seeded if row["logical_group"] == "filtered_syn")
        alias = dict(filtered)
        alias.update(
            {
                "logical_group": "all_mixed",
                "physical_run": "false",
                "alias_of": "filtered_syn",
            }
        )
        rows.extend([*seeded, alias])
    return rows


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in CSV_COLUMNS})
    os.replace(temporary, path)


def _format(value: Any) -> str:
    return f"{float(value):.4f}"


def _mean_std(values: Sequence[float]) -> str:
    require(bool(values), "Cannot summarise an empty metric series")
    if len(values) == 1:
        return f"{float(values[0]):.4f}"
    return f"{float(np.mean(values)):.4f} ± {float(np.std(values, ddof=1)):.4f}"


def _select(
    rows: Sequence[Mapping[str, Any]],
    *,
    object_name: str,
    group_name: str,
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row["object"] == object_name and row["logical_group"] == group_name
    ]
    seeds = [int(row["seed"]) for row in selected]
    require(len(set(seeds)) == len(seeds), f"Duplicate seed: {object_name}/{group_name}")
    return sorted(selected, key=lambda row: int(row["seed"]))


def build_report(rows: Sequence[Mapping[str, Any]], seeds: Sequence[int] = SEEDS) -> str:
    objects = sorted({str(row["object"]) for row in rows})
    lines = [
        "# M20 segmentation results",
        "",
        (
            "All values below were rebuilt from returned raw `training_report.json` files. "
            "Notebook output text and the per-runtime CSV files were not used."
        ),
        "",
        (
            f"Seeds: {', '.join(str(seed) for seed in seeds)}. Seed {ANCHOR_SEED} is the "
            "pre-registered anchor; ADR-032 added the remaining seeds to all eight formal "
            "groups so that no group lacks error bars."
        ),
        "",
    ]
    for object_name in objects:
        lines.extend(
            [
                f"## {object_name}",
                "",
                f"### Seed {ANCHOR_SEED} (pre-registered anchor)",
                "",
                "| Group | Physical run | Dice | mIoU | pixel AUROC | AUPRO |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for group_name in LOGICAL_GROUPS:
            row = next(
                item
                for item in _select(rows, object_name=object_name, group_name=group_name)
                if int(item["seed"]) == ANCHOR_SEED
            )
            physical = "yes" if row["physical_run"] == "true" else "no; cites filtered_syn"
            lines.append(
                f"| {group_name} | {physical} | {_format(row['dice'])} | "
                f"{_format(row['miou'])} | {_format(row['pixel_auroc'])} | "
                f"{_format(row['aupro'])} |"
            )
        lines.extend(
            [
                "",
                f"### Mean ± std across {len(seeds)} seeds",
                "",
                "| Group | Seeds | Dice | mIoU | pixel AUROC | AUPRO |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for group_name in LOGICAL_GROUPS:
            selected = _select(rows, object_name=object_name, group_name=group_name)
            require(
                len(selected) == len(seeds),
                f"Expected {len(seeds)} seeds: {object_name}/{group_name}",
            )
            cells = [
                _mean_std([float(row[metric]) for row in selected]) for metric in METRICS
            ]
            lines.append(f"| {group_name} | {len(selected)} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(
        [
            "## Two-object macro mean",
            "",
            (
                "Each cell first averages the seeds within an object, then averages the two "
                "objects; the spread is the std of the per-seed macro means."
            ),
            "",
            "| Group | Dice | mIoU | pixel AUROC | AUPRO |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group_name in LOGICAL_GROUPS:
        cells = []
        for metric in METRICS:
            per_seed_macro = [
                float(
                    np.mean(
                        [
                            float(row[metric])
                            for object_name in objects
                            for row in _select(
                                rows,
                                object_name=object_name,
                                group_name=group_name,
                            )
                            if int(row["seed"]) == seed
                        ]
                    )
                )
                for seed in seeds
            ]
            require(
                len(per_seed_macro) == len(seeds),
                f"Macro mean needs every seed: {group_name}",
            )
            cells.append(_mean_std(per_seed_macro))
        lines.append(f"| {group_name} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            (
                "`procedural_only` means zero real defect **images/pixels** in training, while its "
                "procedural generator used aggregate real-mask statistics as preregistered in "
                "ADR-011."
            ),
            "",
            "`all_mixed` is the logical ninth group and cites the exact `filtered_syn` physical run.",
            "",
        ]
    )
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/segmenter.yaml"))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/colab/segmentation/3seed"),
    )
    parser.add_argument(
        "--seed",
        dest="seeds",
        type=int,
        action="append",
        help=(
            "Repeatable. Defaults to the ADR-032 replication set "
            f"({', '.join(str(seed) for seed in SEEDS)})."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("results/segmentation.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/segmentation_results.md"))
    parser.add_argument(
        "--validation-out",
        type=Path,
        default=Path("reports/segmentation_validation.json"),
    )
    args = parser.parse_args()
    paths = load_paths(args.paths)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "Invalid segmenter config")
    results_root = args.results_root.resolve(strict=True)
    seeds = tuple(dict.fromkeys(args.seeds)) if args.seeds else SEEDS
    require(ANCHOR_SEED in seeds, f"The pre-registered anchor seed {ANCHOR_SEED} is required")
    rows: list[dict[str, Any]] = []
    validations: dict[str, Any] = {}
    raw_report_hashes: dict[str, str] = {}
    for object_name in paths.objects:
        object_root = import_result_archive(
            results_root,
            object_name=object_name,
            seeds=seeds,
        )
        run_root = object_root / "runs"
        validations[object_name] = validate(
            paths=paths,
            config=config,
            run_root=run_root.resolve(strict=True),
            object_name=object_name,
            reload_model=False,
            seeds=seeds,
        )
        rows.extend(logical_rows(run_root, object_name, seeds))
        for run_name in _run_names(object_name, seeds):
            report_path = run_root / run_name / "training_report.json"
            raw_report_hashes[f"{object_name}/{run_name}"] = sha256_file(report_path)

    expected_physical = len(FORMAL_GROUPS) * len(paths.objects) * len(seeds)
    expected_logical = len(LOGICAL_GROUPS) * len(paths.objects) * len(seeds)
    require(len(rows) == expected_logical, f"M20 must contain {expected_logical} logical rows")
    require(
        sum(row["physical_run"] == "true" for row in rows) == expected_physical,
        "Physical run count changed",
    )
    atomic_write_csv(args.output, rows)
    report_text = build_report(rows, seeds)
    write_text_lf(args.report, report_text)
    validation_payload = {
        "status": "passed",
        "schema_version": 2,
        "seeds": list(seeds),
        "anchor_seed": ANCHOR_SEED,
        "physical_runs": expected_physical,
        "logical_rows": expected_logical,
        "all_mixed_alias_of": "filtered_syn",
        "source": "raw_training_report_json",
        "raw_report_sha256": raw_report_hashes,
        "object_validation": validations,
        "segmentation_csv_sha256": sha256_file(args.output),
        "report_sha256": sha256_file(args.report),
    }
    write_text_lf(
        args.validation_out,
        json.dumps(validation_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(validation_payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
