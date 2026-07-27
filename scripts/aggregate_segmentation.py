"""Rebuild M20 segmentation.csv and its report only from raw M18 run artefacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_segmenter_runs import FORMAL_GROUPS, validate
from src.common.paths import load_paths

LOGICAL_GROUPS = (*FORMAL_GROUPS, "all_mixed")
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


class SegmentationAggregationError(RuntimeError):
    """Raised when raw M18 artefacts cannot produce an exact M20 table."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SegmentationAggregationError(message)


def load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON mapping: {path}")
    return value


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


def logical_rows(run_root: Path, object_name: str) -> list[dict[str, Any]]:
    rows = [
        _physical_row(
            run_root / f"m18_{group_name}_{object_name}_seed42",
            group_name,
            object_name,
        )
        for group_name in FORMAL_GROUPS
    ]
    filtered = next(row for row in rows if row["logical_group"] == "filtered_syn")
    alias = dict(filtered)
    alias.update(
        {
            "logical_group": "all_mixed",
            "physical_run": "false",
            "alias_of": "filtered_syn",
        }
    )
    rows.append(alias)
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


def build_report(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# M20 segmentation results",
        "",
        (
            "All values below were rebuilt from returned raw `training_report.json` files. "
            "Notebook output text and the per-runtime CSV files were not used."
        ),
        "",
    ]
    for object_name in sorted({str(row["object"]) for row in rows}):
        lines.extend(
            [
                f"## {object_name}",
                "",
                "| Group | Physical run | Dice | mIoU | pixel AUROC | AUPRO |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if row["object"] != object_name:
                continue
            physical = "yes" if row["physical_run"] == "true" else "no; cites filtered_syn"
            lines.append(
                f"| {row['logical_group']} | {physical} | {_format(row['dice'])} | "
                f"{_format(row['miou'])} | {_format(row['pixel_auroc'])} | "
                f"{_format(row['aupro'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Two-object macro mean",
            "",
            "| Group | Dice | mIoU | pixel AUROC | AUPRO |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group_name in LOGICAL_GROUPS:
        group_rows = [row for row in rows if row["logical_group"] == group_name]
        require(len(group_rows) == 2, f"Macro mean needs two objects: {group_name}")
        means = {
            metric: float(np.mean([float(row[metric]) for row in group_rows]))
            for metric in METRICS
        }
        lines.append(
            f"| {group_name} | {_format(means['dice'])} | {_format(means['miou'])} | "
            f"{_format(means['pixel_auroc'])} | {_format(means['aupro'])} |"
        )
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
        default=Path("results/colab/segmentation"),
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
    rows: list[dict[str, Any]] = []
    validations: dict[str, Any] = {}
    raw_report_hashes: dict[str, str] = {}
    for object_name in paths.objects:
        object_root = results_root / object_name
        run_root = object_root / "runs"
        validations[object_name] = validate(
            paths=paths,
            config=config,
            run_root=run_root.resolve(strict=True),
            object_name=object_name,
            reload_model=False,
        )
        rows.extend(logical_rows(run_root, object_name))
        for group_name in FORMAL_GROUPS:
            report_path = (
                run_root
                / f"m18_{group_name}_{object_name}_seed42"
                / "training_report.json"
            )
            raw_report_hashes[f"{object_name}/{group_name}"] = sha256_file(report_path)

    require(len(rows) == 18, "M20 must contain 18 logical rows")
    require(sum(row["physical_run"] == "true" for row in rows) == 16, "Physical run count changed")
    atomic_write_csv(args.output, rows)
    report_text = build_report(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text, encoding="utf-8")
    validation_payload = {
        "status": "passed",
        "schema_version": 1,
        "physical_runs": 16,
        "logical_rows": 18,
        "all_mixed_alias_of": "filtered_syn",
        "source": "raw_training_report_json",
        "raw_report_sha256": raw_report_hashes,
        "object_validation": validations,
        "segmentation_csv_sha256": sha256_file(args.output),
        "report_sha256": sha256_file(args.report),
    }
    args.validation_out.parent.mkdir(parents=True, exist_ok=True)
    args.validation_out.write_text(
        json.dumps(validation_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation_payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
