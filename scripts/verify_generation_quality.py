"""Verify M14 CSV, Markdown, validation, figures, and cache fingerprints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file
from src.common.paths import load_paths
from src.evaluation.quality_reporting import (
    CSV_FIELDS,
    embedded_summary,
    summary_sha256,
)
from src.filtering.pipeline import load_yaml


def _read_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise RuntimeError(f"Generation quality CSV fields changed: {reader.fieldnames}")
        for row in reader:
            normalized: dict[str, Any] = dict(row)
            for field in ("n_real", "n_generated"):
                normalized[field] = int(row[field])
            for field in (
                "nn_mean",
                "nn_median",
                "nn_p05",
                "nn_p95",
                "mnn_score",
                "kid",
                "fid",
            ):
                normalized[field] = None if row[field] == "" else float(row[field])
            rows.append(normalized)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/quality.yaml"))
    args = parser.parse_args()
    paths = load_paths(args.paths)
    config = load_yaml(args.config)
    output = config["output"]
    report_path = paths.project_root / str(output["report"])
    validation_path = paths.project_root / str(output["validation"])
    csv_path = paths.project_root / str(output["csv"])
    with validation_path.open("r", encoding="utf-8") as handle:
        validation = json.load(handle)
    embedded = embedded_summary(report_path.read_text(encoding="utf-8"))
    if validation["status"] != "passed" or not validation["sanity_passed"]:
        raise RuntimeError("M14 mandatory sanity gate did not pass")
    if validation["summary_sha256"] != summary_sha256(embedded):
        raise RuntimeError("M14 validation/report digest differs")
    for key in embedded:
        if key not in {"schema_version", "pipeline_version"} and validation[key] != embedded[key]:
            raise RuntimeError(f"M14 validation/report field differs: {key}")
    if _read_csv(csv_path) != embedded["rows"]:
        raise RuntimeError("M14 CSV rows differ from the report")
    for cache in embedded["caches"].values():
        path = Path(cache["path"])
        if not path.is_file() or sha256_file(path) != cache["sha256"]:
            raise RuntimeError(f"M14 feature cache changed: {path}")
    for field in ("nn_figure", "comparison_figure"):
        figure = paths.project_root / str(output[field])
        if not figure.is_file() or figure.stat().st_size == 0:
            raise RuntimeError(f"M14 figure missing: {figure}")
    result = {
        "status": "passed",
        "rows": len(embedded["rows"]),
        "sanity_checks": len(embedded["sanity"]),
        "blocklist_hits": embedded["source_audit"]["blocklist_hits"],
        "summary_sha256": summary_sha256(embedded),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
