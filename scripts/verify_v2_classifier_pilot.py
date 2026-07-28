"""Independently verify M26 result aggregation against raw development runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_v2_classifier_pilot import evaluate_gate, load_pilot_config
from src.common.integrity import sha256_file
from src.common.paths import load_paths


class PilotVerificationError(RuntimeError):
    """Raised when tracked M26 evidence differs from raw run artifacts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotVerificationError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=Path("configs/classifier_v2_pilot.yaml"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("results/v2/pilot_classification.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v2_pilot_validation.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_paths(args.paths)
    pilot_path = args.pilot_config.resolve(strict=True)
    result_path = args.result.resolve(strict=True)
    config = load_pilot_config(pilot_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(result.get("schema_version") == 1, "Unsupported result schema")
    require(result.get("status") == "passed", "Pilot aggregation did not pass")
    require(result.get("test_data_loaded") is False, "Pilot claims test data was loaded")
    require(
        result.get("pilot_config_sha256") == sha256_file(pilot_path),
        "Pilot config changed after execution",
    )
    records = result.get("runs")
    require(isinstance(records, list) and len(records) == 8, "Pilot must contain eight runs")
    by_run_name = {str(record["run_name"]): record for record in records}
    require(len(by_run_name) == len(records), "Duplicate pilot result record")
    reports: dict[str, dict[str, Mapping[str, Any]]] = {}
    raw_runs: list[dict[str, str]] = []
    run_root = paths.runs / str(config["output"]["run_subdirectory"])
    for candidate in config["candidates"]:
        candidate_name = str(candidate["name"])
        for object_name in config["objects"]:
            run_name = f"m26_{candidate_name}_{object_name}_seed_{config['seed']}_dev"
            record = by_run_name.get(run_name)
            require(record is not None, f"Missing pilot result: {run_name}")
            output_dir = run_root / str(record["output_directory_name"])
            require(output_dir.name == run_name, f"Unsafe run directory name: {run_name}")
            report_path = output_dir / "training_report.json"
            manifest_path = output_dir / "data_manifest.json"
            require(report_path.is_file() and manifest_path.is_file(), f"Raw run missing: {run_name}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            require(report.get("status") == "passed", f"Raw run failed: {run_name}")
            require(report.get("mode") == "development", f"Non-development run: {run_name}")
            require(manifest.get("mode") == "development", f"Manifest mode mismatch: {run_name}")
            require(manifest.get("test") == [], f"Test records loaded: {run_name}")
            validation = manifest.get("validation")
            require(isinstance(validation, list) and validation, f"Validation missing: {run_name}")
            require(
                all(item.get("kind") == "real" for item in validation),
                f"Synthetic validation record: {run_name}",
            )
            require(
                report.get("run_signature") == record["run_signature"],
                f"Run signature changed: {run_name}",
            )
            require(report.get("model_sha256") == record["model_sha256"], f"Model changed: {run_name}")
            require(report.get("metrics") == record["metrics"], f"Metrics changed: {run_name}")
            require(
                report.get("sample_exposure") == record["sample_exposure"],
                f"Exposure changed: {run_name}",
            )
            reports.setdefault(candidate_name, {})[str(object_name)] = report
            raw_runs.append(
                {
                    "run_name": run_name,
                    "training_report_sha256": sha256_file(report_path),
                    "data_manifest_sha256": sha256_file(manifest_path),
                }
            )
    observed_gate = evaluate_gate(reports, config)
    require(observed_gate == result.get("gate"), "Pilot gate aggregation changed")
    require(
        observed_gate["confirmatory_run_authorized_by_gate"] is False,
        "Verifier unexpectedly authorized M27",
    )
    validation = {
        "schema_version": 1,
        "status": "passed",
        "pilot_config_sha256": sha256_file(pilot_path),
        "result_sha256": sha256_file(result_path),
        "run_count": len(records),
        "test_data_loaded": False,
        "gate_status": observed_gate["status"],
        "m27_authorized": False,
        "raw_runs": sorted(raw_runs, key=lambda record: record["run_name"]),
    }
    output = args.output.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
