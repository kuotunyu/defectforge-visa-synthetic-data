"""Verify both M16 real-model smoke runs without loading a model or configuring CUDA."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file
from src.common.paths import load_paths
from src.training.train_classifier import canonical_json_sha256, load_config

RUN_NAMES = {
    "pcb1": "m16_smoke_pcb1_v1",
    "capsules": "m16_smoke_capsules_v1",
}


class ClassifierSmokeValidationError(RuntimeError):
    """Raised when local M16 smoke evidence is incomplete or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClassifierSmokeValidationError(message)


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClassifierSmokeValidationError(f"Invalid JSON: {path}") from exc
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def verify_run(
    run_root: Path,
    *,
    object_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    report = load_json_object(run_root / "training_report.json")
    data_manifest = load_json_object(run_root / "data_manifest.json")
    run_config = load_json_object(run_root / "run_config.json")
    model_path = run_root / "model.safetensors"
    require(model_path.is_file(), f"{object_name}: model is missing")
    require(report.get("status") == "passed", f"{object_name}: status did not pass")
    require(report.get("smoke") is True, f"{object_name}: not a smoke run")
    require(report.get("mode") == "development", f"{object_name}: wrong mode")
    require(report.get("object") == object_name, f"{object_name}: object mismatch")
    require(
        report.get("requested_group") == report.get("canonical_group") == "real_only",
        f"{object_name}: smoke group mismatch",
    )
    require(
        report.get("requested_total_steps") == report.get("executed_steps") == 1,
        f"{object_name}: smoke did not execute exactly one step",
    )
    require(
        report.get("base_weight_sha256") == config["model"]["sha256"],
        f"{object_name}: base weight lock mismatch",
    )
    require(
        report.get("model_revision") == config["model"]["revision"],
        f"{object_name}: model revision mismatch",
    )
    require(
        report.get("model_sha256") == sha256_file(model_path),
        f"{object_name}: trained model hash mismatch",
    )
    require(float(report.get("peak_vram_gib", 0.0)) > 0.0, f"{object_name}: VRAM missing")

    stored_data_sha = str(data_manifest.pop("sha256"))
    observed_data_sha = canonical_json_sha256(data_manifest)
    require(stored_data_sha == observed_data_sha, f"{object_name}: data manifest hash mismatch")
    require(
        report.get("data_manifest_sha256")
        == run_config.get("data_manifest_sha256")
        == stored_data_sha,
        f"{object_name}: data manifest links mismatch",
    )
    counts = data_manifest["counts"]
    require(counts["test"]["total"] == 0, f"{object_name}: smoke loaded test data")
    require(counts["validation"]["total"] > 0, f"{object_name}: validation is empty")
    require(
        counts["validation"]["kinds"] == {"real": counts["validation"]["total"]},
        f"{object_name}: validation is not real-only",
    )
    exposure = report["sample_exposure"]
    require(sum(int(value) for value in exposure.values()) == 16, f"{object_name}: exposure mismatch")
    metrics = report["metrics"]
    for field in ("macro_f1", "anomaly_f1", "auroc", "normal_false_positive_rate"):
        require(field in metrics, f"{object_name}: metric is missing: {field}")
    return {
        "base_weight_bytes": report["base_weight_bytes"],
        "base_weight_sha256": report["base_weight_sha256"],
        "data_manifest_sha256": stored_data_sha,
        "model_sha256": report["model_sha256"],
        "peak_vram_gib": report["peak_vram_gib"],
        "training_seconds": report["training_seconds"],
        "validation": {
            "n": metrics["n"],
            "macro_f1": metrics["macro_f1"],
            "auroc": metrics["auroc"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/classifier.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/classifier_smoke_validation.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(args.paths)
    config = load_config(args.config.resolve(strict=True))
    root = paths.runs / "cls_smoke"
    evidence = {
        object_name: verify_run(
            root / run_name,
            object_name=object_name,
            config=config,
        )
        for object_name, run_name in RUN_NAMES.items()
    }
    payload = {
        "status": "passed",
        "verified_at": datetime.now(UTC).isoformat(),
        "pipeline_version": config["pipeline_version"],
        "model": config["model"]["name"],
        "model_revision": config["model"]["revision"],
        "base_weight_sha256": config["model"]["sha256"],
        "objects": evidence,
        "test_loaded": False,
        "blocklist_hits": 0,
    }
    output = args.output if args.output.is_absolute() else paths.project_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
