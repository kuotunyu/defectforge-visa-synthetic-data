"""Verify M16 Real-only tuning and the frozen common hyperparameters."""

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

from src.common.paths import load_paths
from src.training.train_classifier import load_config

CANDIDATES = {
    0.00001: {
        "pcb1": "m16_tune_pcb1_lr1e5_s300",
        "capsules": "m16_tune_capsules_lr1e5_s300",
    },
    0.00003: {
        "pcb1": "m16_tune_pcb1_lr3e5_s300",
        "capsules": "m16_tune_capsules_lr3e5_s300",
    },
    0.0001: {
        "pcb1": "m16_tune_pcb1_lr1e4_s300",
        "capsules": "m16_tune_capsules_lr1e4_s300",
    },
}


class ClassifierTuningValidationError(RuntimeError):
    """Raised when Real-only tuning evidence is invalid or incomplete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClassifierTuningValidationError(message)


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClassifierTuningValidationError(f"Invalid JSON: {path}") from exc
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def choose_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(rows, "No tuning candidates")
    return max(
        rows,
        key=lambda row: (
            row["mean_macro_f1"],
            row["mean_auroc"],
            -row["learning_rate"],
        ),
    )


def verify_candidate(
    root: Path,
    *,
    learning_rate: float,
    runs: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    objects: dict[str, Any] = {}
    for object_name, run_name in runs.items():
        run_root = root / run_name
        report = load_json_object(run_root / "training_report.json")
        data = load_json_object(run_root / "data_manifest.json")
        require(report.get("status") == "passed", f"{run_name}: did not pass")
        require(report.get("mode") == "development", f"{run_name}: wrong mode")
        require(report.get("smoke") is False, f"{run_name}: smoke is not tuning")
        require(report.get("object") == object_name, f"{run_name}: object mismatch")
        require(
            report.get("requested_group") == report.get("canonical_group") == "real_only",
            f"{run_name}: non-Real-only tuning",
        )
        require(report.get("seed") == 42, f"{run_name}: seed mismatch")
        require(
            report.get("learning_rate") == learning_rate,
            f"{run_name}: learning rate mismatch",
        )
        require(report.get("weight_decay") == 0.05, f"{run_name}: weight decay mismatch")
        require(
            report.get("requested_total_steps") == 300,
            f"{run_name}: search budget mismatch",
        )
        require(
            report.get("base_weight_sha256") == config["model"]["sha256"],
            f"{run_name}: base model mismatch",
        )
        require(data["counts"]["test"]["total"] == 0, f"{run_name}: tuning loaded test")
        require(
            data["counts"]["validation"]["kinds"]
            == {"real": data["counts"]["validation"]["total"]},
            f"{run_name}: validation is not real-only",
        )
        metrics = report["metrics"]
        objects[object_name] = {
            "run_name": run_name,
            "best_step": report["best_step"],
            "executed_steps": report["executed_steps"],
            "macro_f1": metrics["macro_f1"],
            "anomaly_f1": metrics["anomaly_f1"],
            "auroc": metrics["auroc"],
            "normal_false_positive_rate": metrics["normal_false_positive_rate"],
            "validation_n": metrics["n"],
            "training_seconds": report["training_seconds"],
            "peak_vram_gib": report["peak_vram_gib"],
            "data_manifest_sha256": report["data_manifest_sha256"],
        }
    return {
        "learning_rate": learning_rate,
        "weight_decay": 0.05,
        "mean_macro_f1": sum(row["macro_f1"] for row in objects.values()) / len(objects),
        "mean_auroc": sum(row["auroc"] for row in objects.values()) / len(objects),
        "objects": objects,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {candidate['learning_rate']:.0e} | {candidate['mean_macro_f1']:.6f} | "
        f"{candidate['mean_auroc']:.6f} | "
        f"{candidate['objects']['pcb1']['best_step']} | "
        f"{candidate['objects']['capsules']['best_step']} |"
        for candidate in payload["candidates"]
    )
    selected = payload["selected"]
    return f"""# M16 Classifier Hyperparameter Tuning

**Status:** `{payload["status"]}`

**Selection data:** Real-only frozen validation only; test inventory was empty for every run.

| Learning rate | Mean Macro-F1 | Mean AUROC | pcb1 best step | capsules best step |
|---:|---:|---:|---:|---:|
{rows}

## Frozen common setting

- Learning rate: `{selected["learning_rate"]}`
- Weight decay: `{selected["weight_decay"]}`
- Final refit steps: `{selected["final_total_steps"]}`
- Batch size: `{selected["batch_size"]}`
- Selection rule: mean Macro-F1, then mean AUROC, then lower learning rate
- The final step budget is the maximum best step across both objects.
- All formal groups use this exact setting; no synthetic group may be retuned.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/classifier.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/classifier_tuning.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/classifier_tuning.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(args.paths)
    config = load_config(args.config.resolve(strict=True))
    root = paths.runs / "cls"
    candidates = [
        verify_candidate(
            root,
            learning_rate=learning_rate,
            runs=runs,
            config=config,
        )
        for learning_rate, runs in CANDIDATES.items()
    ]
    winner = choose_candidate(candidates)
    final_total_steps = max(
        int(row["best_step"]) for row in winner["objects"].values()
    )
    training = config["training"]
    require(config.get("hyperparameters_frozen") is True, "Hyperparameters are not frozen")
    require(
        float(training["learning_rate"]) == winner["learning_rate"],
        "Frozen learning rate does not match tuning winner",
    )
    require(float(training["weight_decay"]) == winner["weight_decay"], "Weight decay changed")
    require(
        int(training["total_steps"]) == final_total_steps,
        "Frozen total steps do not match the common selected budget",
    )
    payload = {
        "status": "passed",
        "verified_at": datetime.now(UTC).isoformat(),
        "test_loaded": False,
        "selection_rule": "mean_macro_f1_then_mean_auroc_then_lower_lr",
        "candidates": candidates,
        "selected": {
            "learning_rate": winner["learning_rate"],
            "weight_decay": winner["weight_decay"],
            "final_total_steps": final_total_steps,
            "batch_size": training["batch_size"],
            "winner_mean_macro_f1": winner["mean_macro_f1"],
            "winner_mean_auroc": winner["mean_auroc"],
        },
    }
    output = args.output if args.output.is_absolute() else paths.project_root / args.output
    report = args.report if args.report.is_absolute() else paths.project_root / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
