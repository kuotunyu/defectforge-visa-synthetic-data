"""Independently verify the complete M16 classifier matrix and result ledger."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file  # isort: skip
from src.common.paths import load_paths  # isort: skip
from src.training.classifier_data import (  # isort: skip
    build_classification_group,
    group_payload_sha256,
)
from src.training.train_classifier import (  # isort: skip
    RESULT_COLUMNS,
    canonical_json_sha256,
    load_config,
)

OBJECTS = ("pcb1", "capsules")
SEED_42_GROUPS = (
    "real_only",
    "std_aug",
    "unfiltered_syn",
    "filtered_syn",
    "full_real",
    "real_20",
    "syn_125",
    "syn_250",
    "src_procedural",
    "src_copypaste",
    "src_diffusion",
    "procedural_norealstats",
    "bucket_original",
    "bucket_searched",
    "base_sdxl",
)
REPLICATE_GROUPS = ("real_only", "filtered_syn")
ALIASES = ("real_60", "syn_500", "base_sd2")
FLOAT_RESULT_FIELDS = (
    "learning_rate",
    "weight_decay",
    "macro_f1",
    "anomaly_f1",
    "auroc",
    "normal_false_positive_rate",
    "peak_vram_gib",
    "training_seconds",
)


class ClassifierMatrixValidationError(RuntimeError):
    """Formal classifier evidence is missing or violates the frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClassifierMatrixValidationError(message)


def expected_plan() -> list[tuple[str, str, int, str]]:
    plan = [
        (group, object_name, 42, f"m16_{group}_{object_name}_seed_42")
        for group in SEED_42_GROUPS
        for object_name in OBJECTS
    ]
    plan.extend(
        (
            group,
            object_name,
            seed,
            f"m16_{group}_{object_name}_seed_{seed}",
        )
        for group in REPLICATE_GROUPS
        for object_name in OBJECTS
        for seed in (43, 44)
    )
    require(len(plan) == len({item[3] for item in plan}) == 38, "Invalid expected plan")
    return plan


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClassifierMatrixValidationError(f"Invalid JSON: {path}") from error
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def load_csv_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = tuple(reader.fieldnames or ())
            rows = list(reader)
    except OSError as error:
        raise ClassifierMatrixValidationError(f"Cannot read result ledger: {path}") from error
    return header, rows


def as_int(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as error:
        raise ClassifierMatrixValidationError(f"Invalid integer field {field}") from error


def as_float(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as error:
        raise ClassifierMatrixValidationError(f"Invalid float field {field}") from error
    require(math.isfinite(value), f"Non-finite result field: {field}")
    return value


def same_float(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def verify_run(
    *,
    run_root: Path,
    row: dict[str, str],
    group_name: str,
    object_name: str,
    seed: int,
    config: dict[str, Any],
    config_sha256: str,
    paths: Any,
    blocked_sha256: set[str],
) -> dict[str, Any]:
    run_name = f"m16_{group_name}_{object_name}_seed_{seed}"
    report = load_json_object(run_root / "training_report.json")
    run_config = load_json_object(run_root / "run_config.json")
    data_manifest = load_json_object(run_root / "data_manifest.json")
    model_path = run_root / "model.safetensors"
    require(model_path.is_file(), f"{run_name}: trained model is missing")

    expected_scalars = {
        "status": "passed",
        "mode": "final",
        "smoke": False,
        "run_name": run_name,
        "requested_group": group_name,
        "object": object_name,
        "seed": seed,
        "requested_total_steps": int(config["training"]["total_steps"]),
        "batch_size": int(config["training"]["batch_size"]),
        "model": config["model"]["name"],
        "model_revision": config["model"]["revision"],
        "base_weight_sha256": config["model"]["sha256"],
        "config_sha256": config_sha256,
    }
    for field, expected in expected_scalars.items():
        require(report.get(field) == expected, f"{run_name}: report mismatch for {field}")
    require(
        same_float(report.get("learning_rate"), config["training"]["learning_rate"]),
        f"{run_name}: learning rate mismatch",
    )
    require(
        same_float(report.get("weight_decay"), config["training"]["weight_decay"]),
        f"{run_name}: weight decay mismatch",
    )
    require(
        1 <= int(report["executed_steps"]) <= int(report["requested_total_steps"]),
        f"{run_name}: executed step count is invalid",
    )
    require(float(report.get("peak_vram_gib", 0.0)) > 0.0, f"{run_name}: VRAM missing")
    require(
        report.get("model_sha256") == sha256_file(model_path),
        f"{run_name}: model hash mismatch",
    )

    stored_data_sha = str(data_manifest.pop("sha256", ""))
    require(
        stored_data_sha == canonical_json_sha256(data_manifest),
        f"{run_name}: portable data manifest hash mismatch",
    )
    expected_group = build_classification_group(
        paths,
        config,
        group_name=group_name,
        object_name=object_name,
        seed=seed,
        mode="final",
    )
    expected_data_sha = group_payload_sha256(expected_group)
    require(
        stored_data_sha
        == expected_data_sha
        == report.get("data_manifest_sha256")
        == run_config.get("data_manifest_sha256"),
        f"{run_name}: data provenance links mismatch",
    )
    train_hashes = {str(sample["sha256"]) for sample in data_manifest["train"]}
    validation_hashes = {str(sample["sha256"]) for sample in data_manifest["validation"]}
    test_hashes = {str(sample["sha256"]) for sample in data_manifest["test"]}
    require(
        not (train_hashes & test_hashes) and not (validation_hashes & test_hashes),
        f"{run_name}: train/validation hashes overlap test",
    )
    require(
        not (train_hashes & blocked_sha256) and not (validation_hashes & blocked_sha256),
        f"{run_name}: train/validation input hit the frozen test blocklist",
    )
    require(
        all(
            sample["root"] == "visa_raw"
            and sample["kind"] == "real"
            and sample["source_name"] == "real_test"
            for sample in data_manifest["test"]
        ),
        f"{run_name}: formal test inventory is not exclusively frozen real test",
    )
    require(
        len(test_hashes) == int(data_manifest["counts"]["test"]["total"]) > 0,
        f"{run_name}: formal test inventory is empty or duplicated",
    )
    require(
        test_hashes.issubset(blocked_sha256),
        f"{run_name}: frozen test images are absent from the published blocklist",
    )

    signature_payload = {
        field: run_config[field]
        for field in (
            "pipeline_version",
            "config_sha256",
            "data_manifest_sha256",
            "model_sha256",
            "seed",
            "total_steps",
            "learning_rate",
            "weight_decay",
            "mode",
            "smoke",
        )
    }
    expected_signature = canonical_json_sha256(signature_payload)
    require(
        report.get("run_signature")
        == run_config.get("run_signature")
        == row.get("run_signature")
        == expected_signature,
        f"{run_name}: run signature mismatch",
    )

    report_counts = report["train_counts"]
    metrics = report["metrics"]
    expected_row = {
        "run_name": run_name,
        "requested_group": group_name,
        "canonical_group": report["canonical_group"],
        "object": object_name,
        "seed": seed,
        "model": config["model"]["name"],
        "model_revision": config["model"]["revision"],
        "input_size": config["model"]["input_size"],
        "total_steps": report["executed_steps"],
        "batch_size": report["batch_size"],
        "train_total": report_counts["total"],
        "train_real": report_counts["kinds"].get("real", 0),
        "train_synthetic": report_counts["kinds"].get("synthetic", 0),
        "sampled_real_good": report["sample_exposure"]["real_good"],
        "sampled_real_bad": report["sample_exposure"]["real_bad"],
        "sampled_synthetic_bad": report["sample_exposure"]["synthetic_bad"],
        "test_total": report["evaluation_counts"]["total"],
        "test_good": report["evaluation_counts"]["labels"].get("good", 0),
        "test_bad": report["evaluation_counts"]["labels"].get("bad", 0),
        "data_manifest_sha256": stored_data_sha,
        "split_manifest_sha256": report["split_manifest_sha256"],
        "selection_sha256": report["selection_sha256"],
    }
    for field, expected in expected_row.items():
        if isinstance(expected, int):
            require(as_int(row, field) == expected, f"{run_name}: CSV mismatch for {field}")
        else:
            require(row.get(field) == str(expected), f"{run_name}: CSV mismatch for {field}")
    float_expected = {
        "learning_rate": report["learning_rate"],
        "weight_decay": report["weight_decay"],
        "macro_f1": metrics["macro_f1"],
        "anomaly_f1": metrics["anomaly_f1"],
        "auroc": metrics["auroc"],
        "normal_false_positive_rate": metrics["normal_false_positive_rate"],
        "peak_vram_gib": report["peak_vram_gib"],
        "training_seconds": report["training_seconds"],
    }
    for field, expected in float_expected.items():
        require(same_float(as_float(row, field), expected), f"{run_name}: CSV mismatch for {field}")
    for field in ("macro_f1", "anomaly_f1", "auroc", "normal_false_positive_rate"):
        require(0.0 <= as_float(row, field) <= 1.0, f"{run_name}: metric out of range")

    return {
        "anomaly_f1": float(metrics["anomaly_f1"]),
        "auroc": float(metrics["auroc"]),
        "canonical_group": str(report["canonical_group"]),
        "data_manifest_sha256": stored_data_sha,
        "executed_steps": int(report["executed_steps"]),
        "macro_f1": float(metrics["macro_f1"]),
        "normal_false_positive_rate": float(metrics["normal_false_positive_rate"]),
        "object": object_name,
        "peak_vram_gib": float(report["peak_vram_gib"]),
        "run_name": run_name,
        "seed": seed,
        "training_seconds": float(report["training_seconds"]),
    }


def aggregate_replicates(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        group_name = str(run["run_name"])[len("m16_") :].split(
            f"_{run['object']}_seed_",
            maxsplit=1,
        )[0]
        if group_name in REPLICATE_GROUPS:
            grouped[(group_name, str(run["object"]))].append(run)
    result: dict[str, Any] = {}
    for (group_name, object_name), values in sorted(grouped.items()):
        require(
            {int(value["seed"]) for value in values} == {42, 43, 44},
            f"Replicate coverage mismatch: {group_name}/{object_name}",
        )
        key = f"{group_name}/{object_name}"
        result[key] = {
            "seeds": [42, 43, 44],
            "macro_f1_mean": statistics.fmean(value["macro_f1"] for value in values),
            "macro_f1_std": statistics.stdev(value["macro_f1"] for value in values),
            "auroc_mean": statistics.fmean(value["auroc"] for value in values),
            "auroc_std": statistics.stdev(value["auroc"] for value in values),
        }
    require(len(result) == 4, "Expected four three-seed aggregate groups")
    return result


def render_markdown(payload: dict[str, Any]) -> str:
    seed42 = sorted(
        (run for run in payload["runs"] if run["seed"] == 42),
        key=lambda run: (run["run_name"], run["object"]),
    )
    rows = "\n".join(
        f"| {run['run_name']} | {run['macro_f1']:.4f} | {run['anomaly_f1']:.4f} | "
        f"{run['auroc']:.4f} | {run['normal_false_positive_rate']:.4f} |"
        for run in seed42
    )
    replicate_rows = "\n".join(
        f"| {name} | {values['macro_f1_mean']:.4f} ± {values['macro_f1_std']:.4f} | "
        f"{values['auroc_mean']:.4f} ± {values['auroc_std']:.4f} |"
        for name, values in payload["three_seed"].items()
    )
    return f"""# M16 Classification Results

**Status:** `{payload["status"]}`

**Formal runs:** `{payload["formal_runs"]}`

**Test source:** frozen `2cls_highshot` test only

## Seed 42

| Run | Macro-F1 | Anomaly F1 | AUROC | Normal FPR |
|---|---:|---:|---:|---:|
{rows}

## Three-seed preregistered groups

| Group / object | Macro-F1 mean ± std | AUROC mean ± std |
|---|---:|---:|
{replicate_rows}

All run signatures, portable data manifests, trained model hashes, frozen test inventories,
CSV rows, and test-disjoint train/validation hashes were independently verified.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/classifier.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/classifier_matrix_validation.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/classifier_results.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(args.paths)
    config_path = args.config.resolve(strict=True)
    config = load_config(config_path)
    config_sha256 = sha256_file(config_path)
    blocklist_path = paths.splits / "test_blocklist.json"
    blocklist = load_json_object(blocklist_path)
    blocked_sha256 = {str(value) for value in blocklist["sha256"]}
    require(blocked_sha256, "Published test blocklist is empty")
    results_path = paths.project_root / str(config["output"]["results_csv"])
    header, rows = load_csv_rows(results_path)
    require(header == RESULT_COLUMNS, "classification.csv header mismatch")
    require(len(rows) == 38, f"Expected 38 result rows, found {len(rows)}")
    require(
        len({row["run_name"] for row in rows}) == len({row["run_signature"] for row in rows}) == 38,
        "Result run names or signatures are not unique",
    )
    expected = expected_plan()
    expected_names = {item[3] for item in expected}
    require({row["run_name"] for row in rows} == expected_names, "Result run inventory mismatch")
    require(
        not any(row["requested_group"] in ALIASES for row in rows),
        "Alias group was executed as a separate formal run",
    )

    run_root = paths.runs / str(config["output"]["run_subdirectory"])
    require(not list(run_root.glob(".*.working")), "Incomplete working directories remain")
    matrix_state = load_json_object(run_root / "m16_matrix_state.json")
    require(
        matrix_state.get("status") == "passed"
        and matrix_state.get("planned") == 38
        and matrix_state.get("completed") == 38,
        "Matrix state is not complete",
    )
    row_by_name = {row["run_name"]: row for row in rows}
    evidence = [
        verify_run(
            run_root=run_root / run_name,
            row=row_by_name[run_name],
            group_name=group_name,
            object_name=object_name,
            seed=seed,
            config=config,
            config_sha256=config_sha256,
            paths=paths,
            blocked_sha256=blocked_sha256,
        )
        for group_name, object_name, seed, run_name in expected
    ]
    payload = {
        "status": "passed",
        "verified_at": datetime.now(UTC).isoformat(),
        "formal_runs": len(evidence),
        "aliases_not_rerun": list(ALIASES),
        "blocklist_hits": 0,
        "blocklist_sha256": sha256_file(blocklist_path),
        "config_sha256": config_sha256,
        "model_sha256": config["model"]["sha256"],
        "runs": evidence,
        "three_seed": aggregate_replicates(evidence),
    }
    output = args.output if args.output.is_absolute() else paths.project_root / args.output
    report = args.report if args.report.is_absolute() else paths.project_root / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
