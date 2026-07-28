"""Run and evaluate the preregistered M26 validation-only classifier pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file
from src.common.paths import Paths, load_paths
from src.training.classifier_data import build_classification_group, summarize_samples
from src.training.train_classifier import SAMPLER_STRATEGIES


class PilotError(RuntimeError):
    """Raised when the M26 pilot violates its preregistered contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotError(message)


def load_pilot_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "Pilot config must be a mapping")
    require(payload.get("schema_version") == 1, "Unsupported pilot config schema")
    require(payload.get("status") == "preregistered", "Pilot config is not preregistered")
    objects = payload.get("objects")
    require(objects == ["pcb1", "capsules"], "Pilot must cover both frozen objects")
    candidates = payload.get("candidates")
    require(isinstance(candidates, list) and len(candidates) >= 3, "Pilot candidates are incomplete")
    names: set[str] = set()
    for candidate in candidates:
        require(isinstance(candidate, dict), "Invalid pilot candidate")
        name = str(candidate.get("name"))
        require(name not in names, f"Duplicate pilot candidate: {name}")
        names.add(name)
        require(
            candidate.get("sampler_strategy") in SAMPLER_STRATEGIES,
            f"Invalid sampler strategy: {name}",
        )
        share = float(candidate.get("real_bad_share"))
        require(0.0 < share < 1.0, f"Invalid real_bad_share: {name}")
        if candidate["sampler_strategy"] == "class_balanced":
            require(share == 0.5, f"Misleading class-balanced share: {name}")
    selected = set(payload["selection"]["candidates"])
    require(selected and selected < names, "Selection candidates must be a strict candidate subset")
    require("real_only" in names, "Pilot requires a real_only baseline")
    return payload


def _run_name(candidate: Mapping[str, Any], object_name: str, seed: int) -> str:
    return f"m26_{candidate['name']}_{object_name}_seed_{seed}_dev"


def _command(
    *,
    paths_file: Path,
    base_config: Path,
    candidate: Mapping[str, Any],
    object_name: str,
    seed: int,
    total_steps: int,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "src/training/train_classifier.py",
        "--paths",
        str(paths_file),
        "--config",
        str(base_config),
        "--object",
        object_name,
        "--seed",
        str(seed),
        "--group",
        str(candidate["group"]),
        "--run-name",
        _run_name(candidate, object_name, seed),
        "--mode",
        "development",
        "--total-steps",
        str(total_steps),
        "--sampler-strategy",
        str(candidate["sampler_strategy"]),
        "--real-bad-share",
        str(candidate["real_bad_share"]),
        "--output-dir",
        str(output_dir),
    ]
    if candidate["group"] != "real_only":
        command.append("--experimental-synthetic-development")
    return command


def _load_completed_report(
    output_dir: Path,
    *,
    candidate: Mapping[str, Any],
    object_name: str,
    seed: int,
) -> dict[str, Any] | None:
    if not output_dir.exists():
        return None
    report_path = output_dir / "training_report.json"
    require(report_path.is_file(), f"Partial pilot output requires manual review: {output_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report.get("status") == "passed", f"Pilot run did not pass: {output_dir}")
    require(report.get("mode") == "development", f"Pilot loaded test data: {output_dir}")
    require(report.get("object") == object_name, f"Pilot object mismatch: {output_dir}")
    require(report.get("seed") == seed, f"Pilot seed mismatch: {output_dir}")
    require(
        report.get("sampler_strategy") == candidate["sampler_strategy"],
        f"Pilot sampler mismatch: {output_dir}",
    )
    require(
        float(report.get("real_bad_share")) == float(candidate["real_bad_share"]),
        f"Pilot domain share mismatch: {output_dir}",
    )
    require(
        bool(report.get("experimental_synthetic_development"))
        == (candidate["group"] != "real_only"),
        f"Pilot experimental flag mismatch: {output_dir}",
    )
    return report


def evaluate_gate(
    reports: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    objects = [str(value) for value in config["objects"]]
    baseline = reports["real_only"]
    selected_names = [str(value) for value in config["selection"]["candidates"]]
    primary = str(config["selection"]["primary_metric"])
    secondary = str(config["selection"]["secondary_metric"])

    def mean_metric(candidate_name: str, metric: str) -> float:
        return sum(float(reports[candidate_name][obj]["metrics"][metric]) for obj in objects) / len(
            objects
        )

    ranking = sorted(
        selected_names,
        key=lambda name: (mean_metric(name, primary), mean_metric(name, secondary), name),
        reverse=True,
    )
    winner = ranking[0]
    gate = config["gate"]
    macro_tolerance = float(gate["per_object_macro_f1_tolerance"])
    auroc_tolerance = float(gate["per_object_auroc_tolerance"])
    mean_gain_required = float(gate["mean_macro_f1_min_gain"])
    object_checks: dict[str, Any] = {}
    for object_name in objects:
        winner_metrics = reports[winner][object_name]["metrics"]
        baseline_metrics = baseline[object_name]["metrics"]
        macro_delta = float(winner_metrics["macro_f1"]) - float(baseline_metrics["macro_f1"])
        auroc_delta = float(winner_metrics["auroc"]) - float(baseline_metrics["auroc"])
        object_checks[object_name] = {
            "macro_f1_delta": macro_delta,
            "auroc_delta": auroc_delta,
            "macro_f1_noninferior": macro_delta >= -macro_tolerance,
            "auroc_noninferior": auroc_delta >= -auroc_tolerance,
        }
    mean_macro_gain = mean_metric(winner, "macro_f1") - mean_metric("real_only", "macro_f1")
    noninferior = all(
        check["macro_f1_noninferior"] and check["auroc_noninferior"]
        for check in object_checks.values()
    )
    passed = noninferior and mean_macro_gain >= mean_gain_required
    return {
        "status": "passed" if passed else "stopped",
        "winner": winner,
        "ranking": ranking,
        "mean_macro_f1_gain_vs_real_only": mean_macro_gain,
        "object_checks": object_checks,
        "confirmatory_run_authorized_by_gate": passed,
    }


def _result_record(report: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "run_name": report["run_name"],
        "run_signature": report["run_signature"],
        "output_directory_name": output_dir.name,
        "object": report["object"],
        "sampler_strategy": report["sampler_strategy"],
        "real_bad_share": report["real_bad_share"],
        "executed_steps": report["executed_steps"],
        "best_step": report["best_step"],
        "sample_exposure": report["sample_exposure"],
        "metrics": report["metrics"],
        "training_seconds": report["training_seconds"],
        "peak_vram_gib": report["peak_vram_gib"],
        "data_manifest_sha256": report["data_manifest_sha256"],
        "model_sha256": report["model_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=Path("configs/classifier_v2_pilot.yaml"),
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths_file = args.paths.resolve(strict=True)
    paths: Paths = load_paths(paths_file)
    pilot_path = args.pilot_config.resolve(strict=True)
    config = load_pilot_config(pilot_path)
    base_config = (paths.project_root / str(config["base_config"])).resolve(strict=True)
    base_payload = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    total_steps = int(config["total_steps"])
    run_root = paths.runs / str(config["output"]["run_subdirectory"])
    plans: list[dict[str, Any]] = []
    for candidate in config["candidates"]:
        for object_name in config["objects"]:
            group = build_classification_group(
                paths,
                base_payload,
                group_name=str(candidate["group"]),
                object_name=str(object_name),
                seed=seed,
                mode="development",
                allow_synthetic_development=candidate["group"] != "real_only",
            )
            run_name = _run_name(candidate, str(object_name), seed)
            output_dir = run_root / run_name
            plans.append(
                {
                    "candidate": candidate,
                    "object": str(object_name),
                    "output_dir": output_dir,
                    "counts": summarize_samples(group.train),
                    "command": _command(
                        paths_file=paths_file,
                        base_config=base_config,
                        candidate=candidate,
                        object_name=str(object_name),
                        seed=seed,
                        total_steps=total_steps,
                        output_dir=output_dir,
                    ),
                }
            )
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "pilot_config_sha256": sha256_file(pilot_path),
                    "run_count": len(plans),
                    "runs": [
                        {
                            "candidate": plan["candidate"]["name"],
                            "object": plan["object"],
                            "output_directory_name": plan["output_dir"].name,
                            "counts": plan["counts"],
                            "command": plan["command"],
                        }
                        for plan in plans
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    reports: dict[str, dict[str, dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for plan in plans:
        candidate = plan["candidate"]
        candidate_name = str(candidate["name"])
        object_name = plan["object"]
        output_dir = plan["output_dir"]
        report = _load_completed_report(
            output_dir,
            candidate=candidate,
            object_name=object_name,
            seed=seed,
        )
        if report is None:
            subprocess.run(plan["command"], cwd=paths.project_root, check=True)
            report = _load_completed_report(
                output_dir,
                candidate=candidate,
                object_name=object_name,
                seed=seed,
            )
            require(report is not None, f"Pilot report was not created: {output_dir}")
        reports.setdefault(candidate_name, {})[object_name] = report
        records.append(_result_record(report, output_dir))

    gate = evaluate_gate(reports, config)
    result = {
        "schema_version": 1,
        "status": "passed",
        "experiment": "M26 validation-only domain-mixing pilot",
        "test_data_loaded": False,
        "pilot_config_sha256": sha256_file(pilot_path),
        "base_config_sha256": sha256_file(base_config),
        "seed": seed,
        "total_steps": total_steps,
        "runs": records,
        "gate": gate,
    }
    result_path = (paths.project_root / str(config["output"]["result"])).resolve(strict=False)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    temporary.replace(result_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
