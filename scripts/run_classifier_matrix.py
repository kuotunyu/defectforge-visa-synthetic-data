"""Run the locked 38-run M16 classifier matrix with recoverable checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import Paths, load_paths  # isort: skip
from src.training.classifier_data import build_classification_group  # isort: skip
from src.training.train_classifier import load_config  # isort: skip

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
REPLICATE_SEEDS = (43, 44)


class MatrixRunError(RuntimeError):
    """The formal matrix cannot proceed without violating its frozen contract."""


@dataclass(frozen=True, slots=True)
class RunSpec:
    group: str
    object_name: str
    seed: int

    @property
    def run_name(self) -> str:
        return f"m16_{self.group}_{self.object_name}_seed_{self.seed}"


def matrix_plan() -> list[RunSpec]:
    plan = [
        RunSpec(group=group, object_name=object_name, seed=42)
        for group in SEED_42_GROUPS
        for object_name in OBJECTS
    ]
    plan.extend(
        RunSpec(group=group, object_name=object_name, seed=seed)
        for group in REPLICATE_GROUPS
        for object_name in OBJECTS
        for seed in REPLICATE_SEEDS
    )
    if len(plan) != 38 or len({spec.run_name for spec in plan}) != 38:
        raise MatrixRunError("The frozen M16 matrix must contain 38 unique runs")
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/classifier.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_result_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_completed_run(
    *,
    directory: Path,
    spec: RunSpec,
    config: dict[str, Any],
    result_rows: list[dict[str, str]],
) -> bool:
    report_path = directory / "training_report.json"
    if not report_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "status": "passed",
        "mode": "final",
        "smoke": False,
        "run_name": spec.run_name,
        "requested_group": spec.group,
        "object": spec.object_name,
        "seed": spec.seed,
        "requested_total_steps": int(config["training"]["total_steps"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "weight_decay": float(config["training"]["weight_decay"]),
    }
    mismatches = {
        key: (report.get(key), value) for key, value in expected.items() if report.get(key) != value
    }
    if mismatches:
        raise MatrixRunError(
            f"Completed report contract mismatch for {spec.run_name}: {mismatches}"
        )
    signature = str(report.get("run_signature", ""))
    matching_rows = [
        row
        for row in result_rows
        if row.get("run_name") == spec.run_name and row.get("run_signature") == signature
    ]
    if len(matching_rows) != 1:
        raise MatrixRunError(
            f"Expected one matching classification.csv row for {spec.run_name}, "
            f"found {len(matching_rows)}"
        )
    return True


def quarantine_incomplete(root: Path, working: Path) -> Path:
    root_resolved = root.resolve(strict=True)
    working_resolved = working.resolve(strict=True)
    if root_resolved not in working_resolved.parents:
        raise MatrixRunError(f"Refusing to move an out-of-root working directory: {working}")
    quarantine_root = root / "_incomplete"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = quarantine_root / f"{working.name.lstrip('.')}__{timestamp}"
    counter = 1
    while destination.exists():
        destination = quarantine_root / (f"{working.name.lstrip('.')}__{timestamp}__{counter:02d}")
        counter += 1
    shutil.move(str(working), str(destination))
    return destination


def preflight(paths: Paths, config: dict[str, Any], plan: list[RunSpec]) -> None:
    if not bool(config.get("hyperparameters_frozen")):
        raise MatrixRunError("Formal matrix requires frozen Real-only hyperparameters")
    for spec in plan:
        group = build_classification_group(
            paths,
            config,
            group_name=spec.group,
            object_name=spec.object_name,
            seed=spec.seed,
            mode="final",
        )
        if group.requested_group != spec.group or group.object_name != spec.object_name:
            raise MatrixRunError(f"Preflight group mismatch: {spec.run_name}")


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths)
    config_path = args.config.resolve(strict=True)
    config = load_config(config_path)
    plan = matrix_plan()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "runs": [asdict(spec) | {"run_name": spec.run_name} for spec in plan],
                    "total": len(plan),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    preflight(paths, config, plan)
    run_root = paths.runs / str(config["output"]["run_subdirectory"])
    run_root.mkdir(parents=True, exist_ok=True)
    results_path = paths.project_root / str(config["output"]["results_csv"])
    state_path = run_root / "m16_matrix_state.json"
    completed = 0
    skipped = 0
    quarantined: list[str] = []

    for index, spec in enumerate(plan, start=1):
        final_dir = run_root / spec.run_name
        working_dir = run_root / f".{spec.run_name}.working"
        rows = read_result_rows(results_path)
        if final_dir.exists():
            if not validate_completed_run(
                directory=final_dir,
                spec=spec,
                config=config,
                result_rows=rows,
            ):
                raise MatrixRunError(f"Final run directory is incomplete: {final_dir}")
            completed += 1
            skipped += 1
            continue

        if working_dir.exists():
            if validate_completed_run(
                directory=working_dir,
                spec=spec,
                config=config,
                result_rows=rows,
            ):
                working_dir.rename(final_dir)
                completed += 1
                skipped += 1
                continue
            quarantined.append(str(quarantine_incomplete(run_root, working_dir)))

        command = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "training" / "train_classifier.py"),
            "--paths",
            str(args.paths),
            "--config",
            str(config_path),
            "--object",
            spec.object_name,
            "--group",
            spec.group,
            "--seed",
            str(spec.seed),
            "--mode",
            "final",
            "--run-name",
            spec.run_name,
            "--output-dir",
            str(working_dir),
        ]
        atomic_write_json(
            state_path,
            {
                "completed": completed,
                "current_index": index,
                "current_run": spec.run_name,
                "planned": len(plan),
                "quarantined": quarantined,
                "status": "running",
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        rows = read_result_rows(results_path)
        if not validate_completed_run(
            directory=working_dir,
            spec=spec,
            config=config,
            result_rows=rows,
        ):
            raise MatrixRunError(f"Trainer returned without a passed report: {spec.run_name}")
        working_dir.rename(final_dir)
        completed += 1

    result = {
        "completed": completed,
        "planned": len(plan),
        "quarantined": quarantined,
        "skipped": skipped,
        "status": "passed",
        "updated_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(state_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MatrixRunError, OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
