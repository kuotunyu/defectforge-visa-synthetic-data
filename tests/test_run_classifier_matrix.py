import csv
import json
from pathlib import Path

import pytest

from scripts.run_classifier_matrix import (
    REPLICATE_GROUPS,
    MatrixRunError,
    matrix_plan,
    read_result_rows,
    validate_completed_run,
)


def test_matrix_plan_has_38_unique_canonical_runs() -> None:
    plan = matrix_plan()

    assert len(plan) == 38
    assert len({spec.run_name for spec in plan}) == 38
    assert not {"real_60", "syn_500", "base_sd2"} & {spec.group for spec in plan}
    for group in REPLICATE_GROUPS:
        for object_name in ("pcb1", "capsules"):
            assert {
                spec.seed
                for spec in plan
                if spec.group == group and spec.object_name == object_name
            } == {42, 43, 44}


def write_completed_run(
    directory: Path,
    *,
    run_name: str,
    run_signature: str,
) -> None:
    directory.mkdir(parents=True)
    (directory / "training_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "mode": "final",
                "smoke": False,
                "run_name": run_name,
                "requested_group": "real_only",
                "object": "pcb1",
                "seed": 42,
                "requested_total_steps": 100,
                "learning_rate": 0.00001,
                "weight_decay": 0.05,
                "run_signature": run_signature,
            }
        ),
        encoding="utf-8",
    )


def test_validate_completed_run_requires_matching_csv_evidence(tmp_path: Path) -> None:
    spec = next(
        spec
        for spec in matrix_plan()
        if spec.group == "real_only" and spec.object_name == "pcb1" and spec.seed == 42
    )
    directory = tmp_path / spec.run_name
    write_completed_run(directory, run_name=spec.run_name, run_signature="abc")
    config = {
        "training": {
            "total_steps": 100,
            "learning_rate": 0.00001,
            "weight_decay": 0.05,
        }
    }

    assert validate_completed_run(
        directory=directory,
        spec=spec,
        config=config,
        result_rows=[{"run_name": spec.run_name, "run_signature": "abc"}],
    )
    with pytest.raises(MatrixRunError, match="classification.csv"):
        validate_completed_run(
            directory=directory,
            spec=spec,
            config=config,
            result_rows=[],
        )


def test_read_result_rows_handles_absent_and_present_csv(tmp_path: Path) -> None:
    path = tmp_path / "classification.csv"
    assert read_result_rows(path) == []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("run_name", "run_signature"))
        writer.writeheader()
        writer.writerow({"run_name": "run", "run_signature": "sig"})

    assert read_result_rows(path) == [{"run_name": "run", "run_signature": "sig"}]
