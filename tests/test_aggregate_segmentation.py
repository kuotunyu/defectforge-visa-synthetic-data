from __future__ import annotations

import json
from pathlib import Path

from scripts.aggregate_segmentation import FORMAL_GROUPS, build_report, logical_rows


def _write_run(root: Path, group: str, object_name: str, value: float) -> None:
    run = root / f"m18_{group}_{object_name}_seed42"
    run.mkdir(parents=True)
    report = {
        "run_name": run.name,
        "run_signature": f"signature-{group}",
        "seed": 42,
        "model": "segformer_b0",
        "model_revision": "revision",
        "input_size": 512,
        "executed_steps": 500,
        "batch_size": 4,
        "learning_rate": 6e-5,
        "weight_decay": 0.01,
        "train_counts": {
            "total": 10,
            "real": 5,
            "synthetic": 5,
        },
        "evaluation_counts": {"total": 4},
        "metrics": {
            "dice": value,
            "miou": value,
            "pixel_auroc": value,
            "aupro": value,
        },
        "peak_vram_gib": 2.0,
        "training_seconds": 10.0,
        "model_sha256": "m" * 64,
        "data_manifest_sha256": "d" * 64,
    }
    data = {
        "train": [
            {"kind": "real", "has_defect": True},
            {"kind": "synthetic", "has_defect": True},
        ]
    }
    (run / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
    (run / "data_manifest.json").write_text(json.dumps(data), encoding="utf-8")


def test_logical_rows_adds_nonphysical_all_mixed_alias(tmp_path: Path) -> None:
    for index, group in enumerate(FORMAL_GROUPS):
        _write_run(tmp_path, group, "pcb1", index / 10)
    rows = logical_rows(tmp_path, "pcb1")
    assert len(rows) == 9
    alias = rows[-1]
    filtered = next(row for row in rows if row["logical_group"] == "filtered_syn")
    assert alias["logical_group"] == "all_mixed"
    assert alias["physical_run"] == "false"
    assert alias["alias_of"] == "filtered_syn"
    assert alias["run_signature"] == filtered["run_signature"]
    assert alias["dice"] == filtered["dice"]


def test_build_report_uses_two_object_raw_rows(tmp_path: Path) -> None:
    rows = []
    for object_name in ("pcb1", "capsules"):
        object_root = tmp_path / object_name
        for index, group in enumerate(FORMAL_GROUPS):
            _write_run(object_root, group, object_name, 0.1 + index / 10)
        rows.extend(logical_rows(object_root, object_name))
    report = build_report(rows)
    assert "Two-object macro mean" in report
    assert "no; cites filtered_syn" in report
    assert "zero real defect **images/pixels**" in report
