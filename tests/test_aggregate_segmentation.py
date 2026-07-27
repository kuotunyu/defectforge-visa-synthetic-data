from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from scripts.aggregate_segmentation import (
    FORMAL_GROUPS,
    SegmentationAggregationError,
    build_report,
    import_result_archive,
    logical_rows,
)


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


def _write_result_zip(root: Path, object_name: str) -> Path:
    bundle = root / f"bundle_{object_name}"
    bundle.mkdir()
    validation = {
        "status": "passed",
        "object": object_name,
        "runs": 8,
        "alias_reruns": 0,
        "all_mixed_alias_of": "filtered_syn",
        "fresh_model_reload": True,
    }
    timings = {group: 1.0 for group in FORMAL_GROUPS}
    (bundle / f"m18_{object_name}_validation.json").write_text(
        json.dumps(validation),
        encoding="utf-8",
    )
    (bundle / "timings.json").write_text(json.dumps(timings), encoding="utf-8")
    (bundle / f"segmentation_{object_name}.csv").write_text("fixture\n", encoding="utf-8")
    for group_name in FORMAL_GROUPS:
        run = bundle / "runs" / f"m18_{group_name}_{object_name}_seed42"
        for relative in (
            "training_report.json",
            "run_config.json",
            "data_manifest.json",
            "final/config.json",
            "final/model.safetensors",
        ):
            target = run / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture")
    return Path(
        shutil.make_archive(
            str(root / f"m18_seg_results_{object_name}"),
            "zip",
            bundle,
        )
    )


def test_import_result_archive_is_safe_and_idempotent(tmp_path: Path) -> None:
    _write_result_zip(tmp_path, "pcb1")
    imported = import_result_archive(tmp_path, object_name="pcb1")
    assert imported == tmp_path / "pcb1"
    manifest = json.loads((imported / "import_manifest.json").read_text(encoding="utf-8"))
    assert manifest["members"] == 43
    assert import_result_archive(tmp_path, object_name="pcb1") == imported


def test_import_result_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "m18_seg_results_pcb1.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "no")
    with pytest.raises(SegmentationAggregationError, match="escapes"):
        import_result_archive(tmp_path, object_name="pcb1")
    assert not (tmp_path.parent / "escape.txt").exists()
