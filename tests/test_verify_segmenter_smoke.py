from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
from torch import zeros

from scripts.verify_segmenter_smoke import SegmenterSmokeError, canonical_sha256, verify_run
from src.common.integrity import sha256_file


def _smoke_fixture(root: Path, *, test: list | None = None) -> tuple[dict, str]:
    run_dir = root / "m18_smoke_pcb1_v1"
    final = run_dir / "final"
    final.mkdir(parents=True)
    model_path = final / "model.safetensors"
    save_file(
        {
            "decode_head.classifier.weight": zeros(1, 1, 1, 1),
            "decode_head.classifier.bias": zeros(1),
        },
        model_path,
    )
    data = {
        "mode": "development",
        "split_manifest_sha256": "s" * 64,
        "train": [{"kind": "real"}],
        "validation": [{"kind": "real"}],
        "test": [] if test is None else test,
    }
    data_sha = canonical_sha256(data)
    run_config = {
        "schema_version": 1,
        "data_manifest_sha256": data_sha,
    }
    signature = canonical_sha256(run_config)
    run_config["run_signature"] = signature
    report = {
        "status": "passed",
        "run_signature": signature,
        "smoke": True,
        "mode": "development",
        "object": "pcb1",
        "requested_group": "real_only",
        "canonical_group": "real_only",
        "requested_total_steps": 1,
        "executed_steps": 1,
        "model_repository": "repo",
        "model_revision": "rev",
        "base_weight_sha256": "b" * 64,
        "input_size": 512,
        "peak_vram_gib": 2.0,
        "wall_clock_seconds": 1.0,
        "metrics": {
            "dice": 0.1,
            "miou": 0.2,
            "pixel_auroc": 0.3,
            "aupro": 0.4,
        },
        "data_manifest_sha256": data_sha,
        "model_sha256": sha256_file(model_path),
    }
    for path, value in (
        (run_dir / "data_manifest.json", data),
        (run_dir / "run_config.json", run_config),
        (run_dir / "training_report.json", report),
    ):
        path.write_text(json.dumps(value), encoding="utf-8")
    return {
        "model": {
            "repository": "repo",
            "revision": "rev",
            "sha256": "b" * 64,
        }
    }, "s" * 64


def test_verify_run_accepts_development_smoke(tmp_path: Path) -> None:
    config, manifest_sha = _smoke_fixture(tmp_path)
    result = verify_run(
        run_dir=tmp_path / "m18_smoke_pcb1_v1",
        object_name="pcb1",
        config=config,
        manifest_sha256=manifest_sha,
    )
    assert result["test"] == 0
    assert result["peak_vram_gib"] == 2.0


def test_verify_run_rejects_test_loaded_by_development(tmp_path: Path) -> None:
    config, manifest_sha = _smoke_fixture(tmp_path, test=[{"kind": "real"}])
    with pytest.raises(SegmenterSmokeError, match="frozen test"):
        verify_run(
            run_dir=tmp_path / "m18_smoke_pcb1_v1",
            object_name="pcb1",
            config=config,
            manifest_sha256=manifest_sha,
        )
