from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.common.integrity import sha256_file
from src.inference.demo_gradio import (
    DemoError,
    build_parser,
    render_outputs,
    verify_checkpoint,
)


def _checkpoint(root: Path, *, role: str, model_sha256: str | None = None) -> Path:
    run = root / role
    weight = run / ("final/model.safetensors" if role == "segmenter" else "model.safetensors")
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"safe-tensor-fixture")
    observed = sha256_file(weight)
    report = {
        "status": "passed",
        "smoke": False,
        "mode": "final",
        "object": "pcb1",
        "model_sha256": model_sha256 or observed,
    }
    (run / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
    return weight


@pytest.mark.parametrize("role", ["classifier", "segmenter"])
def test_verify_checkpoint_binds_weight_to_report(tmp_path: Path, role: str) -> None:
    weight = _checkpoint(tmp_path, role=role)
    verified = verify_checkpoint(weight.parent if role == "classifier" else weight.parent.parent, role=role)
    assert verified.weight == weight
    assert verified.object_name == "pcb1"
    assert verified.model_sha256 == sha256_file(weight)


def test_verify_checkpoint_rejects_hash_mismatch(tmp_path: Path) -> None:
    weight = _checkpoint(tmp_path, role="classifier", model_sha256="0" * 64)
    with pytest.raises(DemoError, match="hash"):
        verify_checkpoint(weight, role="classifier")


def test_render_outputs_has_stable_shapes_and_probabilities() -> None:
    image = np.full((5, 7, 3), 80, dtype=np.uint8)
    probability = np.linspace(0.0, 1.0, 35, dtype=np.float32).reshape(5, 7)
    labels, mask, heatmap, latency = render_outputs(
        image,
        anomaly_probability=0.75,
        pixel_probability=probability,
        threshold=0.5,
        elapsed_ms=12.34,
    )
    assert labels == {"Defect": 0.75, "Normal": 0.25}
    assert mask.shape == (5, 7)
    assert set(np.unique(mask)) == {0, 255}
    assert heatmap.shape == image.shape
    assert heatmap.dtype == np.uint8
    assert "12.3 ms" in latency
    assert "mask coverage" in latency


def test_cli_is_local_and_private_by_default() -> None:
    args = build_parser().parse_args(
        ["--cls-ckpt", "classifier", "--seg-ckpt", "segmenter"]
    )
    assert args.port == 7860
    assert args.share is False
    assert args.inbrowser is False


def test_direct_script_entrypoint_exposes_help() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "src/inference/demo_gradio.py", "--help"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--cls-ckpt" in result.stdout
    assert "--seg-ckpt" in result.stdout
