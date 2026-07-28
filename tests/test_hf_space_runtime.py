from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from deploy.hf_space.runtime import SpaceContractError, load_manifest, render_outputs
from src.common.integrity import sha256_file


def _record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _manifest(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    objects: dict[str, object] = {}
    for object_name in ("pcb1", "capsules"):
        classifier = models / object_name / "classifier" / "model.safetensors"
        segmenter = models / object_name / "segmenter" / "model.safetensors"
        classifier.parent.mkdir(parents=True)
        segmenter.parent.mkdir(parents=True)
        classifier.write_bytes(f"{object_name}-classifier".encode())
        segmenter.write_bytes(f"{object_name}-segmenter".encode())
        objects[object_name] = {
            "classifier": _record(classifier, tmp_path),
            "segmenter": _record(segmenter, tmp_path),
        }
    manifest = models / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "objects": objects,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_space_manifest_binds_all_four_model_files(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = load_manifest(manifest)
    assert tuple(payload["objects"]) == ("pcb1", "capsules")


def test_space_manifest_rejects_changed_weight(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    weight = tmp_path / "models/pcb1/classifier/model.safetensors"
    weight.write_bytes(b"changed")
    with pytest.raises(SpaceContractError, match="size changed|SHA256 changed"):
        load_manifest(manifest)


def test_space_render_outputs_respects_visualization_threshold() -> None:
    image = np.full((4, 6, 3), 90, dtype=np.uint8)
    probability = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(4, 6)
    labels, mask, heatmap = render_outputs(
        image,
        anomaly_probability=0.7,
        pixel_probability=probability,
        threshold=0.5,
    )
    assert labels == {"Defect": 0.7, "Normal": pytest.approx(0.3)}
    assert mask.shape == (4, 6)
    assert set(np.unique(mask)) == {0, 255}
    assert heatmap.shape == image.shape


@pytest.mark.parametrize("script", ["package_hf_space.py", "publish_hf_space.py"])
def test_space_scripts_expose_help(script: str) -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, f"scripts/{script}", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_space_requirements_are_fully_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "deploy/hf_space/requirements.txt").read_text(encoding="utf-8")
    lines = [line for line in requirements.splitlines() if line.strip()]
    assert lines
    assert all("==" in line for line in lines)
