from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

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
    assert labels == {
        "Defect（異常）": 0.7,
        "Normal（正常）": pytest.approx(0.3),
    }
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


def test_space_card_metadata_meets_hub_limits() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "deploy/hf_space/README.md").read_text(encoding="utf-8")
    _, front_matter, _ = readme.split("---", maxsplit=2)
    metadata = yaml.safe_load(front_matter)
    assert metadata["sdk"] == "gradio"
    assert len(metadata["short_description"]) <= 60


def test_space_app_is_zh_tw_first_and_has_guided_empty_state() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "deploy/hf_space/app.py").read_text(encoding="utf-8")
    assert "瑕疵影像檢測 Demo" in source
    assert "aria-label='檢測流程'" in source
    assert "<li><b>01</b><span>選擇物件</span></li>" in source
    assert "選擇物件並上傳影像" in source
    assert "按下「開始檢測」後，結果會顯示在這裡" in source
    assert "placeholder=\"將圖片拖放到這裡，或點擊上傳\"" in source
    assert "請先上傳一張待檢影像" in (
        root / "deploy/hf_space/runtime.py"
    ).read_text(encoding="utf-8")
    assert "--text-xs: 1.1875rem" in source
    assert "--text-md: 1.25rem" in source
    assert "--df-radius-lg: 4px" in source
    assert "--df-surface-blue: oklch(" in source
    assert "--df-surface-peach: oklch(" in source
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in source
    assert "grid-auto-rows: 80px" in source
    assert "grid-template-columns: 42px 7.5rem" in source
    assert "height: 80px !important" in source
    assert 'elem_id="df-result-overview"' in source
    assert "df-confidence-bars" in source
    assert ".df-decision h3 {" in source
    assert "color: #fff !important" in source
    assert "gr.Markdown(" not in source
    assert "border-radius: 999px" not in source
    assert 'elem_id="df-result-panel"' in source
    assert "id='df-steps'" not in source
    assert "font-size: .72rem" not in source
    assert "border-left: 7px" not in source
    assert "border-left: 4px" not in source
    assert "border-left: 3px" not in source


def test_space_demo_examples_are_complete_and_hash_bound() -> None:
    root = Path(__file__).resolve().parents[1]
    example_root = root / "deploy/hf_space/examples"
    expected = {
        "pcb1_defect_a.JPG": (
            "e67db34e1045d7b8f01173a2f3e0138a6d89700e8a83d1426b6de0464e1e5e9d"
        ),
        "pcb1_defect_b.JPG": (
            "f589eda90ce0b8865e102878d125f999949286339fa9bbc41f1354866976430f"
        ),
        "pcb1_normal.JPG": (
            "c71fc38d4d03498c0827c114e21157b5e71d3da25b549a693ac23a961db68202"
        ),
        "capsules_defect.JPG": (
            "76dd4a2f47089db32cc3c386be0efcdeb48c8b1858512a4272d698ccbb6e9d1c"
        ),
        "capsules_normal.JPG": (
            "c5ffa21822a8759621e81deedfcbd94cfae213667044f995fbaa6d9affbc1a4f"
        ),
    }
    actual = {
        path.name: sha256_file(path)
        for path in example_root.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    assert actual == expected

    source = (root / "deploy/hf_space/app.py").read_text(encoding="utf-8")
    notices = (root / "deploy/hf_space/THIRD_PARTY_NOTICES.md").read_text(
        encoding="utf-8"
    )
    assert "DEMO_EXAMPLES" in source
    assert "gr.Gallery(" in source
    assert "examples.select(" in source
    assert "CC BY 4.0" in source
    assert "examples/README.md" in notices
