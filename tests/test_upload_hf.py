from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
from torch import zeros

from scripts.upload_hf import HuggingFaceUploadError, inventory, plan_summary
from src.common.integrity import sha256_file


def _write_release_manifest(root: Path) -> None:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "release_manifest.json"
    ]
    (root / "release_manifest.json").write_text(
        json.dumps({"status": "passed", "files": files}),
        encoding="utf-8",
    )


def test_dataset_inventory_requires_synthetic_images_and_masks(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Complete dataset card\n", encoding="utf-8")
    image_dir = tmp_path / "data" / "stageA" / "images"
    mask_dir = tmp_path / "data" / "stageA" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (image_dir / "sample.png").write_bytes(b"image")
    (mask_dir / "sample.png").write_bytes(b"mask")
    _write_release_manifest(tmp_path)
    result = inventory(tmp_path, repo_type="dataset")
    assert result["file_count"] == 4
    assert result["image_count"] == 1
    assert result["mask_count"] == 1
    assert result["release_manifest_verified"] is True


def test_dataset_inventory_rejects_real_data_directory(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Complete dataset card\n", encoding="utf-8")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "real.png").write_bytes(b"image")
    _write_release_manifest(tmp_path)
    with pytest.raises(HuggingFaceUploadError, match="real-data"):
        inventory(tmp_path, repo_type="dataset")


def test_model_inventory_requires_safetensors(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Complete model card\n", encoding="utf-8")
    weights = tmp_path / "lora_sd2" / "pcb1"
    weights.mkdir(parents=True)
    save_file({"weight": zeros(1)}, weights / "adapter_model.safetensors")
    _write_release_manifest(tmp_path)
    result = inventory(tmp_path, repo_type="model")
    assert result["safetensor_count"] == 1


def test_inventory_rejects_unfinished_card(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# TBD\n", encoding="utf-8")
    with pytest.raises(HuggingFaceUploadError, match="unfinished"):
        inventory(tmp_path, repo_type="model")


def test_inventory_rejects_payload_changed_after_packaging(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Complete model card\n", encoding="utf-8")
    weights = tmp_path / "lora_sd2" / "pcb1"
    weights.mkdir(parents=True)
    model = weights / "adapter_model.safetensors"
    save_file({"weight": zeros(1)}, model)
    _write_release_manifest(tmp_path)
    model.write_bytes(model.read_bytes() + b"changed")
    with pytest.raises(HuggingFaceUploadError, match="differs"):
        inventory(tmp_path, repo_type="model")


def test_tracked_plan_omits_redundant_file_rows_and_local_root() -> None:
    summary = plan_summary(
        {
            "repo_id": "owner/repo",
            "root": "D:/local/publish",
            "file_count": 2,
            "files": [{"path": "large", "sha256": "0" * 64}],
            "release_manifest_sha256": "1" * 64,
        }
    )
    assert summary == {
        "repo_id": "owner/repo",
        "file_count": 2,
        "release_manifest_sha256": "1" * 64,
    }
