from __future__ import annotations

from pathlib import Path

import pytest
from safetensors.torch import save_file
from torch import zeros

from scripts.upload_hf import HuggingFaceUploadError, inventory


def test_dataset_inventory_requires_synthetic_images_and_masks(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Complete dataset card\n", encoding="utf-8")
    image_dir = tmp_path / "data" / "stageA" / "images"
    mask_dir = tmp_path / "data" / "stageA" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (image_dir / "sample.png").write_bytes(b"image")
    (mask_dir / "sample.png").write_bytes(b"mask")
    result = inventory(tmp_path, repo_type="dataset")
    assert result["file_count"] == 3
    assert result["image_count"] == 2
    assert result["mask_count"] == 1


def test_dataset_inventory_rejects_real_data_directory(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Complete dataset card\n", encoding="utf-8")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "real.png").write_bytes(b"image")
    with pytest.raises(HuggingFaceUploadError, match="real-data"):
        inventory(tmp_path, repo_type="dataset")


def test_model_inventory_requires_safetensors(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Complete model card\n", encoding="utf-8")
    weights = tmp_path / "lora_sd2" / "pcb1"
    weights.mkdir(parents=True)
    save_file({"weight": zeros(1)}, weights / "adapter_model.safetensors")
    result = inventory(tmp_path, repo_type="model")
    assert result["safetensor_count"] == 1


def test_inventory_rejects_unfinished_card(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# TBD\n", encoding="utf-8")
    with pytest.raises(HuggingFaceUploadError, match="unfinished"):
        inventory(tmp_path, repo_type="model")
