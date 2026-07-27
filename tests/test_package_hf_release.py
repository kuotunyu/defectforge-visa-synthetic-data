from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

from scripts.package_hf_release import (
    MODEL_PAYLOAD_FILES,
    ReleasePackagingError,
    inventory_dataset_source,
    inventory_model_family,
    materialize,
    verify_release_manifest,
)
from src.common.integrity import sha256_file


def _dataset_source(root: Path, *, absolute_background: bool = False) -> Path:
    source = root / "source"
    (source / "images").mkdir(parents=True)
    (source / "masks").mkdir()
    rows = []
    for index in range(2):
        name = f"sample-{index}.png"
        (source / "images" / name).write_bytes(f"image-{index}".encode())
        (source / "masks" / name).write_bytes(f"mask-{index}".encode())
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "image_path": f"images/{name}",
                "mask_path": f"masks/{name}",
                "source": {
                    "background_image": (
                        "C:" + "/Users/private/image.png"
                        if absolute_background and index == 0
                        else f"pcb1/Data/Images/Normal/{index:04d}.JPG"
                    ),
                    "defect_source_image": None,
                    "defect_source_mask": None,
                },
            }
        )
    (source / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return source


def test_dataset_inventory_is_complete_and_materializable(tmp_path: Path) -> None:
    source = _dataset_source(tmp_path)
    files, summary = inventory_dataset_source(
        source,
        destination_root=PurePosixPath("data/unfiltered"),
        expected_samples=2,
        blocked_sha256=set(),
    )
    assert summary["samples"] == 2
    assert summary["files"] == 5
    assert summary["test_blocklist_hits"] == 0

    destination = tmp_path / "release"
    stats = materialize(files, destination)
    assert stats["hardlinked"] + stats["copied"] == 5
    assert (destination / "data/unfiltered/images/sample-0.png").read_bytes() == b"image-0"


def test_dataset_inventory_rejects_absolute_source_path(tmp_path: Path) -> None:
    source = _dataset_source(tmp_path, absolute_background=True)
    with pytest.raises(ReleasePackagingError, match="Unsafe|Absolute"):
        inventory_dataset_source(
            source,
            destination_root=PurePosixPath("data/unfiltered"),
            expected_samples=2,
            blocked_sha256=set(),
        )


def test_dataset_inventory_rejects_frozen_test_hash(tmp_path: Path) -> None:
    source = _dataset_source(tmp_path)
    blocked = {sha256_file(source / "images" / "sample-0.png")}
    with pytest.raises(ReleasePackagingError, match="blocklist"):
        inventory_dataset_source(
            source,
            destination_root=PurePosixPath("data/unfiltered"),
            expected_samples=2,
            blocked_sha256=blocked,
        )


def test_model_inventory_binds_all_adapter_hashes(tmp_path: Path) -> None:
    roots = {}
    objects = {}
    for object_name in ("pcb1", "capsules"):
        final = tmp_path / object_name / "final"
        roots[object_name] = final
        for relative in MODEL_PAYLOAD_FILES:
            path = final / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        objects[object_name] = {
            "adapter_hashes": {
                "unet_adapter_sha256": sha256_file(
                    final / "unet_adapter/adapter_model.safetensors"
                ),
                "text_token_adapter_sha256": sha256_file(
                    final / "text_token_adapter/adapter_model.safetensors"
                ),
            }
        }
    files, summary = inventory_model_family(
        family="lora_sd2",
        roots=roots,
        validation={"status": "passed", "objects": objects},
    )
    assert len(files) == len(MODEL_PAYLOAD_FILES) * 2
    assert set(summary) == {"pcb1", "capsules"}

    changed = roots["pcb1"] / "unet_adapter/adapter_model.safetensors"
    changed.write_text("changed", encoding="utf-8")
    with pytest.raises(ReleasePackagingError, match="hash changed"):
        inventory_model_family(
            family="lora_sd2",
            roots=roots,
            validation={"status": "passed", "objects": objects},
        )


def test_existing_release_manifest_must_match_current_sources(tmp_path: Path) -> None:
    source = _dataset_source(tmp_path)
    files, _ = inventory_dataset_source(
        source,
        destination_root=PurePosixPath("data/unfiltered"),
        expected_samples=2,
        blocked_sha256=set(),
    )
    release = tmp_path / "release"
    materialize(files, release)
    (release / "release_manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "files": [item.manifest_row() for item in files],
            }
        ),
        encoding="utf-8",
    )
    assert len(verify_release_manifest(release, files)) == 64

    source_image = source / "images/sample-0.png"
    source_image.write_bytes(b"changed")
    changed_files, _ = inventory_dataset_source(
        source,
        destination_root=PurePosixPath("data/unfiltered"),
        expected_samples=2,
        blocked_sha256=set(),
    )
    with pytest.raises(ReleasePackagingError, match="stale"):
        verify_release_manifest(release, changed_files)
