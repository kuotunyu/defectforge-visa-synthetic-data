from __future__ import annotations

import copy

import pytest

from scripts.verify_splits import (
    SplitVerificationError,
    validate_manifest,
)


def _fixture() -> tuple[dict, dict, list[dict[str, str]]]:
    rows = [
        {
            "object": "pcb1",
            "split": "train",
            "label": "normal",
            "image": "pcb1/train.png",
            "mask": "",
        },
        {
            "object": "pcb1",
            "split": "test",
            "label": "anomaly",
            "image": "pcb1/test.png",
            "mask": "pcb1/test_mask.png",
        },
    ]
    manifest = {
        "images": [
            {
                "object": "pcb1",
                "group_id": 1,
                "set": "train",
                "split_type": "2cls_highshot",
                "image_path": "pcb1/train.png",
                "sha256": "a" * 64,
                "mask_sha256": None,
            },
            {
                "object": "pcb1",
                "group_id": 2,
                "set": "test",
                "split_type": "2cls_highshot",
                "image_path": "pcb1/test.png",
                "sha256": "b" * 64,
                "mask_sha256": "c" * 64,
            },
        ]
    }
    blocklist = {
        "manifest_sha256": "d" * 64,
        "image_count": 1,
        "mask_count": 1,
        "unique_sha256_count": 2,
        "sha256": ["b" * 64, "c" * 64],
    }
    return manifest, blocklist, rows


def test_manifest_validation_accepts_exact_test_blocklist() -> None:
    manifest, blocklist, rows = _fixture()
    result = validate_manifest(
        manifest=manifest,
        blocklist=blocklist,
        manifest_sha256="d" * 64,
        highshot_rows=rows,
    )
    assert result["manifest_images"] == 2
    assert result["blocked_sha256"] == 2


def test_manifest_validation_rejects_phash_group_crossing_boundary() -> None:
    manifest, blocklist, rows = _fixture()
    changed = copy.deepcopy(manifest)
    changed["images"][1]["group_id"] = 1
    with pytest.raises(SplitVerificationError, match="pHash group"):
        validate_manifest(
            manifest=changed,
            blocklist=blocklist,
            manifest_sha256="d" * 64,
            highshot_rows=rows,
        )


def test_manifest_validation_rejects_missing_mask_hash() -> None:
    manifest, blocklist, rows = _fixture()
    changed = copy.deepcopy(blocklist)
    changed["sha256"] = ["b" * 64]
    changed["unique_sha256_count"] = 1
    with pytest.raises(SplitVerificationError, match="exactly cover"):
        validate_manifest(
            manifest=manifest,
            blocklist=changed,
            manifest_sha256="d" * 64,
            highshot_rows=rows,
        )
