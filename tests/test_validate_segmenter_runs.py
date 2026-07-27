from __future__ import annotations

import pytest

from scripts.validate_segmenter_runs import (
    SegmenterValidationError,
    _validate_data_manifest,
    _validate_training_budget,
)


def _real(image_hash: str, *, defect: bool, mask_hash: str | None = None) -> dict:
    return {
        "sample_id": image_hash,
        "kind": "real",
        "has_defect": defect,
        "image_sha256": image_hash,
        "mask_sha256": mask_hash,
        "manifest_refs": ("source.png",),
    }


def _synthetic(image_hash: str) -> dict:
    return {
        "sample_id": image_hash,
        "kind": "synthetic",
        "has_defect": True,
        "image_sha256": image_hash,
        "mask_sha256": "m" * 64,
        "manifest_refs": ("train.png",),
    }


def test_validate_data_manifest_accepts_zero_real_defect_procedural_group() -> None:
    test_hash = "t" * 64
    test_mask = "u" * 64
    data = {
        "canonical_group": "procedural_only",
        "object": "pcb1",
        "mode": "final",
        "train": [
            _real("n" * 64, defect=False),
            *[_synthetic(f"{index:064x}") for index in range(500)],
        ],
        "validation": [],
        "test": [_real(test_hash, defect=True, mask_hash=test_mask)],
    }
    result = _validate_data_manifest(
        data,
        group_name="procedural_only",
        object_name="pcb1",
        expected_test_images={test_hash},
        expected_test_masks={test_hash: test_mask},
        blocklist={test_hash, test_mask},
    )
    assert result["real_defects"] == 0
    assert result["synthetic_defects"] == 500


def test_validate_data_manifest_rejects_real_defect_in_procedural_group() -> None:
    test_hash = "t" * 64
    data = {
        "canonical_group": "procedural_only",
        "object": "pcb1",
        "mode": "final",
        "train": [
            _real("r" * 64, defect=True, mask_hash="q" * 64),
            *[_synthetic(f"{index:064x}") for index in range(500)],
        ],
        "validation": [],
        "test": [_real(test_hash, defect=False)],
    }
    with pytest.raises(SegmenterValidationError, match="real defect"):
        _validate_data_manifest(
            data,
            group_name="procedural_only",
            object_name="pcb1",
            expected_test_images={test_hash},
            expected_test_masks={test_hash: None},
            blocklist={test_hash},
        )


def test_validate_training_budget_requires_all_formal_steps() -> None:
    config = {"training": {"total_steps": 500}}
    _validate_training_budget(
        {"requested_total_steps": 500, "executed_steps": 500},
        config,
        run_name="complete",
    )
    with pytest.raises(SegmenterValidationError, match="Executed"):
        _validate_training_budget(
            {"requested_total_steps": 500, "executed_steps": 499},
            config,
            run_name="partial",
        )
