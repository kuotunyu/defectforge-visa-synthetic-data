from pathlib import Path

import numpy as np
import pytest

from scripts.validate_diffusion import (
    DiffusionValidationError,
    compare_record_to_placement,
    expected_blend_support,
    expected_candidate_seed,
    read_records,
    validate_candidate_schedule,
    validate_search_baseline,
)
from src.synthetic.generate_diffusion import build_metadata


def placement_record() -> dict[str, object]:
    return {
        "affine": {
            "dx": 1,
            "dy": 2,
            "flip": True,
            "rotation_deg": 3.5,
            "scale": 0.9,
        },
        "background_image": "pcb1/Data/Images/Normal/0000.JPG",
        "background_sha256": "a" * 64,
        "defect_type": "type0",
        "mask_area_px": 16,
        "mask_area_ratio": 0.015625,
        "mask_bbox": [6, 6, 4, 4],
        "object": "pcb1",
        "placement_id": "pcb1__type0__0000__00",
        "roi_bbox": [0, 0, 16, 16],
        "source_component_id": 0,
        "source_image": "pcb1/Data/Images/Anomaly/000.JPG",
        "source_mask": "pcb1/Data/Masks/Anomaly/000.png",
        "trigger_token": "<pcb1-type0>",
    }


def metadata_record(placement: dict[str, object]) -> dict[str, object]:
    return build_metadata(
        placement=placement,
        bucket="original",
        image_path="images/pcb1__type0__0000__00.png",
        mask_path="masks/pcb1__type0__0000__00.png",
        generator_name="stageB_sd2",
        model_id="model",
        adapter_path="adapter",
        description="object",
        negative_prompt="",
        generator_seed=1,
        guidance_scale=7.5,
        num_inference_steps=40,
        strength=1.0,
        crop_ratio=2.5,
        crop_bbox=(4, 4, 12, 12),
        resolution=512,
        blend="feather_alpha",
    )


def test_compare_record_to_placement_detects_source_mutation() -> None:
    placement = placement_record()
    record = metadata_record(placement)
    compare_record_to_placement(record, placement)

    record["source"]["background_sha256"] = "b" * 64
    with pytest.raises(DiffusionValidationError, match="source provenance mismatch"):
        compare_record_to_placement(record, placement)


def test_expected_blend_support_never_escapes_declared_crop() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 255

    support = expected_blend_support(
        mask,
        [5, 5, 10, 10],
        dilation_px=2,
        sigma=3.0,
    )

    assert support.any()
    assert not support[:5].any()
    assert not support[15:].any()
    assert not support[:, :5].any()
    assert not support[:, 15:].any()


def test_read_records_rejects_invalid_metadata(tmp_path: Path) -> None:
    path = tmp_path / "metadata.jsonl"
    path.write_text('{"sample_id": "incomplete"}\n', encoding="utf-8")

    with pytest.raises(DiffusionValidationError, match="fields differ"):
        read_records(path)


def test_candidate_schedule_requires_baseline_and_grid_coverage() -> None:
    config = {
        "generation": {"guidance_scale": 7.5, "crop_ratio": 2.5},
        "refine": {
            "guidance_grid": [5.0, 7.5, 10.0, 12.5],
            "crop_ratio_grid": [1.8, 2.5, 3.5],
        },
    }
    parameters = [(7.5, 2.5), (5.0, 1.8), (10.0, 3.5), (12.5, 2.5)]
    candidates = [
        {
            "candidate_index": index,
            "generator_seed": expected_candidate_seed(42, "pcb1", 3, index),
            "guidance_scale": guidance,
            "crop_ratio": crop,
        }
        for index, (guidance, crop) in enumerate(parameters)
    ]
    validate_candidate_schedule(
        candidates,
        bucket="searched",
        config=config,
        object_name="pcb1",
        placement_index=3,
        seed=42,
        sample_id="sample",
    )

    candidates[0]["guidance_scale"] = 5.0
    with pytest.raises(DiffusionValidationError, match="original baseline"):
        validate_candidate_schedule(
            candidates,
            bucket="searched",
            config=config,
            object_name="pcb1",
            placement_index=3,
            seed=42,
            sample_id="sample",
        )


def test_search_baseline_requires_evidence_and_selected_image_identity() -> None:
    baseline = {"candidate_index": 0, "score": 0.5}
    original = {
        "candidates": [baseline],
        "image_sha256": "a" * 64,
        "selected_candidate_index": 0,
    }
    searched = {
        "candidates": [baseline, {"candidate_index": 1, "score": 0.7}],
        "image_sha256": "b" * 64,
        "selected_candidate_index": 1,
    }
    validate_search_baseline(
        searched_sidecar=searched,
        original_sidecar=original,
        sample_id="sample",
    )

    searched["selected_candidate_index"] = 0
    with pytest.raises(DiffusionValidationError, match="baseline image differs"):
        validate_search_baseline(
            searched_sidecar=searched,
            original_sidecar=original,
            sample_id="sample",
        )
