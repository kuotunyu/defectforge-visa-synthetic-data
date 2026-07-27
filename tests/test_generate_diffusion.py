from pathlib import Path

import numpy as np

from src.synthetic.generate_diffusion import (
    DiffusionGenerationError,
    blend_patch,
    build_metadata,
    candidate_parameters,
    derived_seed,
    feather_alpha_mask,
    logical_adapter_path,
    rebuild_metadata,
    score_candidate,
    write_json_atomic,
)


def placement_record() -> dict[str, object]:
    return {
        "affine": {
            "dx": 3,
            "dy": -2,
            "flip": False,
            "rotation_deg": 4.5,
            "scale": 1.1,
        },
        "background_image": "pcb1/Data/Images/Normal/0000.JPG",
        "background_sha256": "a" * 64,
        "defect_type": "type0",
        "mask_area_px": 16,
        "mask_area_ratio": 0.25,
        "mask_bbox": [2, 2, 4, 4],
        "object": "pcb1",
        "placement_id": "pcb1__type0__0000__00",
        "roi_bbox": [0, 0, 8, 8],
        "source_component_id": 0,
        "source_image": "pcb1/Data/Images/Anomaly/000.JPG",
        "source_mask": "pcb1/Data/Masks/Anomaly/000.png",
        "trigger_token": "<pcb1-type0>",
    }


def test_derived_seed_and_refine_schedule_are_reproducible() -> None:
    first = derived_seed(42, "pcb1", 7, 0)
    second = derived_seed(42, "pcb1", 7, 0)
    assert first == second
    assert first != derived_seed(42, "pcb1", 7, 1)

    parameters = candidate_parameters(
        seed=42,
        object_name="pcb1",
        placement_index=7,
        guidance_grid=[5.0, 7.5],
        crop_ratio_grid=[1.8, 2.5, 3.5],
        count=4,
        baseline_guidance=7.5,
        baseline_crop_ratio=2.5,
    )
    repeated = candidate_parameters(
        seed=42,
        object_name="pcb1",
        placement_index=7,
        guidance_grid=[5.0, 7.5],
        crop_ratio_grid=[1.8, 2.5, 3.5],
        count=4,
        baseline_guidance=7.5,
        baseline_crop_ratio=2.5,
    )
    assert parameters == repeated
    assert parameters[0] == (7.5, 2.5)
    assert len(set(parameters)) == 4
    assert {guidance for guidance, _ in parameters} == {5.0, 7.5}
    assert {crop for _, crop in parameters} == {1.8, 2.5, 3.5}


def test_feather_blend_changes_mask_and_preserves_distant_pixels() -> None:
    background = np.zeros((16, 16, 3), dtype=np.uint8)
    generated_patch = np.full((8, 8, 3), 200, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[6:10, 6:10] = 255

    alpha = feather_alpha_mask(mask[4:12, 4:12], dilation_px=1, sigma=0.0)
    result = blend_patch(
        background,
        generated_patch,
        mask,
        (4, 4, 12, 12),
        method="feather_alpha",
        dilation_px=1,
        sigma=0.0,
    )

    assert np.all(alpha[2:6, 2:6] == 1.0)
    assert np.all(result[6:10, 6:10] == 200)
    assert np.all(result[:4] == 0)
    assert np.all(result[:, :4] == 0)


def test_build_metadata_uses_full_resolution_crop_coordinates() -> None:
    record = build_metadata(
        placement=placement_record(),
        bucket="original",
        image_path="images/pcb1__type0__0000__00.png",
        mask_path="masks/pcb1__type0__0000__00.png",
        generator_name="stageB_sd2",
        model_id="sd2-community/stable-diffusion-2-inpainting",
        adapter_path="runs/lora_sd2/pcb1/seed_42/final",
        description="a printed circuit board",
        negative_prompt="text",
        generator_seed=123,
        guidance_scale=7.5,
        num_inference_steps=40,
        strength=1.0,
        crop_ratio=2.5,
        crop_bbox=(1, 2, 9, 10),
        resolution=512,
        blend="feather_alpha",
    )

    assert record["sample_id"] == "pcb1__type0__0000__00"
    assert record["generator"] == "stageB_sd2"
    assert record["generation"]["crop_bbox"] == [1, 2, 8, 8]
    assert record["generation"]["seed"] == 123
    assert record["placement"]["mask_bbox"] == [2, 2, 4, 4]


def test_boundary_gradient_score_penalizes_checkerboard_seam() -> None:
    background = np.full((32, 32, 3), 100, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 255
    smooth = background.copy()
    smooth[10:22, 10:22] = 125
    artifact = smooth.copy()
    yy, xx = np.indices((12, 12))
    checker = np.where((yy + xx) % 2 == 0, 0, 255).astype(np.uint8)
    artifact[10:22, 10:22] = checker[:, :, None]
    config = {
        "visible_change_target": 0.12,
        "visible_change_tolerance": 0.10,
        "visible_change_weight": 0.45,
        "seam_weight": 0.45,
        "artifact_weight": 0.10,
    }

    smooth_score = score_candidate(background, smooth, mask, config=config)
    artifact_score = score_candidate(background, artifact, mask, config=config)

    assert smooth_score["seam_score"] > artifact_score["seam_score"]
    assert artifact_score["seam_delta"] > smooth_score["seam_delta"]


def test_rebuild_metadata_sorts_sidecars_by_object_and_index(tmp_path: Path) -> None:
    output_root = tmp_path / "original"
    first = build_metadata(
        placement=placement_record(),
        bucket="original",
        image_path="images/first.png",
        mask_path="masks/first.png",
        generator_name="stageB_sd2",
        model_id="model",
        adapter_path="adapter",
        description="object",
        negative_prompt="",
        generator_seed=1,
        guidance_scale=7.5,
        num_inference_steps=2,
        strength=1.0,
        crop_ratio=2.5,
        crop_bbox=(0, 0, 8, 8),
        resolution=512,
        blend="feather_alpha",
    )
    second_placement = placement_record()
    second_placement["placement_id"] = "pcb1__type0__0001__00"
    second = build_metadata(
        placement=second_placement,
        bucket="original",
        image_path="images/second.png",
        mask_path="masks/second.png",
        generator_name="stageB_sd2",
        model_id="model",
        adapter_path="adapter",
        description="object",
        negative_prompt="",
        generator_seed=2,
        guidance_scale=7.5,
        num_inference_steps=2,
        strength=1.0,
        crop_ratio=2.5,
        crop_bbox=(0, 0, 8, 8),
        resolution=512,
        blend="feather_alpha",
    )
    write_json_atomic(
        output_root / ".records" / "second.json",
        {"placement_index": 1, "record": second},
    )
    write_json_atomic(
        output_root / ".records" / "first.json",
        {"placement_index": 0, "record": first},
    )

    records = rebuild_metadata(output_root)

    assert [record["sample_id"] for record in records] == [
        "pcb1__type0__0000__00",
        "pcb1__type0__0001__00",
    ]


def test_logical_adapter_path_never_leaks_local_absolute_path(tmp_path: Path) -> None:
    class FakePaths:
        runs = tmp_path / "data" / "runs"
        project_root = tmp_path / "project"

    paths = FakePaths()

    assert (
        logical_adapter_path(
            paths,  # type: ignore[arg-type]
            paths.runs / "lora_sd2" / "pcb1" / "seed_42" / "final",
        )
        == "runs/lora_sd2/pcb1/seed_42/final"
    )
    assert (
        logical_adapter_path(
            paths,  # type: ignore[arg-type]
            paths.project_root / "results" / "colab" / "lora_sdxl" / "pcb1" / "final",
        )
        == "results/colab/lora_sdxl/pcb1/final"
    )

    outside = tmp_path / "outside" / "adapter"
    try:
        logical_adapter_path(paths, outside)  # type: ignore[arg-type]
    except DiffusionGenerationError:
        pass
    else:
        raise AssertionError("Expected an out-of-project adapter path to be rejected")
