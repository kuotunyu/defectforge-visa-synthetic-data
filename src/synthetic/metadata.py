"""Synthetic metadata construction and validation."""

from __future__ import annotations

from typing import Any

TOP_LEVEL_FIELDS = {
    "sample_id",
    "object",
    "defect_type",
    "trigger_token",
    "generator",
    "bucket",
    "image_path",
    "mask_path",
    "source",
    "placement",
    "generation",
    "filter",
    "pipeline_version",
    "created_at",
}
SOURCE_FIELDS = {
    "background_image",
    "background_sha256",
    "defect_source_image",
    "defect_source_mask",
    "defect_source_component_id",
}
PLACEMENT_FIELDS = {
    "roi_bbox",
    "mask_bbox",
    "affine",
    "mask_area_px",
    "mask_area_ratio",
}
GENERATION_FIELDS = {
    "seed",
    "base_model",
    "lora_path",
    "prompt",
    "negative_prompt",
    "guidance_scale",
    "num_inference_steps",
    "strength",
    "crop_ratio",
    "crop_bbox",
    "model_resolution",
    "blend",
}
AFFINE_FIELDS = {"dx", "dy", "rotation_deg", "scale", "flip"}


class MetadataError(ValueError):
    """A synthetic provenance record does not satisfy the locked schema."""


def _require_exact_fields(
    value: dict[str, Any],
    fields: set[str],
    *,
    location: str,
) -> None:
    observed = set(value)
    if observed != fields:
        raise MetadataError(
            f"{location} fields differ: missing={sorted(fields - observed)}, "
            f"extra={sorted(observed - fields)}"
        )


def validate_metadata(record: dict[str, Any]) -> None:
    """Validate required Stage A/Stage B provenance fields."""

    _require_exact_fields(record, TOP_LEVEL_FIELDS, location="record")
    _require_exact_fields(record["source"], SOURCE_FIELDS, location="source")
    _require_exact_fields(record["placement"], PLACEMENT_FIELDS, location="placement")
    _require_exact_fields(
        record["placement"]["affine"],
        AFFINE_FIELDS,
        location="placement.affine",
    )
    _require_exact_fields(
        record["generation"],
        GENERATION_FIELDS,
        location="generation",
    )
    if record["object"] not in {"pcb1", "capsules"}:
        raise MetadataError(f"Unexpected object: {record['object']}")
    if record["generator"] not in {
        "stageA_copypaste",
        "stageA_procedural",
        "stageB_sd2",
        "stageB_sdxl",
    }:
        raise MetadataError(f"Unexpected generator: {record['generator']}")
    if not record["sample_id"] or not record["image_path"] or not record["mask_path"]:
        raise MetadataError("sample_id/image_path/mask_path must be non-empty")
    if record["placement"]["mask_area_px"] <= 0:
        raise MetadataError("mask_area_px must be positive")
    if not 0 < record["placement"]["mask_area_ratio"] <= 1:
        raise MetadataError("mask_area_ratio must be in (0, 1]")
