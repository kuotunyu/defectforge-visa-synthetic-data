"""Generate full-resolution Stage B defects with a frozen inpainting LoRA."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shutil
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from diffusers import AutoencoderKL, StableDiffusionInpaintPipeline, UNet2DConditionModel
from peft import PeftModel
from PIL import Image, ImageDraw
from transformers import CLIPTextModel, CLIPTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import (  # isort: skip
    assert_not_blocklisted,
    load_json,
    read_checksum_file,
    sha256_file,
    verify_frozen_manifest,
)
from src.common.paths import Paths, load_paths  # isort: skip
from src.synthetic.copy_paste import object_code  # isort: skip
from src.synthetic.metadata import validate_metadata  # isort: skip
from src.training.train_inpaint_lora import (  # isort: skip
    square_context_bbox,
    verify_remote_model,
)

LOGGER = logging.getLogger("generate_diffusion")
PIPELINE_VERSIONS = {
    "original": "0.5.0",
    "searched": "0.6.0",
}
UINT64_MAX = np.iinfo(np.uint64).max


class DiffusionGenerationError(RuntimeError):
    """A Stage B generation invariant failed."""


def pipeline_version(bucket: str) -> str:
    try:
        return PIPELINE_VERSIONS[bucket]
    except KeyError as error:
        raise DiffusionGenerationError(f"Unsupported pipeline bucket: {bucket}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument("--config", default="configs/generate_sd2.yaml", type=Path)
    parser.add_argument("--object", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--defect-type", action="append", dest="defect_types")
    parser.add_argument("--n", type=int)
    parser.add_argument("--bucket", choices=("original", "searched"), default="original")
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--crop-ratio", type=float)
    parser.add_argument("--blend", choices=("feather_alpha", "poisson"))
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--num-search-run", type=int)
    parser.add_argument("--guidance-grid")
    parser.add_argument("--crop-ratio-grid")
    parser.add_argument("--out-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiffusionGenerationError(f"Expected YAML mapping: {path}")
    for key in ("model", "objects", "generation", "refine", "output"):
        if not isinstance(value.get(key), dict):
            raise DiffusionGenerationError(f"Missing config mapping: {key}")
    return value


def parse_float_grid(value: str | None, default: Sequence[float]) -> list[float]:
    if value is None:
        result = [float(item) for item in default]
    else:
        try:
            result = [float(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError as error:
            raise DiffusionGenerationError(f"Invalid float grid: {value!r}") from error
    if not result or any(not math.isfinite(item) or item <= 0 for item in result):
        raise DiffusionGenerationError("Search grids must contain positive finite values")
    return result


def derived_seed(
    seed: int,
    object_name: str,
    placement_index: int,
    candidate_index: int,
) -> int:
    value = np.random.SeedSequence(
        [
            seed,
            object_code(f"{object_name}:stageB-diffusion"),
            placement_index,
            candidate_index,
        ]
    ).generate_state(1, dtype=np.uint64)[0]
    return int(value)


def candidate_parameters(
    *,
    seed: int,
    object_name: str,
    placement_index: int,
    guidance_grid: Sequence[float],
    crop_ratio_grid: Sequence[float],
    count: int,
    baseline_guidance: float,
    baseline_crop_ratio: float,
) -> list[tuple[float, float]]:
    maximum = len(guidance_grid) * len(crop_ratio_grid)
    if count < 1 or count > maximum:
        raise DiffusionGenerationError(
            f"num_search_run must be in [1, {maximum}], observed {count}"
        )
    baseline = (float(baseline_guidance), float(baseline_crop_ratio))
    all_pairs = [
        (float(guidance), float(crop_ratio))
        for guidance in guidance_grid
        for crop_ratio in crop_ratio_grid
    ]
    if baseline not in all_pairs:
        raise DiffusionGenerationError(
            "The original guidance_scale and crop_ratio must be present in the refine grids"
        )
    rng = np.random.Generator(
        np.random.PCG64(
            derived_seed(
                seed,
                object_name,
                placement_index,
                candidate_index=maximum + 1,
            )
        )
    )
    shuffled = [
        all_pairs[int(index)]
        for index in rng.permutation(len(all_pairs))
        if all_pairs[int(index)] != baseline
    ]
    result = [baseline]
    seen_guidance = {baseline[0]}
    seen_crop_ratio = {baseline[1]}
    while len(result) < count:
        best_index = max(
            range(len(shuffled)),
            key=lambda index: (
                int(shuffled[index][0] not in seen_guidance)
                + int(shuffled[index][1] not in seen_crop_ratio),
                -index,
            ),
        )
        pair = shuffled.pop(best_index)
        result.append(pair)
        seen_guidance.add(pair[0])
        seen_crop_ratio.add(pair[1])
    return result


def read_placements(
    *,
    paths: Paths,
    object_name: str,
    expected_sha256: str,
    n: int,
    defect_types: Sequence[str] | None,
) -> tuple[list[dict[str, Any]], Path]:
    metadata_path = paths.synthetic / "placements" / object_name / "placements.jsonl"
    if sha256_file(metadata_path) != expected_sha256:
        raise DiffusionGenerationError(f"Frozen placements checksum mismatch: {object_name}")
    allowed = set(defect_types or ())
    records: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DiffusionGenerationError(
                    f"Placement is not an object: {metadata_path}:{line_number}"
                )
            if value.get("object") != object_name:
                raise DiffusionGenerationError(f"Placement object mismatch at line {line_number}")
            if allowed and str(value.get("defect_type")) not in allowed:
                continue
            records.append(value)
            if len(records) == n:
                break
    if len(records) != n:
        raise DiffusionGenerationError(
            f"Requested {n} placements for {object_name}, found {len(records)}"
        )
    if len({str(record["placement_id"]) for record in records}) != len(records):
        raise DiffusionGenerationError("Duplicate placement_id in selected records")
    return records, metadata_path


def verify_frozen_inputs(
    *,
    paths: Paths,
    placements: Sequence[dict[str, Any]],
    placement_metadata_path: Path,
    expected_placement_sha256: str,
) -> dict[str, str]:
    _, manifest_sha256 = verify_frozen_manifest(paths.splits)
    selection_sha256, selection_filename = read_checksum_file(
        paths.splits / "FEWSHOT_SELECTION.sha256"
    )
    selection_path = paths.splits / selection_filename
    if sha256_file(selection_path) != selection_sha256:
        raise DiffusionGenerationError("Frozen few-shot selection checksum mismatch")
    defect_types_sha256, defect_types_filename = read_checksum_file(
        paths.splits / "DEFECT_TYPES.sha256"
    )
    defect_types_path = paths.splits / defect_types_filename
    if sha256_file(defect_types_path) != defect_types_sha256:
        raise DiffusionGenerationError("Frozen defect-types checksum mismatch")
    placement_validation = load_json(paths.reports / "placements_validation.json")
    if (
        placement_validation.get("status") != "passed"
        or placement_validation.get("manifest_sha256") != manifest_sha256
        or placement_validation.get("defect_types_sha256") != defect_types_sha256
        or int(placement_validation.get("test_blocklist_hits", -1)) != 0
    ):
        raise DiffusionGenerationError("M9 independent validation is absent or incompatible")
    if sha256_file(placement_metadata_path) != expected_placement_sha256:
        raise DiffusionGenerationError("Placement metadata changed during preflight")

    source_paths = sorted(
        {
            paths.visa_raw / str(record[key])
            for record in placements
            for key in ("background_image", "source_image", "source_mask")
        }
    )
    placement_masks = sorted(
        {
            paths.synthetic / "placements" / str(record["object"]) / str(record["mask_path"])
            for record in placements
        }
    )
    assert_not_blocklisted(
        [*source_paths, *placement_masks],
        paths.splits / "test_blocklist.json",
    )
    for record in placements:
        background = paths.visa_raw / str(record["background_image"])
        if sha256_file(background) != str(record["background_sha256"]):
            raise DiffusionGenerationError(
                f"Frozen background changed: {record['background_image']}"
            )
    return {
        "manifest_sha256": manifest_sha256,
        "selection_sha256": selection_sha256,
        "defect_types_sha256": defect_types_sha256,
        "placements_sha256": expected_placement_sha256,
    }


def resolve_adapter(
    *,
    paths: Paths,
    object_config: dict[str, Any],
    override: Path | None,
) -> Path:
    adapter = (
        override.resolve(strict=False)
        if override is not None
        else (paths.runs / str(object_config["adapter"])).resolve(strict=False)
    )
    required = (
        "unet_adapter/adapter_config.json",
        "unet_adapter/adapter_model.safetensors",
        "text_token_adapter/adapter_config.json",
        "text_token_adapter/adapter_model.safetensors",
        "tokenizer/tokenizer.json",
        "trainer_state.json",
    )
    for relative in required:
        if not (adapter / relative).is_file():
            raise DiffusionGenerationError(f"Missing adapter bundle file: {adapter / relative}")
    observed = {
        "unet_adapter_sha256": sha256_file(
            adapter / "unet_adapter" / "adapter_model.safetensors"
        ),
        "text_token_adapter_sha256": sha256_file(
            adapter / "text_token_adapter" / "adapter_model.safetensors"
        ),
    }
    for key, digest in observed.items():
        if digest != str(object_config[key]):
            raise DiffusionGenerationError(f"Frozen {key} mismatch: {adapter}")
    report = load_json(adapter.parent / "training_report.json")
    if report.get("status") != "passed" or int(report.get("global_step", -1)) != 400:
        raise DiffusionGenerationError(f"Adapter training report is not a passed M10 run: {adapter}")
    return adapter


def load_pipeline(
    *,
    config: dict[str, Any],
    adapter: Path,
    device: torch.device,
) -> StableDiffusionInpaintPipeline:
    if str(config["model"]["family"]) != "sd2":
        raise DiffusionGenerationError("This implementation currently accepts SD2 bundles only")
    model_id = str(config["model"]["id"])
    revision = str(config["model"]["revision"])
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    common: dict[str, Any] = {
        "revision": revision,
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    tokenizer = CLIPTokenizer.from_pretrained(adapter / "tokenizer")
    base_unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        **common,
    )
    base_text = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        **common,
    )
    base_text.resize_token_embeddings(len(tokenizer))
    unet = PeftModel.from_pretrained(
        base_unet,
        str(adapter / "unet_adapter"),
    )
    text_encoder = PeftModel.from_pretrained(
        base_text,
        str(adapter / "text_token_adapter"),
    )
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        **common,
    )
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        use_safetensors=True,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(device)
    return pipeline


def feather_alpha_mask(
    mask: np.ndarray,
    *,
    dilation_px: int,
    sigma: float,
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        raise DiffusionGenerationError("Cannot blend an empty mask")
    if dilation_px < 0 or sigma < 0:
        raise DiffusionGenerationError("Blend dilation and sigma must be non-negative")
    if dilation_px:
        side = dilation_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
        binary = cv2.dilate(binary, kernel)
    alpha = binary.astype(np.float32)
    if sigma:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(alpha, 0.0, 1.0)


def blend_patch(
    background: np.ndarray,
    generated_patch: np.ndarray,
    mask: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    *,
    method: str,
    dilation_px: int,
    sigma: float,
) -> np.ndarray:
    x0, y0, x1, y1 = crop_bbox
    if generated_patch.shape[:2] != (y1 - y0, x1 - x0):
        raise DiffusionGenerationError("Generated patch dimensions do not match crop bbox")
    if mask.shape != background.shape[:2]:
        raise DiffusionGenerationError("Mask and background dimensions differ")
    crop_mask = mask[y0:y1, x0:x1]
    if method == "poisson":
        binary = (crop_mask > 0).astype(np.uint8) * 255
        if dilation_px:
            side = dilation_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
            binary = cv2.dilate(binary, kernel)
        center = ((x0 + x1) // 2, (y0 + y1) // 2)
        try:
            return cv2.seamlessClone(
                cv2.cvtColor(generated_patch, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(background, cv2.COLOR_RGB2BGR),
                binary,
                center,
                cv2.NORMAL_CLONE,
            )[:, :, ::-1]
        except cv2.error:
            LOGGER.warning("Poisson blend failed; falling back to feather_alpha")
    alpha = feather_alpha_mask(crop_mask, dilation_px=dilation_px, sigma=sigma)
    destination = background.copy()
    background_crop = background[y0:y1, x0:x1].astype(np.float32)
    blended = (
        alpha[:, :, None] * generated_patch.astype(np.float32)
        + (1.0 - alpha[:, :, None]) * background_crop
    )
    destination[y0:y1, x0:x1] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return destination


def score_candidate(
    background: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    *,
    config: dict[str, Any],
) -> dict[str, float]:
    mask_bool = mask > 0
    difference = np.abs(generated.astype(np.float32) - background.astype(np.float32)) / 255.0
    visible_change = float(difference[mask_bool].mean())
    target = float(config["visible_change_target"])
    tolerance = float(config["visible_change_tolerance"])
    visible_score = max(0.0, 1.0 - abs(visible_change - target) / tolerance)

    binary = mask_bool.astype(np.uint8)
    outer = cv2.dilate(binary, np.ones((5, 5), np.uint8)) > 0
    inner = cv2.erode(binary, np.ones((5, 5), np.uint8)) > 0
    boundary = outer & ~inner
    background_gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY).astype(np.float32)
    generated_gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY).astype(np.float32)
    background_gradient = cv2.magnitude(
        cv2.Sobel(background_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(background_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    generated_gradient = cv2.magnitude(
        cv2.Sobel(generated_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(generated_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    excess_gradient = np.maximum(generated_gradient - background_gradient, 0.0) / 1020.0
    seam_delta = float(excess_gradient[boundary].mean()) if np.any(boundary) else 0.0
    seam_score = max(0.0, 1.0 - seam_delta / 0.12)

    generated_masked = generated[mask_bool]
    clipped_fraction = float(
        np.mean(np.any((generated_masked <= 2) | (generated_masked >= 253), axis=1))
    )
    artifact_score = max(0.0, 1.0 - clipped_fraction * 4.0)
    score = (
        visible_score * float(config["visible_change_weight"])
        + seam_score * float(config["seam_weight"])
        + artifact_score * float(config["artifact_weight"])
    )
    return {
        "score": score,
        "visible_change": visible_change,
        "visible_score": visible_score,
        "seam_delta": seam_delta,
        "seam_score": seam_score,
        "artifact_score": artifact_score,
    }


def render_candidate(
    *,
    pipeline: StableDiffusionInpaintPipeline,
    background_image: Image.Image,
    mask_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    generator_seed: int,
    guidance_scale: float,
    crop_ratio: float,
    num_inference_steps: int,
    strength: float,
    resolution: int,
    blend: str,
    blend_dilation_px: int,
    blend_feather_sigma: float,
    device: torch.device,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    background = np.asarray(background_image.convert("RGB"))
    mask = np.asarray(mask_image.convert("L"))
    crop_bbox = square_context_bbox(mask > 0, crop_ratio=crop_ratio)
    image_crop = background_image.crop(crop_bbox).resize(
        (resolution, resolution),
        resample=Image.Resampling.BICUBIC,
    )
    mask_crop = mask_image.crop(crop_bbox).resize(
        (resolution, resolution),
        resample=Image.Resampling.NEAREST,
    )
    vae_dtype = next(pipeline.vae.parameters()).dtype

    def cast_sampling_latents(
        _pipeline: StableDiffusionInpaintPipeline,
        _step_index: int,
        _timestep: torch.Tensor,
        callback_kwargs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        callback_kwargs["latents"] = callback_kwargs["latents"].to(dtype=vae_dtype)
        return callback_kwargs

    generator = torch.Generator(device=device).manual_seed(generator_seed)
    with torch.inference_mode():
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_crop,
            mask_image=mask_crop,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=generator,
            callback_on_step_end=cast_sampling_latents,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
    patch_size = (crop_bbox[2] - crop_bbox[0], crop_bbox[3] - crop_bbox[1])
    generated_patch = np.asarray(
        result.resize(patch_size, resample=Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    blended = blend_patch(
        background,
        generated_patch,
        mask,
        crop_bbox,
        method=blend,
        dilation_px=blend_dilation_px,
        sigma=blend_feather_sigma,
    )
    return blended, crop_bbox


def logical_adapter_path(paths: Paths, adapter: Path) -> str:
    try:
        return (Path("runs") / adapter.relative_to(paths.runs)).as_posix()
    except ValueError:
        return adapter.as_posix()


def build_metadata(
    *,
    placement: dict[str, Any],
    bucket: str,
    image_path: str,
    mask_path: str,
    model_id: str,
    adapter_path: str,
    description: str,
    negative_prompt: str,
    generator_seed: int,
    guidance_scale: float,
    num_inference_steps: int,
    strength: float,
    crop_ratio: float,
    crop_bbox: tuple[int, int, int, int],
    resolution: int,
    blend: str,
) -> dict[str, Any]:
    prompt = f"a photo of {placement['trigger_token']} defect on {description}"
    x0, y0, x1, y1 = crop_bbox
    record = {
        "bucket": bucket,
        "created_at": datetime.now(UTC).isoformat(),
        "defect_type": str(placement["defect_type"]),
        "filter": None,
        "generation": {
            "base_model": model_id,
            "blend": blend,
            "crop_bbox": [x0, y0, x1 - x0, y1 - y0],
            "crop_ratio": crop_ratio,
            "guidance_scale": guidance_scale,
            "lora_path": adapter_path,
            "model_resolution": resolution,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "prompt": prompt,
            "seed": generator_seed,
            "strength": strength,
        },
        "generator": "stageB_sd2",
        "image_path": image_path,
        "mask_path": mask_path,
        "object": str(placement["object"]),
        "pipeline_version": pipeline_version(bucket),
        "placement": {
            "affine": {
                key: placement["affine"][key]
                for key in ("dx", "dy", "rotation_deg", "scale", "flip")
            },
            "mask_area_px": int(placement["mask_area_px"]),
            "mask_area_ratio": float(placement["mask_area_ratio"]),
            "mask_bbox": [int(value) for value in placement["mask_bbox"]],
            "roi_bbox": [int(value) for value in placement["roi_bbox"]],
        },
        "sample_id": str(placement["placement_id"]),
        "source": {
            "background_image": str(placement["background_image"]),
            "background_sha256": str(placement["background_sha256"]),
            "defect_source_component_id": int(placement["source_component_id"]),
            "defect_source_image": str(placement["source_image"]),
            "defect_source_mask": str(placement["source_mask"]),
        },
        "trigger_token": str(placement["trigger_token"]),
    }
    validate_metadata(record)
    return record


def write_png_atomic(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        image.save(handle, format="PNG", compress_level=6)
    temporary.replace(path)


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_completed_sidecar(
    path: Path,
    *,
    output_root: Path,
    expected: dict[str, Any],
    blocklist: set[str],
) -> dict[str, Any]:
    sidecar = load_json(path)
    record = sidecar.get("record")
    if not isinstance(record, dict):
        raise DiffusionGenerationError(f"Missing record in sidecar: {path}")
    validate_metadata(record)
    for key, value in expected.items():
        if sidecar.get(key) != value:
            raise DiffusionGenerationError(f"Resume signature mismatch for {path.name}: {key}")
    image_path = output_root / str(record["image_path"])
    mask_path = output_root / str(record["mask_path"])
    if (
        not image_path.is_file()
        or not mask_path.is_file()
        or sha256_file(image_path) != sidecar.get("image_sha256")
        or sha256_file(mask_path) != sidecar.get("mask_sha256")
    ):
        raise DiffusionGenerationError(f"Resume output checksum mismatch: {record['sample_id']}")
    if sidecar["image_sha256"] in blocklist or sidecar["mask_sha256"] in blocklist:
        raise DiffusionGenerationError(f"Resume output hit test blocklist: {record['sample_id']}")
    return sidecar


def rebuild_metadata(output_root: Path) -> list[dict[str, Any]]:
    sidecars = []
    for path in (output_root / ".records").glob("*.json"):
        sidecar = load_json(path)
        if not isinstance(sidecar.get("record"), dict):
            raise DiffusionGenerationError(f"Malformed record sidecar: {path}")
        sidecars.append(sidecar)
    sidecars.sort(
        key=lambda value: (
            str(value["record"]["object"]),
            int(value["placement_index"]),
            str(value["record"]["sample_id"]),
        )
    )
    records = [value["record"] for value in sidecars]
    rendered = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    metadata_path = output_root / "metadata.jsonl"
    temporary = metadata_path.with_suffix(".jsonl.tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    temporary.replace(metadata_path)
    return records


def create_contact_sheet(
    destination: Path,
    *,
    paths: Paths,
    output_root: Path,
    records: Sequence[dict[str, Any]],
    maximum: int = 12,
) -> None:
    selected = list(records[:maximum])
    if not selected:
        raise DiffusionGenerationError("Cannot render an empty contact sheet")
    panel = 256
    caption = 36
    canvas = Image.new("RGB", (panel * 3, (panel + caption) * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, record in enumerate(selected):
        with Image.open(paths.visa_raw / record["source"]["background_image"]) as handle:
            background = handle.convert("RGB")
        with Image.open(output_root / record["mask_path"]) as handle:
            mask = handle.convert("RGB")
        with Image.open(output_root / record["image_path"]) as handle:
            generated = handle.convert("RGB")
        x, y, width, height = (
            int(value) for value in record["generation"]["crop_bbox"]
        )
        crop = (x, y, x + width, y + height)
        background = background.crop(crop)
        mask = mask.crop(crop)
        generated = generated.crop(crop)
        for column, image in enumerate((background, mask, generated)):
            thumbnail = image.copy()
            thumbnail.thumbnail((panel, panel), Image.Resampling.LANCZOS)
            x = column * panel + (panel - thumbnail.width) // 2
            y = row * (panel + caption) + (panel - thumbnail.height) // 2
            canvas.paste(thumbnail, (x, y))
        label = (
            f"{record['sample_id']}  g={record['generation']['guidance_scale']} "
            f"crop={record['generation']['crop_ratio']}"
        )
        draw.text((4, row * (panel + caption) + panel + 7), label, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_png_atomic(destination, canvas)


def output_inventory_for_object(
    output_root: Path,
    object_name: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in (output_root / ".records").glob(f"{object_name}__*.json"):
        value = load_json(path)
        if isinstance(value.get("record"), dict):
            records.append(value["record"])
    records.sort(key=lambda record: str(record["sample_id"]))
    return records


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    started = time.perf_counter()
    pipeline: StableDiffusionInpaintPipeline | None = None
    try:
        paths = load_paths(args.paths)
        config = load_config(args.config)
        if args.object not in paths.objects or args.object not in config["objects"]:
            raise DiffusionGenerationError(f"Unsupported object: {args.object}")
        if args.n is not None and args.n < 1:
            raise DiffusionGenerationError("--n must be positive")
        if args.refine and args.bucket != "original":
            raise DiffusionGenerationError("Use --refine without also setting --bucket")
        if not args.refine and args.bucket == "searched":
            raise DiffusionGenerationError("The searched bucket requires --refine")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise DiffusionGenerationError("CUDA was requested but is unavailable")

        object_config = config["objects"][args.object]
        generation_config = config["generation"]
        refine_config = config["refine"]
        seed = paths.seed if args.seed is None else args.seed
        if seed < 0 or seed > UINT64_MAX:
            raise DiffusionGenerationError("--seed must fit uint64")
        n = int(args.n if args.n is not None else generation_config["n"])
        bucket = "searched" if args.refine else args.bucket
        out_name = args.out_name or str(config["output"]["name"])
        output_root = paths.synthetic / out_name / bucket
        placements, placement_metadata_path = read_placements(
            paths=paths,
            object_name=args.object,
            expected_sha256=str(object_config["placements_sha256"]),
            n=n,
            defect_types=args.defect_types,
        )
        frozen_sha256 = verify_frozen_inputs(
            paths=paths,
            placements=placements,
            placement_metadata_path=placement_metadata_path,
            expected_placement_sha256=str(object_config["placements_sha256"]),
        )
        adapter = resolve_adapter(
            paths=paths,
            object_config=object_config,
            override=args.lora,
        )
        model_id = str(config["model"]["id"])
        revision = str(config["model"]["revision"])
        verify_remote_model(config, model_id, revision)

        guidance_scale = float(
            args.guidance_scale
            if args.guidance_scale is not None
            else generation_config["guidance_scale"]
        )
        crop_ratio = float(
            args.crop_ratio if args.crop_ratio is not None else generation_config["crop_ratio"]
        )
        num_inference_steps = int(
            args.num_inference_steps
            if args.num_inference_steps is not None
            else generation_config["num_inference_steps"]
        )
        blend = str(args.blend or generation_config["blend"])
        strength = float(generation_config["strength"])
        negative_prompt = str(generation_config["negative_prompt"])
        resolution = int(config["model"]["resolution"])
        if guidance_scale <= 0 or crop_ratio <= 0 or num_inference_steps < 1:
            raise DiffusionGenerationError("Generation parameters must be positive")

        guidance_grid = parse_float_grid(args.guidance_grid, refine_config["guidance_grid"])
        crop_ratio_grid = parse_float_grid(
            args.crop_ratio_grid,
            refine_config["crop_ratio_grid"],
        )
        num_search_run = int(
            args.num_search_run
            if args.num_search_run is not None
            else refine_config["num_search_run"]
        )
        summary = {
            "adapter": str(adapter),
            "bucket": bucket,
            "config_sha256": sha256_file(args.config),
            "device": args.device,
            "frozen_sha256": frozen_sha256,
            "model": model_id,
            "n": n,
            "object": args.object,
            "output": str(output_root),
            "refine": bool(args.refine),
            "revision": revision,
            "seed": seed,
        }
        LOGGER.info("Preflight passed: %s", json.dumps(summary, sort_keys=True))
        if args.dry_run:
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        record_dir = output_root / ".records"
        existing_for_object = list(record_dir.glob(f"{args.object}__*.json"))
        if existing_for_object and not args.resume:
            raise DiffusionGenerationError(
                f"Output records already exist; use --resume: {output_root}"
            )
        record_dir.mkdir(parents=True, exist_ok=True)
        (output_root / "images").mkdir(parents=True, exist_ok=True)
        (output_root / "masks").mkdir(parents=True, exist_ok=True)
        blocklist = set(load_json(paths.splits / "test_blocklist.json")["sha256"])
        device = torch.device(args.device)

        if device.type == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.reset_peak_memory_stats(device)
        prompt = (
            f"a photo of {{token}} defect on {object_config['description']}"
        )
        logical_lora = logical_adapter_path(paths, adapter)
        generated = 0
        skipped = 0

        for placement_index, placement in enumerate(placements):
            sample_id = str(placement["placement_id"])
            image_relative = f"images/{sample_id}.png"
            mask_relative = f"masks/{sample_id}.png"
            image_path = output_root / image_relative
            mask_path = output_root / mask_relative
            sidecar_path = record_dir / f"{sample_id}.json"
            effective_parameters = {
                "blend": blend,
                "blend_dilation_px": int(generation_config["blend_dilation_px"]),
                "blend_feather_sigma": float(generation_config["blend_feather_sigma"]),
                "crop_ratio": crop_ratio,
                "crop_ratio_grid": crop_ratio_grid if args.refine else None,
                "guidance_grid": guidance_grid if args.refine else None,
                "guidance_scale": guidance_scale,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "num_search_run": num_search_run if args.refine else 1,
                "resolution": resolution,
                "score_version": (
                    str(refine_config["score_version"]) if args.refine else None
                ),
                "strength": strength,
            }
            expected_sidecar = {
                "bucket": bucket,
                "config_sha256": sha256_file(args.config),
                "effective_parameters": effective_parameters,
                "model_revision": revision,
                "object": args.object,
                "pipeline_version": pipeline_version(bucket),
                "placement_index": placement_index,
                "placements_sha256": str(object_config["placements_sha256"]),
                "seed": seed,
            }
            if sidecar_path.is_file():
                if not args.resume:
                    raise DiffusionGenerationError(f"Refusing to overwrite {sidecar_path}")
                load_completed_sidecar(
                    sidecar_path,
                    output_root=output_root,
                    expected=expected_sidecar,
                    blocklist=blocklist,
                )
                skipped += 1
                continue

            with Image.open(paths.visa_raw / str(placement["background_image"])) as handle:
                background_image = handle.convert("RGB")
            placement_mask_path = (
                paths.synthetic
                / "placements"
                / args.object
                / str(placement["mask_path"])
            )
            with Image.open(placement_mask_path) as handle:
                mask_image = handle.convert("L")
            background = np.asarray(background_image, dtype=np.uint8)
            mask = np.asarray(mask_image, dtype=np.uint8)

            if args.refine:
                parameters = candidate_parameters(
                    seed=seed,
                    object_name=args.object,
                    placement_index=placement_index,
                    guidance_grid=guidance_grid,
                    crop_ratio_grid=crop_ratio_grid,
                    count=num_search_run,
                    baseline_guidance=guidance_scale,
                    baseline_crop_ratio=crop_ratio,
                )
            else:
                parameters = [(guidance_scale, crop_ratio)]

            if pipeline is None:
                pipeline = load_pipeline(config=config, adapter=adapter, device=device)
            candidate_evidence: list[dict[str, Any]] = []
            selected_image: np.ndarray | None = None
            selected_crop_bbox: tuple[int, int, int, int] | None = None
            selected_parameters: tuple[float, float] | None = None
            selected_seed: int | None = None
            selected_score = -math.inf
            for candidate_index, (candidate_guidance, candidate_crop) in enumerate(parameters):
                generator_seed = derived_seed(
                    seed,
                    args.object,
                    placement_index,
                    candidate_index,
                )
                candidate_image, candidate_crop_bbox = render_candidate(
                    pipeline=pipeline,
                    background_image=background_image,
                    mask_image=mask_image,
                    prompt=prompt.format(token=placement["trigger_token"]),
                    negative_prompt=negative_prompt,
                    generator_seed=generator_seed,
                    guidance_scale=candidate_guidance,
                    crop_ratio=candidate_crop,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                    resolution=resolution,
                    blend=blend,
                    blend_dilation_px=int(generation_config["blend_dilation_px"]),
                    blend_feather_sigma=float(generation_config["blend_feather_sigma"]),
                    device=device,
                )
                evidence = score_candidate(
                    background,
                    candidate_image,
                    mask,
                    config=refine_config,
                )
                evidence.update(
                    {
                        "candidate_index": candidate_index,
                        "crop_bbox": list(candidate_crop_bbox),
                        "crop_ratio": candidate_crop,
                        "generator_seed": generator_seed,
                        "guidance_scale": candidate_guidance,
                    }
                )
                candidate_evidence.append(evidence)
                if evidence["score"] > selected_score:
                    selected_score = float(evidence["score"])
                    selected_image = candidate_image
                    selected_crop_bbox = candidate_crop_bbox
                    selected_parameters = (candidate_guidance, candidate_crop)
                    selected_seed = generator_seed

            if (
                selected_image is None
                or selected_crop_bbox is None
                or selected_parameters is None
                or selected_seed is None
            ):
                raise DiffusionGenerationError(f"No candidate selected: {sample_id}")
            record = build_metadata(
                placement=placement,
                bucket=bucket,
                image_path=image_relative,
                mask_path=mask_relative,
                model_id=model_id,
                adapter_path=logical_lora,
                description=str(object_config["description"]),
                negative_prompt=negative_prompt,
                generator_seed=selected_seed,
                guidance_scale=selected_parameters[0],
                num_inference_steps=num_inference_steps,
                strength=strength,
                crop_ratio=selected_parameters[1],
                crop_bbox=selected_crop_bbox,
                resolution=resolution,
                blend=blend,
            )
            write_png_atomic(image_path, Image.fromarray(selected_image))
            copy_atomic(placement_mask_path, mask_path)
            image_sha256 = sha256_file(image_path)
            mask_sha256 = sha256_file(mask_path)
            if image_sha256 in blocklist or mask_sha256 in blocklist:
                raise DiffusionGenerationError(f"Generated output hit test blocklist: {sample_id}")
            sidecar = {
                **expected_sidecar,
                "candidates": candidate_evidence,
                "image_sha256": image_sha256,
                "mask_sha256": mask_sha256,
                "record": record,
                "selected_candidate_index": next(
                    int(value["candidate_index"])
                    for value in candidate_evidence
                    if int(value["generator_seed"]) == selected_seed
                ),
            }
            write_json_atomic(sidecar_path, sidecar)
            generated += 1
            if (placement_index + 1) % 10 == 0 or placement_index + 1 == len(placements):
                rebuild_metadata(output_root)
                LOGGER.info(
                    "%s/%s: completed %d/%d (%d generated, %d resumed)",
                    args.object,
                    bucket,
                    placement_index + 1,
                    len(placements),
                    generated,
                    skipped,
                )

        all_records = rebuild_metadata(output_root)
        object_records = [
            record for record in all_records if str(record["object"]) == args.object
        ]
        if len(object_records) != n:
            raise DiffusionGenerationError(
                f"Expected {n} completed records for {args.object}, found {len(object_records)}"
            )
        if args.contact_sheet is not None:
            destination = (
                args.contact_sheet
                if args.contact_sheet.is_absolute()
                else paths.project_root / args.contact_sheet
            )
            create_contact_sheet(
                destination,
                paths=paths,
                output_root=output_root,
                records=object_records,
            )

        elapsed = time.perf_counter() - started
        result = {
            **summary,
            "elapsed_seconds": elapsed,
            "generated": generated,
            "peak_vram_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            ),
            "records": len(object_records),
            "skipped": skipped,
            "status": "passed",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        DiffusionGenerationError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        LOGGER.error("%s", error)
        return 2
    finally:
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
