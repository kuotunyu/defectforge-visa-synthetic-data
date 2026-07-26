"""Train resumable per-object inpainting LoRA and trigger-token adapters."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfApi
from peft import (
    LoraConfig,
    PeftModel,
    TrainableTokensConfig,
    get_peft_model,
)
from PIL import Image
from skimage.measure import label
from torch.nn import functional
from transformers import CLIPTextModel, CLIPTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.imaging import mask_bbox
from src.common.integrity import (
    IntegrityError,
    assert_not_blocklisted,
    load_json,
    read_checksum_file,
    sha256_file,
    verify_frozen_manifest,
)
from src.common.paths import Paths, load_paths
from src.synthetic.copy_paste import ValidationError, object_code

LOGGER = logging.getLogger("train_inpaint_lora")
PIPELINE_VERSION = "0.2.0"
LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]


class TrainingError(RuntimeError):
    """Raised when an M10 training invariant fails."""


@dataclass(frozen=True)
class TrainingSample:
    object_name: str
    cluster_id: int
    trigger_token: str
    image_path: str
    mask_path: str
    component_id: int
    area_px: int
    crop_bbox: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument("--config", default="configs/lora_sd2.yaml", type=Path)
    parser.add_argument("--object", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--base-model")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--alpha", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--sample-every", type=int)
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        help="Controlled interruption point for checkpoint/resume validation.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--drive-sync", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TrainingError(f"Expected config object in {path}")
    return value


def output_directory(
    paths: Paths,
    config: dict[str, Any],
    *,
    object_name: str,
    seed: int,
    smoke: bool,
    override: Path | None,
) -> Path:
    if override is not None:
        return override if override.is_absolute() else PROJECT_ROOT / override
    name = str(config["output"]["name"])
    if smoke:
        name += "_smoke"
    return paths.runs / name / object_name / f"seed_{seed}"


def step_sample_indices(
    sample_count: int,
    batch_size: int,
    micro_step: int,
    *,
    seed: int,
    object_name: str,
) -> list[int]:
    if sample_count < 1 or batch_size < 1 or micro_step < 0:
        raise TrainingError("Invalid deterministic sample schedule arguments")
    result: list[int] = []
    start = micro_step * batch_size
    for absolute_position in range(start, start + batch_size):
        epoch, offset = divmod(absolute_position, sample_count)
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence([seed, object_code(f"{object_name}:lora-order"), epoch])
            )
        )
        permutation = rng.permutation(sample_count)
        result.append(int(permutation[offset]))
    return result


def sample_generator_seed(
    seed: int,
    object_name: str,
    micro_step: int,
    batch_offset: int,
) -> int:
    return int(
        np.random.SeedSequence(
            [
                seed,
                object_code(f"{object_name}:lora-sample"),
                micro_step,
                batch_offset,
            ]
        ).generate_state(1, dtype=np.uint64)[0]
    )


def load_training_samples(
    paths: Paths,
    object_name: str,
) -> tuple[list[TrainingSample], str, str, str]:
    manifest, manifest_sha256 = verify_frozen_manifest(paths.splits)
    selection_sha256, selection_filename = read_checksum_file(
        paths.splits / "FEWSHOT_SELECTION.sha256"
    )
    selection_path = paths.splits / selection_filename
    if sha256_file(selection_path) != selection_sha256:
        raise TrainingError("Few-shot selection checksum mismatch")
    selection = load_json(selection_path)
    if selection["manifest_sha256"] != manifest_sha256:
        raise TrainingError("Few-shot selection points to another manifest")
    defect_types_path = paths.splits / "defect_types.json"
    defect_types_sha256 = sha256_file(defect_types_path)
    defect_types = load_json(defect_types_path)
    if (
        defect_types["manifest_sha256"] != manifest_sha256
        or defect_types["fewshot_selection_sha256"] != selection_sha256
    ):
        raise TrainingError("Defect types do not match the frozen training selection")
    if object_name not in selection["objects"] or object_name not in defect_types["objects"]:
        raise TrainingError(f"Unknown object: {object_name}")

    seed_records = selection["objects"][object_name]["fewshot_seed"]
    if len(seed_records) != 10:
        raise TrainingError(f"{object_name} must have exactly 10 few-shot source images")
    seed_paths = {str(record["image_path"]) for record in seed_records}
    manifest_records = {
        str(record["image_path"]): record
        for record in manifest["images"]
        if record["object"] == object_name
    }
    samples: list[TrainingSample] = []
    for type_record in defect_types["objects"][object_name]["types"]:
        cluster_id = int(type_record["cluster_id"])
        trigger_token = str(type_record["trigger_token"])
        for component in type_record["components"]:
            image_path = str(component["image_path"])
            mask_path = str(component["mask_path"])
            if image_path not in seed_paths:
                raise TrainingError(f"Component is outside the frozen few-shot seed: {image_path}")
            manifest_record = manifest_records.get(image_path)
            if (
                manifest_record is None
                or manifest_record["mask_path"] != mask_path
                or manifest_record["set"] != "train"
                or manifest_record["label"] != "bad"
            ):
                raise TrainingError(f"Invalid frozen source component: {image_path}")
            samples.append(
                TrainingSample(
                    object_name=object_name,
                    cluster_id=cluster_id,
                    trigger_token=trigger_token,
                    image_path=image_path,
                    mask_path=mask_path,
                    component_id=int(component["component_id"]),
                    area_px=int(component["area_px"]),
                    crop_bbox=tuple(int(value) for value in component["crop_bbox"]),
                )
            )
    samples.sort(
        key=lambda sample: (
            sample.cluster_id,
            sample.image_path,
            sample.component_id,
        )
    )
    if not samples:
        raise TrainingError(f"No frozen training components for {object_name}")
    used_types = {sample.cluster_id for sample in samples}
    expected_types = {
        int(type_record["cluster_id"])
        for type_record in defect_types["objects"][object_name]["types"]
    }
    if used_types != expected_types:
        raise TrainingError(f"Training components do not cover every defect type: {object_name}")

    unique_files = sorted(
        {
            paths.visa_raw / relative
            for sample in samples
            for relative in (sample.image_path, sample.mask_path)
        }
    )
    assert_not_blocklisted(unique_files, paths.splits / "test_blocklist.json")
    selection_by_image = {str(record["image_path"]): record for record in seed_records}
    for sample in samples:
        selected = selection_by_image[sample.image_path]
        if (
            sha256_file(paths.visa_raw / sample.image_path) != selected["sha256"]
            or sha256_file(paths.visa_raw / sample.mask_path) != selected["mask_sha256"]
        ):
            raise TrainingError(f"Frozen few-shot source changed: {sample.image_path}")
    return samples, manifest_sha256, selection_sha256, defect_types_sha256


def sample_tensors(
    paths: Paths,
    sample: TrainingSample,
    *,
    resolution: int,
    flip: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with Image.open(paths.visa_raw / sample.image_path) as image_handle:
        image = np.asarray(image_handle.convert("RGB"))
    with Image.open(paths.visa_raw / sample.mask_path) as mask_handle:
        full_mask = np.asarray(mask_handle.convert("L")) > 0
    labels = label(full_mask, connectivity=2)
    component = labels == sample.component_id + 1
    if int(component.sum()) != sample.area_px:
        raise TrainingError(
            f"Frozen component area changed: {sample.image_path} #{sample.component_id}"
        )
    x0, y0, x1, y1 = sample.crop_bbox
    if (
        x0 < 0
        or y0 < 0
        or x1 > image.shape[1]
        or y1 > image.shape[0]
        or x1 <= x0
        or y1 <= y0
        or x1 - x0 != y1 - y0
    ):
        raise TrainingError(f"Invalid frozen crop bbox: {sample.crop_bbox}")
    image_crop = image[y0:y1, x0:x1]
    mask_crop = component[y0:y1, x0:x1]
    if flip:
        image_crop = np.ascontiguousarray(np.fliplr(image_crop))
        mask_crop = np.ascontiguousarray(np.fliplr(mask_crop))
    image_resized = np.asarray(
        Image.fromarray(image_crop).resize(
            (resolution, resolution),
            resample=Image.Resampling.BICUBIC,
        ),
        dtype=np.float32,
    )
    mask_resized = np.asarray(
        Image.fromarray(mask_crop.astype(np.uint8) * 255).resize(
            (resolution, resolution),
            resample=Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )
    mask_tensor = torch.from_numpy((mask_resized > 0).astype(np.float32))[None, :, :]
    image_tensor = torch.from_numpy(image_resized / 127.5 - 1.0).permute(2, 0, 1)
    masked_image = image_tensor * (1.0 - mask_tensor)
    if not torch.any(mask_tensor):
        raise TrainingError(f"Component disappeared during resize: {sample.image_path}")
    return image_tensor, mask_tensor, masked_image


def training_batch(
    paths: Paths,
    samples: list[TrainingSample],
    tokenizer: CLIPTokenizer,
    *,
    config: dict[str, Any],
    object_name: str,
    resolution: int,
    seed: int,
    micro_step: int,
) -> dict[str, torch.Tensor]:
    batch_size = int(config["training"]["train_batch_size"])
    indices = step_sample_indices(
        len(samples),
        batch_size,
        micro_step,
        seed=seed,
        object_name=object_name,
    )
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    masked_images: list[torch.Tensor] = []
    prompts: list[str] = []
    description = str(config["objects"][object_name]["description"])
    flip_probability = float(config["training"]["horizontal_flip_probability"])
    for batch_offset, index in enumerate(indices):
        rng = np.random.Generator(
            np.random.PCG64(sample_generator_seed(seed, object_name, micro_step, batch_offset))
        )
        image, mask, masked_image = sample_tensors(
            paths,
            samples[index],
            resolution=resolution,
            flip=bool(rng.random() < flip_probability),
        )
        images.append(image)
        masks.append(mask)
        masked_images.append(masked_image)
        prompts.append(f"a photo of {samples[index].trigger_token} defect on {description}")
    tokenized = tokenizer(
        prompts,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "pixel_values": torch.stack(images),
        "masks": torch.stack(masks),
        "masked_images": torch.stack(masked_images),
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
    }


def verify_remote_model(config: dict[str, Any], model_id: str, revision: str | None) -> str:
    if Path(model_id).exists():
        return "local"
    if revision is None:
        raise TrainingError("Remote base model must use an immutable revision")
    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    if info.sha != revision or info.private or info.gated or info.disabled:
        raise TrainingError(
            f"Base model is not the expected public immutable revision: {model_id}@{revision}"
        )
    expected = config["model"].get("expected_files", {})
    siblings = {file.rfilename: file for file in info.siblings}
    for filename, expected_sha256 in expected.items():
        sibling = siblings.get(filename)
        lfs = getattr(sibling, "lfs", None) if sibling is not None else None
        if lfs is None or lfs.sha256 != expected_sha256:
            raise TrainingError(f"Base model file metadata mismatch: {filename}")
    return str(info.sha)


def add_trigger_tokens(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    trigger_tokens: list[str],
) -> list[int]:
    added = tokenizer.add_special_tokens({"additional_special_tokens": sorted(trigger_tokens)})
    if added != len(trigger_tokens):
        raise TrainingError("One or more trigger tokens already exist in the base tokenizer")
    original_vocab_size = text_encoder.get_input_embeddings().num_embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in sorted(trigger_tokens)]
    initializer_ids = tokenizer.encode("defect", add_special_tokens=False)
    if not initializer_ids:
        raise TrainingError("Base tokenizer cannot encode the initializer word 'defect'")
    with torch.no_grad():
        embeddings = text_encoder.get_input_embeddings().weight
        initializer = embeddings[int(initializer_ids[0])].detach().clone()
        for token_id in token_ids:
            if token_id < original_vocab_size:
                raise TrainingError("Trigger token did not receive a new vocabulary row")
            embeddings[token_id] = initializer
    return token_ids


def cast_trainable_parameters(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)


def create_or_load_adapters(
    *,
    model_id: str,
    revision: str | None,
    tokenizer: CLIPTokenizer,
    trigger_tokens: list[str],
    rank: int,
    alpha: int,
    dropout: float,
    dtype: torch.dtype,
    resume_checkpoint: Path | None,
    gradient_checkpointing: bool,
) -> tuple[PeftModel, PeftModel, AutoencoderKL, DDPMScheduler, list[int]]:
    load_kwargs: dict[str, Any] = {
        "subfolder": "unet",
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    text_kwargs: dict[str, Any] = {
        "subfolder": "text_encoder",
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    vae_kwargs: dict[str, Any] = {
        "subfolder": "vae",
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    scheduler_kwargs: dict[str, Any] = {"subfolder": "scheduler"}
    if revision is not None:
        for kwargs in (load_kwargs, text_kwargs, vae_kwargs, scheduler_kwargs):
            kwargs["revision"] = revision
    base_unet = UNet2DConditionModel.from_pretrained(model_id, **load_kwargs)
    base_text = CLIPTextModel.from_pretrained(model_id, **text_kwargs)
    vae = AutoencoderKL.from_pretrained(model_id, **vae_kwargs)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, **scheduler_kwargs)
    if int(base_unet.config.in_channels) != 9 or int(base_unet.config.out_channels) != 4:
        raise TrainingError(
            f"Expected a 9->4 inpainting UNet, observed "
            f"{base_unet.config.in_channels}->{base_unet.config.out_channels}"
        )

    if resume_checkpoint is None:
        token_ids = add_trigger_tokens(tokenizer, base_text, trigger_tokens)
        unet = get_peft_model(
            base_unet,
            LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                init_lora_weights="gaussian",
                target_modules=LORA_TARGET_MODULES,
            ),
        )
        text_encoder = get_peft_model(
            base_text,
            TrainableTokensConfig(token_indices=token_ids),
        )
    else:
        saved_tokenizer = CLIPTokenizer.from_pretrained(resume_checkpoint / "tokenizer")
        if saved_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise TrainingError("Resume tokenizer differs from the selected checkpoint")
        base_text.resize_token_embeddings(len(saved_tokenizer))
        token_ids = [
            int(saved_tokenizer.convert_tokens_to_ids(token)) for token in sorted(trigger_tokens)
        ]
        unet = PeftModel.from_pretrained(
            base_unet,
            str(resume_checkpoint / "unet_adapter"),
            is_trainable=True,
        )
        text_encoder = PeftModel.from_pretrained(
            base_text,
            str(resume_checkpoint / "text_token_adapter"),
            is_trainable=True,
        )
    if gradient_checkpointing:
        base_unet.enable_gradient_checkpointing()
        base_text.gradient_checkpointing_enable()
    cast_trainable_parameters(unet)
    cast_trainable_parameters(text_encoder)
    vae.requires_grad_(False).eval()
    return unet, text_encoder, vae, noise_scheduler, token_ids


def checkpoint_candidates(output_dir: Path) -> list[Path]:
    result: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        required = (
            path / "trainer_state.json",
            path / "training_state.pt",
            path / "unet_adapter" / "adapter_config.json",
            path / "text_token_adapter" / "adapter_config.json",
            path / "tokenizer" / "tokenizer_config.json",
        )
        if all(item.is_file() for item in required):
            result.append((step, path))
    return [path for _, path in sorted(result)]


def resolve_resume_checkpoint(output_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    if value == "latest":
        candidates = checkpoint_candidates(output_dir)
        if not candidates:
            raise TrainingError(f"No complete checkpoint under {output_dir}")
        return candidates[-1]
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_dir() or not (path / "trainer_state.json").is_file():
        raise TrainingError(f"Invalid resume checkpoint: {path}")
    return path


def save_adapter_bundle(
    destination: Path,
    *,
    accelerator: Accelerator,
    unet: torch.nn.Module,
    text_encoder: torch.nn.Module,
    tokenizer: CLIPTokenizer,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    trainer_state: dict[str, Any],
) -> None:
    if destination.exists():
        raise TrainingError(f"Refusing to overwrite adapter bundle: {destination}")
    destination.mkdir(parents=True)
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_text = accelerator.unwrap_model(text_encoder)
    unwrapped_unet.save_pretrained(
        str(destination / "unet_adapter"),
        safe_serialization=True,
    )
    unwrapped_text.save_pretrained(
        str(destination / "text_token_adapter"),
        safe_serialization=True,
    )
    tokenizer.save_pretrained(destination / "tokenizer")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        destination / "training_state.pt",
    )
    (destination / "trainer_state.json").write_text(
        json.dumps(trainer_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def square_context_bbox(
    mask: np.ndarray,
    *,
    crop_ratio: float,
) -> tuple[int, int, int, int]:
    x, y, width, height = mask_bbox(mask)
    image_height, image_width = mask.shape
    side = max(32, math.ceil(max(width, height) * crop_ratio))
    side = min(side, image_width, image_height)
    center_x = x + width / 2
    center_y = y + height / 2
    x0 = max(0, min(round(center_x - side / 2), image_width - side))
    y0 = max(0, min(round(center_y - side / 2), image_height - side))
    return x0, y0, x0 + side, y0 + side


def heldout_placement(
    paths: Paths,
    object_name: str,
    trigger_token: str,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    metadata_path = paths.synthetic / "placements" / object_name / "placements.jsonl"
    record: dict[str, Any] | None = None
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            candidate = json.loads(line)
            if candidate["trigger_token"] == trigger_token:
                record = candidate
                break
    if record is None:
        raise TrainingError(f"No held-out placement for trigger token: {trigger_token}")
    with Image.open(paths.visa_raw / record["background_image"]) as image_handle:
        image = image_handle.convert("RGB")
    with Image.open(
        paths.synthetic / "placements" / object_name / record["mask_path"]
    ) as mask_handle:
        mask = mask_handle.convert("L")
    return image, mask, record


def render_training_sample(
    destination: Path,
    *,
    paths: Paths,
    config: dict[str, Any],
    object_name: str,
    seed: int,
    step: int,
    resolution: int,
    model_id: str,
    revision: str | None,
    accelerator: Accelerator,
    unet: torch.nn.Module,
    text_encoder: torch.nn.Module,
    tokenizer: CLIPTokenizer,
    vae: AutoencoderKL,
    trigger_token: str,
    smoke: bool,
) -> None:
    image, mask_image, placement_record = heldout_placement(
        paths,
        object_name,
        trigger_token,
    )
    mask_array = np.asarray(mask_image) > 0
    crop_bbox = square_context_bbox(
        mask_array,
        crop_ratio=float(config["training"]["sample_crop_ratio"]),
    )
    image_crop = image.crop(crop_bbox).resize(
        (resolution, resolution),
        resample=Image.Resampling.BICUBIC,
    )
    mask_crop = mask_image.crop(crop_bbox).resize(
        (resolution, resolution),
        resample=Image.Resampling.NEAREST,
    )
    description = str(config["objects"][object_name]["description"])
    prompt = f"a photo of {trigger_token} defect on {description}"
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_text = accelerator.unwrap_model(text_encoder)
    unwrapped_unet.eval()
    unwrapped_text.eval()
    vae_dtype = next(vae.parameters()).dtype

    def cast_sampling_latents(
        _pipeline: StableDiffusionInpaintPipeline,
        _step_index: int,
        _timestep: torch.Tensor,
        callback_kwargs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # PNDM can promote the last latent to float32 while the SD2 VAE remains
        # float16. Diffusers 0.39 does not cast again before VAE decode.
        callback_kwargs["latents"] = callback_kwargs["latents"].to(dtype=vae_dtype)
        return callback_kwargs

    pipeline_kwargs: dict[str, Any] = {
        "unet": unwrapped_unet,
        "text_encoder": unwrapped_text,
        "tokenizer": tokenizer,
        "vae": vae,
        "torch_dtype": torch.float16,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    if revision is not None:
        pipeline_kwargs["revision"] = revision
    pipeline: StableDiffusionInpaintPipeline | None = None
    try:
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            **pipeline_kwargs,
        ).to(accelerator.device)
        generator_seed = sample_generator_seed(seed, object_name, step, 0)
        generator = torch.Generator(device=accelerator.device).manual_seed(generator_seed)
        with torch.inference_mode():
            generated = pipeline(
                prompt=prompt,
                image=image_crop,
                mask_image=mask_crop,
                num_inference_steps=(
                    2 if smoke else int(config["training"]["sample_inference_steps"])
                ),
                guidance_scale=float(config["training"]["sample_guidance_scale"]),
                generator=generator,
                callback_on_step_end=cast_sampling_latents,
                callback_on_step_end_tensor_inputs=["latents"],
            ).images[0]
    finally:
        del pipeline
        unwrapped_unet.train()
        unwrapped_text.train()
        torch.cuda.empty_cache()
    panel = Image.new("RGB", (resolution * 3, resolution), "white")
    panel.paste(image_crop, (0, 0))
    panel.paste(mask_crop.convert("RGB"), (resolution, 0))
    panel.paste(generated, (resolution * 2, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.save(destination, format="PNG", compress_level=6)
    sample_metadata = {
        "background_image": placement_record["background_image"],
        "background_sha256": placement_record["background_sha256"],
        "crop_bbox": list(crop_bbox),
        "generator_seed": generator_seed,
        "mask_path": placement_record["mask_path"],
        "model_id": model_id,
        "model_revision": revision,
        "panel_sha256": sha256_file(destination),
        "placement_id": placement_record["placement_id"],
        "prompt": prompt,
        "step": step,
        "trigger_token": trigger_token,
    }
    destination.with_suffix(".json").write_text(
        json.dumps(sample_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_saved_bundle(
    bundle: Path,
    *,
    model_id: str,
    revision: str | None,
) -> dict[str, Any]:
    tokenizer = CLIPTokenizer.from_pretrained(bundle / "tokenizer")
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16,
        "use_safetensors": True,
    }
    if revision is not None:
        kwargs["revision"] = revision
    base_unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        **kwargs,
    )
    base_text = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        **kwargs,
    )
    base_text.resize_token_embeddings(len(tokenizer))
    loaded_unet = PeftModel.from_pretrained(
        base_unet,
        str(bundle / "unet_adapter"),
    )
    loaded_text = PeftModel.from_pretrained(
        base_text,
        str(bundle / "text_token_adapter"),
    )
    if not isinstance(loaded_unet, PeftModel) or not isinstance(loaded_text, PeftModel):
        raise TrainingError("Saved adapters did not reload through PeftModel")
    result = {
        "unet_adapter_sha256": sha256_file(bundle / "unet_adapter" / "adapter_model.safetensors"),
        "text_token_adapter_sha256": sha256_file(
            bundle / "text_token_adapter" / "adapter_model.safetensors"
        ),
        "tokenizer_vocab_size": len(tokenizer),
        "reload_class": type(loaded_unet).__name__,
    }
    del loaded_unet, loaded_text, base_unet, base_text
    gc.collect()
    return result


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    started_at = time.perf_counter()
    try:
        paths = load_paths(args.paths)
        config = load_config(args.config)
        if args.object not in paths.objects or args.object not in config["objects"]:
            raise TrainingError(f"Unknown object: {args.object}")
        seed = paths.seed if args.seed is None else args.seed
        model_id = args.base_model or str(config["model"]["id"])
        revision = (
            str(config["model"]["revision"]) if model_id == str(config["model"]["id"]) else None
        )
        resolution = int(args.resolution or config["model"]["resolution"])
        rank = int(args.rank or config["training"]["rank"])
        alpha = int(args.alpha or config["training"]["alpha"])
        max_train_steps = int(args.max_train_steps or config["training"]["max_train_steps"])
        learning_rate = float(args.lr or config["training"]["learning_rate"])
        sample_every = int(args.sample_every or config["training"]["sample_every"])
        if args.smoke:
            max_train_steps = 1
            sample_every = 1
        if (
            resolution < 32
            or resolution % 8
            or rank < 1
            or alpha < 1
            or max_train_steps < 1
            or learning_rate <= 0
            or sample_every < 1
        ):
            raise TrainingError("Invalid training hyperparameters")
        if args.stop_after_steps is not None and not (1 <= args.stop_after_steps < max_train_steps):
            raise TrainingError("--stop-after-steps must be between 1 and max steps - 1")
        run_until_step = args.stop_after_steps or max_train_steps
        samples, manifest_sha256, selection_sha256, defect_types_sha256 = load_training_samples(
            paths, args.object
        )
        model_revision = verify_remote_model(config, model_id, revision)
        trigger_tokens = sorted({sample.trigger_token for sample in samples})
        dry_run_summary = {
            "object": args.object,
            "source_images": len({sample.image_path for sample in samples}),
            "component_samples": len(samples),
            "type_counts": {
                token: sum(sample.trigger_token == token for sample in samples)
                for token in trigger_tokens
            },
            "base_model": model_id,
            "base_model_revision": model_revision,
            "resolution": resolution,
            "rank": rank,
            "alpha": alpha,
            "max_train_steps": max_train_steps,
            "learning_rate": learning_rate,
            "token_learning_rate": float(config["training"]["token_learning_rate"]),
            "training_config_sha256": sha256_file(args.config),
            "manifest_sha256": manifest_sha256,
            "selection_sha256": selection_sha256,
            "defect_types_sha256": defect_types_sha256,
            "status": "validated",
        }
        signature_payload = {
            key: dry_run_summary[key]
            for key in (
                "object",
                "component_samples",
                "type_counts",
                "base_model",
                "base_model_revision",
                "resolution",
                "rank",
                "alpha",
                "max_train_steps",
                "learning_rate",
                "token_learning_rate",
                "training_config_sha256",
                "manifest_sha256",
                "selection_sha256",
                "defect_types_sha256",
            )
        }
        dry_run_summary["run_signature"] = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if args.dry_run:
            print(json.dumps(dry_run_summary, indent=2, sort_keys=True))
            return 0

        output_dir = output_directory(
            paths,
            config,
            object_name=args.object,
            seed=seed,
            smoke=args.smoke,
            override=args.output_dir,
        )
        if output_dir.exists() and args.resume_from_checkpoint is None:
            raise TrainingError(
                f"Output exists; pass --resume-from-checkpoint or choose another path: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        resume_checkpoint = resolve_resume_checkpoint(
            output_dir,
            args.resume_from_checkpoint,
        )
        if args.resume_from_checkpoint is not None and resume_checkpoint is None:
            raise TrainingError("Resume was requested but no checkpoint was selected")

        mixed_precision = str(config["training"]["mixed_precision"])
        accelerator = Accelerator(
            gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
            mixed_precision=mixed_precision,
        )
        if accelerator.num_processes != 1:
            raise TrainingError("M10 currently requires a single local process")
        if accelerator.device.type != "cuda":
            raise TrainingError("M10 SD2 training requires CUDA")
        if bool(config["training"]["allow_tf32"]):
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.reset_peak_memory_stats(accelerator.device)
        weight_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16

        tokenizer_kwargs: dict[str, Any] = {"subfolder": "tokenizer"}
        if revision is not None:
            tokenizer_kwargs["revision"] = revision
        tokenizer = CLIPTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        if resume_checkpoint is not None:
            tokenizer = CLIPTokenizer.from_pretrained(resume_checkpoint / "tokenizer")
        (
            unet,
            text_encoder,
            vae,
            noise_scheduler,
            token_ids,
        ) = create_or_load_adapters(
            model_id=model_id,
            revision=revision,
            tokenizer=tokenizer,
            trigger_tokens=trigger_tokens,
            rank=rank,
            alpha=alpha,
            dropout=float(config["training"]["lora_dropout"]),
            dtype=weight_dtype,
            resume_checkpoint=resume_checkpoint,
            gradient_checkpointing=bool(config["training"]["gradient_checkpointing"]),
        )
        trainable_unet = [parameter for parameter in unet.parameters() if parameter.requires_grad]
        trainable_text = [
            parameter for parameter in text_encoder.parameters() if parameter.requires_grad
        ]
        if not trainable_unet or not trainable_text:
            raise TrainingError("LoRA or trigger-token adapter has no trainable parameters")
        optimizer = torch.optim.AdamW(
            [
                {"params": trainable_unet, "lr": learning_rate},
                {
                    "params": trainable_text,
                    "lr": float(config["training"]["token_learning_rate"]),
                },
            ],
            betas=(
                float(config["training"]["adam_beta1"]),
                float(config["training"]["adam_beta2"]),
            ),
            weight_decay=float(config["training"]["adam_weight_decay"]),
            eps=float(config["training"]["adam_epsilon"]),
        )
        lr_scheduler = get_scheduler(
            str(config["training"]["lr_scheduler"]),
            optimizer=optimizer,
            num_warmup_steps=int(config["training"]["lr_warmup_steps"]),
            num_training_steps=max_train_steps,
        )
        global_step = 0
        micro_step = 0
        previous_elapsed = 0.0
        if resume_checkpoint is not None:
            trainer_state = load_json(resume_checkpoint / "trainer_state.json")
            if (
                trainer_state["run_signature"] != dry_run_summary["run_signature"]
                or int(trainer_state["seed"]) != seed
            ):
                raise TrainingError("Resume checkpoint configuration mismatch")
            global_step = int(trainer_state["global_step"])
            micro_step = int(trainer_state["micro_step"])
            previous_elapsed = float(trainer_state["training_elapsed_seconds"])
            saved_state = torch.load(
                resume_checkpoint / "training_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            optimizer.load_state_dict(saved_state["optimizer"])
            lr_scheduler.load_state_dict(saved_state["lr_scheduler"])
            LOGGER.info("Resuming %s from step %d", args.object, global_step)
        if global_step >= max_train_steps:
            raise TrainingError(
                f"Checkpoint step {global_step} already reaches requested max {max_train_steps}"
            )
        if global_step >= run_until_step:
            raise TrainingError(
                f"Checkpoint step {global_step} already reaches controlled stop {run_until_step}"
            )

        unet, text_encoder, optimizer, lr_scheduler = accelerator.prepare(
            unet,
            text_encoder,
            optimizer,
            lr_scheduler,
        )
        vae.to(accelerator.device, dtype=weight_dtype)
        unet.train()
        text_encoder.train()
        optimizer.zero_grad(set_to_none=True)
        training_started = time.perf_counter()
        last_loss = float("nan")
        checkpoint_every = int(config["training"]["checkpoint_every"])
        report_every = int(config["training"]["report_every"])
        mask_loss_weight = float(config["training"]["mask_loss_weight"])

        while global_step < run_until_step:
            batch = training_batch(
                paths,
                samples,
                tokenizer,
                config=config,
                object_name=args.object,
                resolution=resolution,
                seed=seed,
                micro_step=micro_step,
            )
            with accelerator.accumulate(unet, text_encoder):
                pixel_values = batch["pixel_values"].to(
                    accelerator.device,
                    dtype=weight_dtype,
                )
                masks = batch["masks"].to(accelerator.device, dtype=weight_dtype)
                masked_images = batch["masked_images"].to(
                    accelerator.device,
                    dtype=weight_dtype,
                )
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)
                generator = torch.Generator(device=accelerator.device).manual_seed(
                    sample_generator_seed(seed, args.object, micro_step, 99)
                )
                with torch.no_grad():
                    encoded = vae.encode(
                        torch.cat((pixel_values, masked_images), dim=0)
                    ).latent_dist.sample(generator=generator)
                    encoded = encoded * float(vae.config.scaling_factor)
                    latents, masked_latents = encoded.chunk(2, dim=0)
                noise = torch.randn(
                    latents.shape,
                    generator=generator,
                    device=latents.device,
                    dtype=latents.dtype,
                )
                timesteps = torch.randint(
                    0,
                    int(noise_scheduler.config.num_train_timesteps),
                    (latents.shape[0],),
                    generator=generator,
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                latent_masks = functional.interpolate(
                    masks,
                    size=noisy_latents.shape[-2:],
                    mode="nearest",
                )
                model_input = torch.cat(
                    (noisy_latents, latent_masks, masked_latents),
                    dim=1,
                )
                encoder_hidden_states = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )[0]
                model_prediction = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states,
                ).sample
                prediction_type = str(noise_scheduler.config.prediction_type)
                if prediction_type == "epsilon":
                    target = noise
                elif prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise TrainingError(f"Unsupported scheduler prediction type: {prediction_type}")
                spatial_weights = 1.0 + (mask_loss_weight - 1.0) * latent_masks
                loss = (
                    (model_prediction.float() - target.float()).square() * spatial_weights.float()
                ).sum() / (spatial_weights.float().sum() * model_prediction.shape[1])
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [*trainable_unet, *trainable_text],
                        float(config["training"]["max_grad_norm"]),
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            micro_step += 1
            last_loss = float(loss.detach().item())
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            elapsed = previous_elapsed + time.perf_counter() - training_started
            if global_step % report_every == 0 or global_step == max_train_steps:
                LOGGER.info(
                    "%s step %d/%d loss=%.6f elapsed=%.1fs peak_vram=%.2fGiB",
                    args.object,
                    global_step,
                    max_train_steps,
                    last_loss,
                    elapsed,
                    torch.cuda.max_memory_allocated(accelerator.device) / 2**30,
                )
            trainer_state = {
                **dry_run_summary,
                "seed": seed,
                "global_step": global_step,
                "micro_step": micro_step,
                "training_elapsed_seconds": elapsed,
                "last_loss": last_loss,
                "token_ids": token_ids,
                "created_at": datetime.now(UTC).isoformat(),
                "pipeline_version": PIPELINE_VERSION,
            }
            if (
                global_step % checkpoint_every == 0
                or global_step == max_train_steps
                or global_step == run_until_step
            ):
                checkpoint_dir = output_dir / f"checkpoint-{global_step:06d}"
                save_adapter_bundle(
                    checkpoint_dir,
                    accelerator=accelerator,
                    unet=unet,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    trainer_state=trainer_state,
                )
                if args.drive_sync is not None:
                    sync_root = (
                        args.drive_sync
                        if args.drive_sync.is_absolute()
                        else PROJECT_ROOT / args.drive_sync
                    )
                    sync_root.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(
                        checkpoint_dir,
                        sync_root / checkpoint_dir.name,
                        dirs_exist_ok=False,
                    )
            if global_step % sample_every == 0 or global_step == max_train_steps:
                sample_type_index = ((global_step - 1) // sample_every) % len(trigger_tokens)
                render_training_sample(
                    output_dir / "samples" / f"step_{global_step:06d}.png",
                    paths=paths,
                    config=config,
                    object_name=args.object,
                    seed=seed,
                    step=global_step,
                    resolution=resolution,
                    model_id=model_id,
                    revision=revision,
                    accelerator=accelerator,
                    unet=unet,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    vae=vae,
                    trigger_token=trigger_tokens[sample_type_index],
                    smoke=args.smoke,
                )

        accelerator.wait_for_everyone()
        training_elapsed = previous_elapsed + time.perf_counter() - training_started
        if global_step < max_train_steps:
            controlled_stop_report = {
                **trainer_state,
                "checkpoint": f"checkpoint-{global_step:06d}",
                "output_dir": str(output_dir),
                "status": "controlled_stop",
            }
            (output_dir / "controlled_stop_report.json").write_text(
                json.dumps(controlled_stop_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            LOGGER.info(
                "Controlled stop for %s at step %d/%d; resume checkpoint is ready",
                args.object,
                global_step,
                max_train_steps,
            )
            return 0
        final_dir = output_dir / "final"
        final_state = {
            **dry_run_summary,
            "seed": seed,
            "global_step": global_step,
            "micro_step": micro_step,
            "training_elapsed_seconds": training_elapsed,
            "last_loss": last_loss,
            "token_ids": token_ids,
            "created_at": datetime.now(UTC).isoformat(),
            "pipeline_version": PIPELINE_VERSION,
        }
        save_adapter_bundle(
            final_dir,
            accelerator=accelerator,
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            trainer_state=final_state,
        )
        peak_vram_gib = torch.cuda.max_memory_allocated(accelerator.device) / 2**30
        del optimizer, lr_scheduler, unet, text_encoder, vae
        gc.collect()
        torch.cuda.empty_cache()
        reload_evidence = validate_saved_bundle(
            final_dir,
            model_id=model_id,
            revision=revision,
        )
        report = {
            **final_state,
            **reload_evidence,
            "output_dir": str(output_dir),
            "peak_vram_gib": peak_vram_gib,
            "steps_per_second": global_step / max(training_elapsed, 1e-9),
            "wall_clock_seconds": time.perf_counter() - started_at,
            "sample_files": sorted(
                path.relative_to(output_dir).as_posix()
                for path in (output_dir / "samples").glob("*.png")
            ),
            "status": "passed",
        }
        (output_dir / "training_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        LOGGER.info(
            "M10 %s complete: %d steps in %.1fs, peak %.2fGiB",
            args.object,
            global_step,
            training_elapsed,
            peak_vram_gib,
        )
        return 0
    except (
        IntegrityError,
        OSError,
        RuntimeError,
        TrainingError,
        ValidationError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
