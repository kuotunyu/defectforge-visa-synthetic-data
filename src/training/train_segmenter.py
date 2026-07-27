"""Train leakage-safe per-object SegFormer-B0 models for M18."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from skimage.measure import label as connected_components
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf
from transformers import SegformerConfig, SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file
from src.common.paths import Paths, load_paths
from src.training.segmenter_data import (
    SegmentationGroup,
    SegmentationSample,
    build_segmentation_group,
    group_payload_sha256,
    resolve_image_path,
    resolve_mask_path,
    summarize_samples,
)

RESULT_COLUMNS = (
    "run_name",
    "run_signature",
    "requested_group",
    "canonical_group",
    "object",
    "seed",
    "model",
    "model_revision",
    "input_size",
    "total_steps",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "train_total",
    "train_real",
    "train_synthetic",
    "train_real_defect",
    "train_synthetic_defect",
    "test_total",
    "dice",
    "miou",
    "pixel_auroc",
    "aupro",
    "peak_vram_gib",
    "training_seconds",
    "model_sha256",
    "data_manifest_sha256",
    "split_manifest_sha256",
    "selection_sha256",
)


class SegmenterTrainingError(RuntimeError):
    """Raised when an M18 run violates the frozen experiment contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SegmenterTrainingError(message)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Invalid segmenter config: {path}")
    require(value.get("schema_version") == 1, "Unsupported segmenter config schema")
    return value


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _stable_seed(seed: int, *values: str) -> int:
    payload = "\0".join([str(seed), *values]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _random_resized_crop(
    image: Image.Image,
    mask: Image.Image,
    *,
    size: int,
    scale_range: tuple[float, float],
    ratio_range: tuple[float, float],
    rng: random.Random,
) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    area = width * height
    log_ratio = (math.log(ratio_range[0]), math.log(ratio_range[1]))
    crop: tuple[int, int, int, int] | None = None
    for _ in range(10):
        target_area = area * rng.uniform(*scale_range)
        aspect = math.exp(rng.uniform(*log_ratio))
        crop_width = round(math.sqrt(target_area * aspect))
        crop_height = round(math.sqrt(target_area / aspect))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            left = rng.randint(0, width - crop_width)
            top = rng.randint(0, height - crop_height)
            crop = (top, left, crop_height, crop_width)
            break
    if crop is None:
        source_ratio = width / height
        if source_ratio < ratio_range[0]:
            crop_width = width
            crop_height = round(crop_width / ratio_range[0])
        elif source_ratio > ratio_range[1]:
            crop_height = height
            crop_width = round(crop_height * ratio_range[1])
        else:
            crop_width, crop_height = width, height
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        crop = (top, left, crop_height, crop_width)
    top, left, crop_height, crop_width = crop
    return (
        tvf.resized_crop(
            image,
            top,
            left,
            crop_height,
            crop_width,
            (size, size),
            InterpolationMode.BILINEAR,
            antialias=True,
        ),
        tvf.resized_crop(
            mask,
            top,
            left,
            crop_height,
            crop_width,
            (size, size),
            InterpolationMode.NEAREST,
        ),
    )


def _paired_transform(
    image: Image.Image,
    mask: Image.Image,
    *,
    config: Mapping[str, Any],
    standard_augmentation: bool,
    draw_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = int(config["model"]["input_size"])
    rng = random.Random(draw_seed)
    if standard_augmentation:
        augmentation = config["standard_augmentation"]
        image, mask = _random_resized_crop(
            image,
            mask,
            size=size,
            scale_range=tuple(
                float(value) for value in augmentation["random_resized_crop_scale"]
            ),
            ratio_range=tuple(
                float(value) for value in augmentation["random_resized_crop_ratio"]
            ),
            rng=rng,
        )
        if rng.random() < float(augmentation["horizontal_flip_probability"]):
            image = tvf.hflip(image)
            mask = tvf.hflip(mask)
        translation = float(augmentation["translation_fraction"])
        translations = [
            round(rng.uniform(-translation, translation) * size),
            round(rng.uniform(-translation, translation) * size),
        ]
        angle = rng.uniform(
            -float(augmentation["rotation_degrees"]),
            float(augmentation["rotation_degrees"]),
        )
        affine_scale = rng.uniform(*(float(value) for value in augmentation["scale"]))
        image = tvf.affine(
            image,
            angle=angle,
            translate=translations,
            scale=affine_scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        mask = tvf.affine(
            mask,
            angle=angle,
            translate=translations,
            scale=affine_scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        jitter = [float(value) for value in augmentation["color_jitter"]]
        factors = [
            ("brightness", rng.uniform(max(0.0, 1.0 - jitter[0]), 1.0 + jitter[0])),
            ("contrast", rng.uniform(max(0.0, 1.0 - jitter[1]), 1.0 + jitter[1])),
            ("saturation", rng.uniform(max(0.0, 1.0 - jitter[2]), 1.0 + jitter[2])),
            ("hue", rng.uniform(-jitter[3], jitter[3])),
        ]
        rng.shuffle(factors)
        for name, factor in factors:
            if name == "brightness":
                image = tvf.adjust_brightness(image, factor)
            elif name == "contrast":
                image = tvf.adjust_contrast(image, factor)
            elif name == "saturation":
                image = tvf.adjust_saturation(image, factor)
            else:
                image = tvf.adjust_hue(image, factor)
    else:
        image = tvf.resize(
            image,
            (size, size),
            InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = tvf.resize(mask, (size, size), InterpolationMode.NEAREST)

    image_tensor = tvf.to_tensor(image)
    image_tensor = tvf.normalize(
        image_tensor,
        mean=[float(value) for value in config["model"]["mean"]],
        std=[float(value) for value in config["model"]["std"]],
    )
    mask_tensor = (tvf.pil_to_tensor(mask).float() > 0).float()
    return image_tensor, mask_tensor


class SegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Load a portable M18 record and create a zero mask for normal images."""

    def __init__(
        self,
        paths: Paths,
        samples: Sequence[SegmentationSample],
        config: Mapping[str, Any],
        *,
        standard_augmentation: bool,
        seed: int,
    ) -> None:
        self.paths = paths
        self.samples = tuple(samples)
        self.config = config
        self.standard_augmentation = standard_augmentation
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        key: int | tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if isinstance(key, tuple):
            index, draw_index = key
        else:
            index, draw_index = key, key
        sample = self.samples[index]
        with Image.open(resolve_image_path(self.paths, sample)) as handle:
            image = handle.convert("RGB")
        mask_path = resolve_mask_path(self.paths, sample)
        if mask_path is None:
            mask = Image.new("L", image.size, color=0)
        else:
            with Image.open(mask_path) as handle:
                mask = handle.convert("L")
        draw_seed = _stable_seed(self.seed, sample.sample_id, str(draw_index))
        image_tensor, mask_tensor = _paired_transform(
            image,
            mask,
            config=self.config,
            standard_augmentation=self.standard_augmentation,
            draw_seed=draw_seed,
        )
        return image_tensor, mask_tensor, index


class FixedDrawSampler(Sampler[tuple[int, int]]):
    """Yield a deterministic precomputed sample index and its exposure number."""

    def __init__(self, indices: Sequence[int], *, offset: int = 0) -> None:
        self.indices = tuple(int(index) for index in indices)
        self.offset = offset

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(
            (sample_index, draw_index)
            for draw_index, sample_index in enumerate(self.indices)
            if draw_index >= self.offset
        )

    def __len__(self) -> int:
        return max(0, len(self.indices) - self.offset)


def balanced_draw_indices(
    samples: Sequence[SegmentationSample],
    *,
    num_samples: int,
    seed: int,
) -> list[int]:
    counts = Counter(sample.has_defect for sample in samples)
    require(set(counts) == {False, True}, "Training data must contain normal and defect masks")
    weights = torch.tensor(
        [1.0 / counts[sample.has_defect] for sample in samples],
        dtype=torch.double,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.multinomial(
        weights,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    ).tolist()


def binary_dice_bce_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    dice_weight: float,
    bce_weight: float,
    smooth: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    require(logits.shape == masks.shape, "Logit/mask shapes differ")
    bce = nnf.binary_cross_entropy_with_logits(logits, masks)
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(1, masks.ndim))
    intersection = (probabilities * masks).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + masks.sum(dim=dimensions)
    dice = ((2.0 * intersection + smooth) / (denominator + smooth)).mean()
    dice_loss = 1.0 - dice
    loss = dice_weight * dice_loss + bce_weight * bce
    return loss, {
        "dice_loss": float(dice_loss.detach().cpu()),
        "bce_loss": float(bce.detach().cpu()),
    }


def _aupro(
    probabilities: np.ndarray,
    masks: np.ndarray,
    *,
    max_fpr: float,
    thresholds: int,
) -> float:
    require(probabilities.shape == masks.shape, "AUPRO inputs differ in shape")
    require(0.0 < max_fpr <= 1.0, "AUPRO max_fpr is invalid")
    require(thresholds >= 2, "AUPRO requires at least two thresholds")
    background_scores = probabilities[~masks]
    require(background_scores.size > 0, "AUPRO has no background pixels")
    region_scores: list[np.ndarray] = []
    for score_map, mask in zip(probabilities, masks, strict=True):
        labels = connected_components(mask.astype(np.uint8), connectivity=2)
        for region_id in range(1, int(labels.max()) + 1):
            region_scores.append(score_map[labels == region_id])
    require(region_scores, "AUPRO has no anomaly regions")

    threshold_values = np.linspace(1.0 + 1e-7, -1e-7, thresholds)
    sorted_background = np.sort(background_scores)
    background_hits = background_scores.size - np.searchsorted(
        sorted_background,
        threshold_values,
        side="left",
    )
    fpr = background_hits.astype(np.float64) / background_scores.size
    pros = np.zeros_like(fpr)
    for scores in region_scores:
        sorted_scores = np.sort(scores)
        hits = scores.size - np.searchsorted(sorted_scores, threshold_values, side="left")
        pros += hits.astype(np.float64) / scores.size
    pros /= len(region_scores)

    order = np.argsort(fpr, kind="stable")
    fpr = fpr[order]
    pros = pros[order]
    keep = fpr <= max_fpr
    clipped_fpr = fpr[keep].tolist()
    clipped_pro = pros[keep].tolist()
    if not clipped_fpr or clipped_fpr[0] > 0.0:
        clipped_fpr.insert(0, 0.0)
        clipped_pro.insert(0, 0.0)
    if clipped_fpr[-1] < max_fpr:
        right = int(np.searchsorted(fpr, max_fpr, side="right"))
        if right < len(fpr):
            left = max(0, right - 1)
            x0, x1 = fpr[left], fpr[right]
            y0, y1 = pros[left], pros[right]
            if x1 == x0:
                interpolated = float(max(y0, y1))
            else:
                interpolated = float(y0 + (max_fpr - x0) * (y1 - y0) / (x1 - x0))
        else:
            interpolated = float(pros[-1])
        clipped_fpr.append(max_fpr)
        clipped_pro.append(interpolated)
    area = float(np.trapezoid(np.asarray(clipped_pro), np.asarray(clipped_fpr)))
    return area / max_fpr


def segmentation_metrics(
    probabilities: np.ndarray,
    masks: np.ndarray,
    *,
    threshold: float,
    au_pro_max_fpr: float,
    au_pro_thresholds: int,
) -> dict[str, float]:
    require(probabilities.shape == masks.shape and probabilities.ndim == 3, "Invalid metrics")
    masks = masks.astype(bool)
    predictions = probabilities >= threshold
    intersection = int(np.logical_and(predictions, masks).sum())
    predicted = int(predictions.sum())
    positives = int(masks.sum())
    dice = (2.0 * intersection) / max(1, predicted + positives)
    union_foreground = int(np.logical_or(predictions, masks).sum())
    foreground_iou = intersection / max(1, union_foreground)
    background_prediction = ~predictions
    background_mask = ~masks
    background_intersection = int(np.logical_and(background_prediction, background_mask).sum())
    background_union = int(np.logical_or(background_prediction, background_mask).sum())
    background_iou = background_intersection / max(1, background_union)
    flat_masks = masks.reshape(-1)
    require(bool(flat_masks.any()) and bool((~flat_masks).any()), "Pixel AUROC needs both classes")
    return {
        "dice": float(dice),
        "miou": float((foreground_iou + background_iou) / 2.0),
        "pixel_auroc": float(roc_auc_score(flat_masks, probabilities.reshape(-1))),
        "aupro": float(
            _aupro(
                probabilities,
                masks,
                max_fpr=au_pro_max_fpr,
                thresholds=au_pro_thresholds,
            )
        ),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp: bool,
    metrics_config: Mapping[str, Any],
) -> dict[str, float]:
    model.eval()
    probabilities: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for images, batch_masks, _ in loader:
        images = images.to(device, non_blocking=False)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(pixel_values=images).logits
            logits = nnf.interpolate(
                logits,
                size=batch_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy()[:, 0])
        masks.append(batch_masks.numpy()[:, 0] > 0)
    return segmentation_metrics(
        np.concatenate(probabilities),
        np.concatenate(masks),
        threshold=float(metrics_config["threshold"]),
        au_pro_max_fpr=float(metrics_config["au_pro_max_fpr"]),
        au_pro_thresholds=int(metrics_config["au_pro_thresholds"]),
    )


def download_locked_weights(paths: Paths, model_config: Mapping[str, Any]) -> Path:
    weight_path = Path(
        hf_hub_download(
            repo_id=str(model_config["repository"]),
            filename=str(model_config["filename"]),
            revision=str(model_config["revision"]),
            cache_dir=str(paths.cache / "huggingface"),
        )
    )
    observed = sha256_file(weight_path)
    require(observed == model_config["sha256"], f"SegFormer base hash mismatch: {observed}")
    return weight_path


def load_model(paths: Paths, model_config: Mapping[str, Any]) -> tuple[nn.Module, Path]:
    weight_path = download_locked_weights(paths, model_config)
    definition = SegformerConfig.from_pretrained(
        str(model_config["repository"]),
        revision=str(model_config["revision"]),
        cache_dir=str(paths.cache / "huggingface"),
    )
    definition.num_labels = int(model_config["num_labels"])
    definition.id2label = {0: "defect"}
    definition.label2id = {"defect": 0}
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(model_config["repository"]),
        config=definition,
        revision=str(model_config["revision"]),
        cache_dir=str(paths.cache / "huggingface"),
        use_safetensors=True,
        ignore_mismatched_sizes=True,
    )
    return model, weight_path


def cosine_multiplier(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    denominator = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    sampler: Sampler | None = None,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def _save_model(model: nn.Module, output_dir: Path) -> str:
    model_dir = output_dir / "final"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir, safe_serialization=True)
    weight_path = model_dir / "model.safetensors"
    require(weight_path.is_file(), "SegFormer final SafeTensors file is missing")
    return sha256_file(weight_path)


def _checkpoint_dirs(root: Path) -> list[Path]:
    candidates = []
    if root.is_dir():
        for path in root.glob("checkpoint-*"):
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit():
                candidates.append(path)
    return sorted(candidates, key=lambda path: int(path.name.removeprefix("checkpoint-")))


def _sync_output(output_dir: Path, drive_sync: Path | None) -> None:
    if drive_sync is None:
        return
    require(output_dir.resolve() != drive_sync.resolve(), "Drive sync equals local output")
    drive_sync.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output_dir, drive_sync, dirs_exist_ok=True)


def _save_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    output_dir: Path,
    step: int,
    run_signature: str,
    drive_sync: Path | None,
) -> Path:
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint / "model", safe_serialization=True)
    torch.save(
        {
            "schema_version": 1,
            "step": step,
            "run_signature": run_signature,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        checkpoint / "state.pt",
    )
    atomic_write_json(
        checkpoint / "checkpoint.json",
        {
            "schema_version": 1,
            "step": step,
            "run_signature": run_signature,
            "model_sha256": sha256_file(checkpoint / "model" / "model.safetensors"),
        },
    )
    _sync_output(output_dir, drive_sync)
    return checkpoint


def _restore_checkpoint(
    *,
    checkpoint: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    run_signature: str,
) -> int:
    metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    require(metadata["run_signature"] == run_signature, "Checkpoint run signature changed")
    weight_path = checkpoint / "model" / "model.safetensors"
    require(sha256_file(weight_path) == metadata["model_sha256"], "Checkpoint model changed")
    model.load_state_dict(load_file(str(weight_path), device="cpu"), strict=True)
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
    require(state["run_signature"] == run_signature, "Checkpoint state signature changed")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    return int(state["step"])


def _exposure_counts(
    samples: Sequence[SegmentationSample],
    indices: Sequence[int],
) -> dict[str, int]:
    return {
        "real_normal": sum(
            samples[index].kind == "real" and not samples[index].has_defect for index in indices
        ),
        "real_defect": sum(
            samples[index].kind == "real" and samples[index].has_defect for index in indices
        ),
        "synthetic_defect": sum(samples[index].kind == "synthetic" for index in indices),
    }


def train(
    *,
    paths: Paths,
    config: Mapping[str, Any],
    group: SegmentationGroup,
    output_dir: Path,
    seed: int,
    total_steps: int,
    learning_rate: float,
    weight_decay: float,
    smoke: bool,
    run_signature: str,
    resume_from_checkpoint: str | None,
    drive_sync: Path | None,
) -> dict[str, Any]:
    training_config = config["training"]
    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config["num_workers"])
    require(num_workers == 0, "M18 requires num_workers=0 for exact resume")
    amp = bool(training_config["amp"])
    set_determinism(seed)
    device = torch.device("cuda")
    require(torch.cuda.is_available(), "M18 training requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    output_dir.mkdir(parents=True, exist_ok=True)
    training_dataset = SegmentationDataset(
        paths,
        group.train,
        config,
        standard_augmentation=group.standard_augmentation,
        seed=seed,
    )
    evaluation_samples = group.validation if group.mode == "development" else group.test
    evaluation_dataset = SegmentationDataset(
        paths,
        evaluation_samples,
        config,
        standard_augmentation=False,
        seed=seed,
    )
    draw_indices = balanced_draw_indices(
        group.train,
        num_samples=total_steps * batch_size,
        seed=seed,
    )

    started_wall = time.perf_counter()
    model, base_weight_path = load_model(paths, config["model"])
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    warmup_steps = min(int(training_config["warmup_steps"]), max(0, total_steps - 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        ),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    completed_steps = 0
    if resume_from_checkpoint is not None:
        checkpoints = _checkpoint_dirs(output_dir)
        checkpoint = (
            checkpoints[-1]
            if resume_from_checkpoint == "latest" and checkpoints
            else Path(resume_from_checkpoint)
        )
        require(checkpoint.is_dir(), f"Resume checkpoint is missing: {checkpoint}")
        completed_steps = _restore_checkpoint(
            checkpoint=checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            run_signature=run_signature,
        )
        model.to(device)
    require(completed_steps < total_steps, "Checkpoint already reached the requested total steps")

    train_loader = _loader(
        training_dataset,
        batch_size=batch_size,
        sampler=FixedDrawSampler(draw_indices, offset=completed_steps * batch_size),
        num_workers=num_workers,
    )
    evaluation_loader = _loader(
        evaluation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    loss_config = config["loss"]
    loss_history: list[dict[str, float]] = []
    validation_history: list[dict[str, Any]] = []
    best_metrics: dict[str, float] | None = None
    best_step = 0
    stale_evaluations = 0
    checkpoint_every = max(1, int(training_config["checkpoint_every_steps"]))
    eval_every = max(1, int(training_config["eval_every_steps"]))
    training_started = time.perf_counter()

    for relative_step, (images, masks, _) in enumerate(train_loader, start=1):
        step = completed_steps + relative_step
        model.train()
        images = images.to(device, non_blocking=False)
        masks = masks.to(device, non_blocking=False)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(pixel_values=images).logits
            logits = nnf.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss, components = binary_dice_bce_loss(
                logits,
                masks,
                dice_weight=float(loss_config["dice_weight"]),
                bce_weight=float(loss_config["bce_weight"]),
                smooth=float(loss_config["smooth"]),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(training_config["gradient_clip_norm"]),
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_history.append({"step": step, "loss": float(loss.detach().cpu()), **components})

        if step % checkpoint_every == 0 and step < total_steps:
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                output_dir=output_dir,
                step=step,
                run_signature=run_signature,
                drive_sync=drive_sync,
            )

        should_evaluate = group.mode == "development" and (
            step % eval_every == 0 or step == total_steps
        )
        if should_evaluate:
            metrics = evaluate(
                model,
                evaluation_loader,
                device=device,
                amp=amp,
                metrics_config=config["metrics"],
            )
            validation_history.append({"step": step, **metrics})
            score = (metrics["dice"], metrics["aupro"], metrics["pixel_auroc"])
            best_score = (
                (-1.0, -1.0, -1.0)
                if best_metrics is None
                else (
                    best_metrics["dice"],
                    best_metrics["aupro"],
                    best_metrics["pixel_auroc"],
                )
            )
            if score > best_score:
                best_metrics = metrics
                best_step = step
                _save_model(model, output_dir)
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            if not smoke and stale_evaluations >= int(training_config["early_stopping_patience"]):
                break
        if step >= total_steps:
            break

    executed_steps = loss_history[-1]["step"]
    training_seconds = time.perf_counter() - training_started
    if group.mode == "development":
        require(best_metrics is not None, "Development run produced no validation metrics")
        final_weight = output_dir / "final" / "model.safetensors"
        model.load_state_dict(load_file(str(final_weight), device="cpu"), strict=True)
        model.to(device)
        metrics = best_metrics
        model_sha256 = sha256_file(final_weight)
    else:
        model_sha256 = _save_model(model, output_dir)
        metrics = evaluate(
            model,
            evaluation_loader,
            device=device,
            amp=amp,
            metrics_config=config["metrics"],
        )
        best_step = executed_steps
    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024**3)
    report = {
        "status": "passed",
        "pipeline_version": config["pipeline_version"],
        "mode": group.mode,
        "smoke": smoke,
        "object": group.object_name,
        "requested_group": group.requested_group,
        "canonical_group": group.canonical_group,
        "standard_augmentation": group.standard_augmentation,
        "seed": seed,
        "model": config["model"]["name"],
        "model_repository": config["model"]["repository"],
        "model_revision": config["model"]["revision"],
        "base_weight_sha256": config["model"]["sha256"],
        "base_weight_bytes": base_weight_path.stat().st_size,
        "input_size": config["model"]["input_size"],
        "batch_size": batch_size,
        "requested_total_steps": total_steps,
        "executed_steps": int(executed_steps),
        "best_step": int(best_step),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "training_seconds": training_seconds,
        "wall_clock_seconds": time.perf_counter() - started_wall,
        "peak_vram_gib": peak_vram_gib,
        "sample_exposure": _exposure_counts(
            group.train,
            draw_indices[: int(executed_steps) * batch_size],
        ),
        "train_counts": summarize_samples(group.train),
        "evaluation_counts": summarize_samples(evaluation_samples),
        "loss_history": loss_history,
        "validation_history": validation_history,
        "metrics": metrics,
        "model_sha256": model_sha256,
    }
    return report


def append_result(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            require(tuple(reader.fieldnames or ()) == RESULT_COLUMNS, "Segmentation CSV schema changed")
            existing = list(reader)
    require(
        not any(item["run_name"] == row["run_name"] for item in existing),
        f"Segmentation result already exists: {row['run_name']}",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow({name: row[name] for name in RESULT_COLUMNS})
    os.replace(temporary, path)


def result_row(
    *,
    run_name: str,
    run_signature: str,
    group: SegmentationGroup,
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    data_manifest_sha256: str,
) -> dict[str, Any]:
    metrics = report["metrics"]
    train_counts = report["train_counts"]
    return {
        "run_name": run_name,
        "run_signature": run_signature,
        "requested_group": group.requested_group,
        "canonical_group": group.canonical_group,
        "object": group.object_name,
        "seed": report["seed"],
        "model": report["model"],
        "model_revision": report["model_revision"],
        "input_size": report["input_size"],
        "total_steps": report["executed_steps"],
        "batch_size": report["batch_size"],
        "learning_rate": report["learning_rate"],
        "weight_decay": report["weight_decay"],
        "train_total": train_counts["total"],
        "train_real": train_counts["real"],
        "train_synthetic": train_counts["synthetic"],
        "train_real_defect": sum(
            sample.kind == "real" and sample.has_defect for sample in group.train
        ),
        "train_synthetic_defect": sum(sample.kind == "synthetic" for sample in group.train),
        "test_total": report["evaluation_counts"]["total"],
        "dice": metrics["dice"],
        "miou": metrics["miou"],
        "pixel_auroc": metrics["pixel_auroc"],
        "aupro": metrics["aupro"],
        "peak_vram_gib": report["peak_vram_gib"],
        "training_seconds": report["training_seconds"],
        "model_sha256": report["model_sha256"],
        "data_manifest_sha256": data_manifest_sha256,
        "split_manifest_sha256": group.manifest_sha256,
        "selection_sha256": group.selection_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/segmenter.yaml"))
    parser.add_argument("--object", dest="object_name", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
    parser.add_argument("--mode", choices=("development", "final"), default="final")
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--drive-sync", type=Path)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = load_paths(args.paths)
    config = load_config(args.config)
    require(args.object_name in paths.objects, f"Unsupported object: {args.object_name}")
    if args.smoke:
        require(args.group == "real_only", "M18 smoke must use real_only")
        require(args.mode == "development", "M18 smoke must use development mode")
    group = build_segmentation_group(
        paths,
        config,
        group_name=args.group,
        object_name=args.object_name,
        seed=args.seed,
        mode=args.mode,
    )
    training_config = config["training"]
    total_steps = 1 if args.smoke else int(args.total_steps or training_config["total_steps"])
    learning_rate = float(args.learning_rate or training_config["learning_rate"])
    weight_decay = float(
        training_config["weight_decay"] if args.weight_decay is None else args.weight_decay
    )
    run_name = args.run_name or (
        f"m18_{'smoke_' if args.smoke else ''}"
        f"{group.canonical_group}_{args.object_name}_seed{args.seed}"
    )
    output_dir = args.output_dir or (
        paths.runs / config["output"]["run_subdirectory"] / run_name
    )
    output_dir = output_dir.resolve(strict=False)
    drive_sync = None if args.drive_sync is None else args.drive_sync.resolve(strict=False)
    if (
        args.resume_from_checkpoint
        and drive_sync is not None
        and not output_dir.exists()
        and drive_sync.exists()
    ):
        shutil.copytree(drive_sync, output_dir)

    data_manifest_sha256 = group_payload_sha256(group)
    run_config = {
        "schema_version": 1,
        "pipeline_version": config["pipeline_version"],
        "run_name": run_name,
        "requested_group": group.requested_group,
        "canonical_group": group.canonical_group,
        "object": args.object_name,
        "seed": args.seed,
        "mode": args.mode,
        "smoke": args.smoke,
        "model": config["model"],
        "training": {
            "total_steps": total_steps,
            "batch_size": training_config["batch_size"],
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_steps": training_config["warmup_steps"],
        },
        "loss": config["loss"],
        "metrics": config["metrics"],
        "data_manifest_sha256": data_manifest_sha256,
        "split_manifest_sha256": group.manifest_sha256,
        "selection_sha256": group.selection_sha256,
    }
    run_signature = canonical_json_sha256(run_config)
    run_config["run_signature"] = run_signature
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "data_manifest.json", group.payload())
    atomic_write_json(output_dir / "run_config.json", run_config)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "run_name": run_name,
                    "run_signature": run_signature,
                    "output_dir": str(output_dir),
                    "group": group.payload()["counts"],
                },
                indent=2,
            )
        )
        return

    existing_report_path = output_dir / "training_report.json"
    if existing_report_path.is_file():
        existing_report = json.loads(existing_report_path.read_text(encoding="utf-8"))
        require(existing_report.get("status") == "passed", "Existing report did not pass")
        require(existing_report.get("run_signature") == run_signature, "Existing run changed")
        _sync_output(output_dir, drive_sync)
        print(f"Already complete: {output_dir}")
        return

    report = train(
        paths=paths,
        config=config,
        group=group,
        output_dir=output_dir,
        seed=args.seed,
        total_steps=total_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        smoke=args.smoke,
        run_signature=run_signature,
        resume_from_checkpoint=args.resume_from_checkpoint,
        drive_sync=drive_sync,
    )
    report["run_name"] = run_name
    report["run_signature"] = run_signature
    report["data_manifest_sha256"] = data_manifest_sha256
    report["split_manifest_sha256"] = group.manifest_sha256
    report["selection_sha256"] = group.selection_sha256
    atomic_write_json(existing_report_path, report)
    if not args.smoke and args.mode == "final":
        append_result(
            paths.project_root / config["output"]["results_csv"],
            result_row(
                run_name=run_name,
                run_signature=run_signature,
                group=group,
                config=config,
                report=report,
                data_manifest_sha256=data_manifest_sha256,
            ),
        )
    _sync_output(output_dir, drive_sync)
    print(f"Completed: {output_dir}")


if __name__ == "__main__":
    main()
