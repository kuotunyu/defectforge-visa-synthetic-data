"""Verified CPU/GPU inference runtime for the public DefectForge Space."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as nnf
from PIL import Image, ImageOps
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf
from transformers import SegformerForSemanticSegmentation

SPACE_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = SPACE_ROOT / "models"
MANIFEST_PATH = MODEL_ROOT / "manifest.json"
MAX_IMAGE_PIXELS = 25_000_000
SUPPORTED_OBJECTS = ("pcb1", "capsules")


class SpaceContractError(RuntimeError):
    """Raised when a public-demo integrity or inference contract fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SpaceContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: object, *, space_root: Path = SPACE_ROOT) -> Path:
    require(isinstance(value, str) and bool(value), "Model path is missing")
    portable = PurePosixPath(value)
    require(not portable.is_absolute(), "Model path must be relative")
    require(".." not in portable.parts, "Model path cannot escape the model root")
    resolved_root = space_root.resolve(strict=False)
    candidate = (resolved_root / Path(*portable.parts)).resolve(strict=False)
    require(candidate.is_relative_to(resolved_root), "Model path escaped the Space root")
    return candidate


def _verified_artifact(
    record: dict[str, Any],
    *,
    label: str,
    space_root: Path = SPACE_ROOT,
) -> Path:
    path = _safe_relative_path(record.get("path"), space_root=space_root)
    require(path.is_file(), f"Missing {label}: {path.name}")
    expected_bytes = record.get("bytes")
    require(isinstance(expected_bytes, int) and expected_bytes > 0, f"Invalid {label} size")
    require(path.stat().st_size == expected_bytes, f"{label} size changed")
    expected_sha256 = record.get("sha256")
    require(
        isinstance(expected_sha256, str)
        and len(expected_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_sha256),
        f"Invalid {label} SHA256",
    )
    require(sha256_file(path) == expected_sha256, f"{label} SHA256 changed")
    return path


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    require(path.is_file(), "The verified model manifest is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "The model manifest must be a JSON object")
    require(payload.get("schema_version") == 1, "Unsupported model manifest schema")
    require(payload.get("status") == "passed", "The model manifest did not pass packaging")
    objects = payload.get("objects")
    require(isinstance(objects, dict), "The model manifest has no object inventory")
    require(tuple(objects) == SUPPORTED_OBJECTS, "The public object inventory changed")
    space_root = path.parent.parent
    for object_name, object_record in objects.items():
        require(isinstance(object_record, dict), f"Invalid manifest entry for {object_name}")
        for role in ("classifier", "segmenter"):
            role_record = object_record.get(role)
            require(isinstance(role_record, dict), f"Missing {object_name}/{role} record")
            _verified_artifact(
                role_record,
                label=f"{object_name} {role}",
                space_root=space_root,
            )
    return payload


def preferred_device() -> torch.device:
    requested = os.environ.get("DEFECTFORGE_DEVICE", "auto").strip().lower()
    require(requested in {"auto", "cpu", "cuda"}, "DEFECTFORGE_DEVICE must be auto/cpu/cuda")
    if requested == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(slots=True)
class ModelBundle:
    """One object-matched classifier and segmenter pair."""

    object_name: str
    classifier: torch.nn.Module
    segmenter: torch.nn.Module
    classifier_transform: transforms.Compose
    segmenter_size: int
    segmenter_mean: tuple[float, float, float]
    segmenter_std: tuple[float, float, float]
    formal_threshold: float
    device: torch.device
    evidence: dict[str, Any]


def _load_classifier(record: dict[str, Any], device: torch.device) -> tuple[Any, Any]:
    weight = _verified_artifact(record, label="classifier")
    architecture = str(record["architecture"])
    model = timm.create_model(architecture, pretrained=False)
    model.reset_classifier(int(record["num_classes"]))
    model.load_state_dict(load_file(str(weight), device="cpu"), strict=True)
    model = model.eval().to(device)
    size = int(record["input_size"])
    mean = tuple(float(value) for value in record["mean"])
    std = tuple(float(value) for value in record["std"])
    transform = transforms.Compose(
        [
            transforms.Resize(
                (size, size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return model, transform


def _load_segmenter(record: dict[str, Any], device: torch.device) -> torch.nn.Module:
    weight = _verified_artifact(record, label="segmenter")
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(weight.parent),
        local_files_only=True,
        use_safetensors=True,
    )
    return model.eval().to(device)


@lru_cache(maxsize=2)
def load_bundle(object_name: str) -> ModelBundle:
    require(object_name in SUPPORTED_OBJECTS, f"Unsupported object: {object_name}")
    manifest = load_manifest()
    object_record = manifest["objects"][object_name]
    device = preferred_device()
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    classifier, classifier_transform = _load_classifier(object_record["classifier"], device)
    segmenter = _load_segmenter(object_record["segmenter"], device)
    segmenter_record = object_record["segmenter"]
    evidence = {
        "object": object_name,
        "device": device.type,
        "source_commit": manifest["source_commit"],
        "classification_results_sha256": manifest["classification_results_sha256"],
        "segmentation_results_sha256": manifest["segmentation_results_sha256"],
        "classifier": {
            key: object_record["classifier"][key]
            for key in ("run_name", "group_name", "primary_metric", "primary_value", "sha256")
        },
        "segmenter": {
            key: segmenter_record[key]
            for key in ("run_name", "group_name", "primary_metric", "primary_value", "sha256")
        },
    }
    return ModelBundle(
        object_name=object_name,
        classifier=classifier,
        segmenter=segmenter,
        classifier_transform=classifier_transform,
        segmenter_size=int(segmenter_record["input_size"]),
        segmenter_mean=tuple(float(value) for value in segmenter_record["mean"]),
        segmenter_std=tuple(float(value) for value in segmenter_record["std"]),
        formal_threshold=float(segmenter_record["formal_threshold"]),
        device=device,
        evidence=evidence,
    )


def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        pil = ImageOps.exif_transpose(image).convert("RGB")
        require(pil.width * pil.height <= MAX_IMAGE_PIXELS, "Image exceeds the 25 MP limit")
        return np.asarray(pil, dtype=np.uint8)
    array = np.asarray(image)
    require(array.ndim in {2, 3}, "Input image must be HxW or HxWxC")
    require(array.shape[0] * array.shape[1] <= MAX_IMAGE_PIXELS, "Image exceeds the 25 MP limit")
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    require(array.shape[2] == 3, "Input image must have one, three, or four channels")
    if np.issubdtype(array.dtype, np.floating):
        maximum = float(np.nanmax(array)) if array.size else 0.0
        array = array * 255.0 if maximum <= 1.0 else array
    return np.clip(array, 0, 255).astype(np.uint8)


def _colorize_probability(probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    stops = np.asarray(
        [
            [5, 18, 27],
            [13, 91, 120],
            [61, 202, 170],
            [250, 194, 63],
            [239, 73, 51],
        ],
        dtype=np.float32,
    )
    position = values * (len(stops) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (position - lower)[..., None]
    return np.rint(stops[lower] * (1.0 - fraction) + stops[upper] * fraction).astype(np.uint8)


def localization_statistics(
    pixel_probability: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    """Summarize the raw pixel scores used to render a localization result."""
    probability = np.asarray(pixel_probability, dtype=np.float32)
    require(probability.ndim == 2 and probability.size > 0, "Pixel probability map is invalid")
    require(np.isfinite(probability).all(), "Pixel probability contains non-finite values")
    require(0.0 <= threshold <= 1.0, "Visualization threshold is outside [0, 1]")
    require(
        float(probability.min()) >= 0.0 and float(probability.max()) <= 1.0,
        "Pixel probability is outside [0, 1]",
    )
    return {
        "minimum": float(probability.min()),
        "mean": float(probability.mean()),
        "p95": float(np.quantile(probability, 0.95)),
        "p99": float(np.quantile(probability, 0.99)),
        "maximum": float(probability.max()),
        "coverage_percent": float((probability >= threshold).mean()) * 100.0,
    }


def render_outputs(
    image: Image.Image | np.ndarray,
    *,
    anomaly_probability: float,
    pixel_probability: np.ndarray,
    threshold: float,
    heatmap_mode: str = "overlay",
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    rgb = _as_rgb_array(image)
    probability = np.asarray(pixel_probability, dtype=np.float32)
    require(probability.shape == rgb.shape[:2], "Pixel probability shape changed")
    require(np.isfinite(probability).all(), "Pixel probability contains non-finite values")
    require(math.isfinite(anomaly_probability), "Classification probability is not finite")
    require(0.0 <= anomaly_probability <= 1.0, "Classification probability is outside [0, 1]")
    require(0.0 <= threshold <= 1.0, "Visualization threshold is outside [0, 1]")
    require(
        heatmap_mode in {"overlay", "probability"},
        "Heatmap display mode is unsupported",
    )
    mask = (probability >= threshold).astype(np.uint8) * 255
    color = _colorize_probability(probability)
    if heatmap_mode == "probability":
        heatmap = color
    else:
        alpha = (0.20 + 0.64 * np.clip(probability, 0.0, 1.0))[..., None]
        heatmap = np.rint(rgb * (1.0 - alpha) + color * alpha).astype(np.uint8)
    return (
        {
            "Defect（異常）": anomaly_probability,
            "Normal（正常）": 1.0 - anomaly_probability,
        },
        mask,
        heatmap,
    )


@torch.inference_mode()
def predict(
    image: Image.Image | np.ndarray | None,
    object_name: str,
    visualization_threshold: float,
    heatmap_mode: str = "overlay",
) -> tuple[dict[str, float], np.ndarray, np.ndarray, str, dict[str, Any]]:
    require(image is not None, "請先上傳一張待檢影像")
    bundle = load_bundle(object_name)
    rgb = _as_rgb_array(image)
    pil = Image.fromarray(rgb, mode="RGB")
    height, width = rgb.shape[:2]
    classifier_input = bundle.classifier_transform(pil).unsqueeze(0).to(bundle.device)
    segmenter_input = tvf.resize(
        pil,
        [bundle.segmenter_size, bundle.segmenter_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    segmenter_input = (
        tvf.normalize(
            tvf.to_tensor(segmenter_input),
            mean=bundle.segmenter_mean,
            std=bundle.segmenter_std,
        )
        .unsqueeze(0)
        .to(bundle.device)
    )
    if bundle.device.type == "cuda":
        torch.cuda.synchronize()
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast = nullcontext()
    started = time.perf_counter()
    with autocast:
        classifier_logits = bundle.classifier(classifier_input)
        segmenter_logits = bundle.segmenter(pixel_values=segmenter_input).logits
        segmenter_logits = nnf.interpolate(
            segmenter_logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    anomaly_probability = float(torch.softmax(classifier_logits.float(), dim=1)[0, 1].cpu())
    pixel_probability = torch.sigmoid(segmenter_logits.float())[0, 0].cpu().numpy()
    if bundle.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    probabilities, mask, heatmap = render_outputs(
        rgb,
        anomaly_probability=anomaly_probability,
        pixel_probability=pixel_probability,
        threshold=float(visualization_threshold),
        heatmap_mode=heatmap_mode,
    )
    localization = localization_statistics(
        pixel_probability,
        threshold=float(visualization_threshold),
    )
    coverage = localization["coverage_percent"]
    decision = "Defect（異常）" if anomaly_probability >= 0.5 else "Normal（正常）"
    summary = (
        f"### 檢測完成：{decision}\n"
        f"物件 `{object_name}` · 執行裝置 `{bundle.device.type.upper()}` · "
        f"耗時 **{elapsed_ms:.1f} ms**  \n"
        f"在顯示 threshold **{float(visualization_threshold):.2f}** 下，"
        f"binary mask 覆蓋影像 **{coverage:.2f}%**。"
        f"正式 preregistered threshold 為 **{bundle.formal_threshold:.2f}**。"
    )
    evidence = {
        **bundle.evidence,
        "input": {"width": width, "height": height},
        "inference": {
            "elapsed_ms": round(elapsed_ms, 3),
            "visualization_threshold": float(visualization_threshold),
            "formal_threshold": bundle.formal_threshold,
            "heatmap_mode": heatmap_mode,
            "mask_coverage_percent": round(coverage, 4),
            "pixel_probability_minimum": round(localization["minimum"], 6),
            "pixel_probability_mean": round(localization["mean"], 6),
            "pixel_probability_p95": round(localization["p95"], 6),
            "pixel_probability_p99": round(localization["p99"], 6),
            "pixel_probability_maximum": round(localization["maximum"], 6),
        },
    }
    return probabilities, mask, heatmap, summary, evidence
