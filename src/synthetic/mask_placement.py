"""Create deterministic SDG-ready clean-image and placed-mask test cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from skimage.measure import label
from torch.nn import functional
from transformers import AutoImageProcessor, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.imaging import ImagingError, detect_legal_roi, mask_bbox, place_on_legal_roi
from src.common.integrity import (
    IntegrityError,
    assert_not_blocklisted,
    load_json,
    read_checksum_file,
    sha256_file,
    verify_frozen_manifest,
)
from src.common.paths import Paths, load_paths
from src.synthetic.copy_paste import (
    OUTPUT_NAME,
    ValidationError,
    allocate_quotas,
    object_code,
    range_value,
)
from src.synthetic.procedural import log_uniform

LOGGER = logging.getLogger("mask_placement")
PIPELINE_VERSION = "0.1.0"
ROI_METHODS = ("otsu", "dinov2", "intersect")
PLACEMENT_FIELDS = {
    "placement_id",
    "object",
    "defect_type",
    "trigger_token",
    "variant_index",
    "background_image",
    "background_sha256",
    "source_image",
    "source_mask",
    "source_component_id",
    "mask_path",
    "roi_method",
    "roi_bbox",
    "mask_bbox",
    "affine",
    "mask_area_px",
    "mask_area_ratio",
    "aspect_ratio",
    "failed_attempts",
    "seed",
    "pipeline_version",
    "created_at",
}
AFFINE_FIELDS = {"dx", "dy", "rotation_deg", "scale", "flip"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
COMPONENT_MASK_CACHE: dict[tuple[str, int], np.ndarray] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument("--config", default="configs/placement.yaml", type=Path)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-per-image", type=int, default=3)
    parser.add_argument("--roi-method", choices=ROI_METHODS, default="intersect")
    parser.add_argument("--max-place-tries", type=int, default=50)
    parser.add_argument("--viz-n", type=int)
    parser.add_argument("--limit-backgrounds", type=int)
    parser.add_argument("--out-name", default="placements")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValidationError(f"Expected config object in {path}")
    return value


def validate_placement_record(record: dict[str, Any]) -> None:
    observed = set(record)
    if observed != PLACEMENT_FIELDS:
        raise ValidationError(
            f"Placement fields differ: missing={sorted(PLACEMENT_FIELDS - observed)}, "
            f"extra={sorted(observed - PLACEMENT_FIELDS)}"
        )
    if set(record["affine"]) != AFFINE_FIELDS:
        raise ValidationError("Placement affine fields differ")
    if record["object"] not in {"pcb1", "capsules"}:
        raise ValidationError(f"Unexpected placement object: {record['object']}")
    if record["roi_method"] not in ROI_METHODS:
        raise ValidationError(f"Unexpected ROI method: {record['roi_method']}")
    if int(record["mask_area_px"]) <= 0:
        raise ValidationError("Placement mask area must be positive")
    if not 0 < float(record["mask_area_ratio"]) <= 1:
        raise ValidationError("Placement mask area ratio must be in (0, 1]")


def placement_seed_sequence(
    seed: int,
    object_name: str,
    index: int,
) -> np.random.SeedSequence:
    return np.random.SeedSequence(
        [seed, object_code(f"{object_name}:placement"), index],
    )


def placement_rng(seed: int, object_name: str, index: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(placement_seed_sequence(seed, object_name, index)))


def placement_seed(seed: int, object_name: str, index: int) -> int:
    return int(
        placement_seed_sequence(seed, object_name, index).generate_state(
            1,
            dtype=np.uint64,
        )[0]
    )


def weighted_schedule(quotas: dict[int, int]) -> list[int]:
    """Interleave exact quotas with deterministic deficit round-robin."""

    total = sum(quotas.values())
    assigned = {key: 0 for key in quotas}
    schedule: list[int] = []
    for step in range(total):
        candidates = [key for key in quotas if assigned[key] < quotas[key]]
        selected = max(
            candidates,
            key=lambda key: (
                (step + 1) * quotas[key] / total - assigned[key],
                quotas[key],
                -key,
            ),
        )
        assigned[selected] += 1
        schedule.append(selected)
    if Counter(schedule) != Counter(quotas):
        raise ValidationError("Weighted placement schedule did not preserve quotas")
    return schedule


def object_section(
    config: dict[str, Any],
    section: str,
    object_name: str,
) -> dict[str, Any]:
    parent = config["roi"][section]
    result = dict(parent["default"])
    result.update(parent.get("objects", {}).get(object_name, {}))
    return result


def dino_cache_key(
    backgrounds: list[dict[str, Any]],
    *,
    manifest_sha256: str,
    model_config: dict[str, Any],
) -> str:
    payload = {
        "manifest_sha256": manifest_sha256,
        "model": model_config,
        "backgrounds": [
            {
                "image_path": record["image_path"],
                "sha256": record["sha256"],
            }
            for record in backgrounds
        ],
        "algorithm": "cosine_local_0.7_global_0.3_v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _token_scores(tokens: torch.Tensor) -> torch.Tensor:
    """Return local/global cosine heterogeneity for BxHxWxC patch tokens."""

    normalized = functional.normalize(tokens.float(), dim=-1)
    channels_first = normalized.permute(0, 3, 1, 2)
    neighbor_mean = functional.avg_pool2d(
        channels_first,
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=False,
    ).permute(0, 2, 3, 1)
    neighbor_mean = functional.normalize(neighbor_mean, dim=-1)
    local = 1.0 - (normalized * neighbor_mean).sum(dim=-1)
    global_mean = functional.normalize(
        normalized.mean(dim=(1, 2), keepdim=True),
        dim=-1,
    )
    global_score = 1.0 - (normalized * global_mean).sum(dim=-1)
    score = local * 0.7 + global_score * 0.3
    score_min = score.amin(dim=(1, 2), keepdim=True)
    score_max = score.amax(dim=(1, 2), keepdim=True)
    return (score - score_min) / torch.clamp(score_max - score_min, min=1e-8)


def load_or_extract_dino_scores(
    paths: Paths,
    backgrounds: list[dict[str, Any]],
    *,
    manifest_sha256: str,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], Path, str]:
    model_config = config["model"]
    cache_key = dino_cache_key(
        backgrounds,
        manifest_sha256=manifest_sha256,
        model_config=model_config,
    )
    cache_file = paths.cache / f"m9_dinov2_scores_{cache_key}.npz"
    expected_paths = [str(record["image_path"]) for record in backgrounds]
    if cache_file.is_file():
        with np.load(cache_file) as cached:
            cached_paths = [str(value) for value in cached["image_paths"].tolist()]
            scores = cached["scores"].astype(np.float32)
            revision = str(cached["model_revision"].item())
        if cached_paths != expected_paths or scores.shape[0] != len(backgrounds):
            raise ValidationError(f"Unexpected DINOv2 ROI cache contents: {cache_file}")
        LOGGER.info("Loaded %d cached DINOv2 ROI score maps", len(backgrounds))
        return dict(zip(cached_paths, scores, strict=True)), cache_file, revision

    if not torch.cuda.is_available():
        raise ValidationError("M9 DINOv2 ROI extraction requires CUDA")
    model_id = str(model_config["id"])
    revision = str(model_config["revision"])
    input_size = int(model_config["input_size"])
    patch_size = int(model_config["patch_size"])
    grid_size = input_size // patch_size
    batch_size = int(model_config["batch_size"])
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision).to("cuda").eval()
    observed_revision = str(getattr(model.config, "_commit_hash", None) or "unavailable")
    if observed_revision != revision:
        raise ValidationError(
            f"DINOv2 revision mismatch: expected {revision}, observed {observed_revision}"
        )

    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(backgrounds), batch_size):
            batch = backgrounds[start : start + batch_size]
            images: list[Image.Image] = []
            for record in batch:
                with Image.open(paths.visa_raw / record["image_path"]) as image_handle:
                    images.append(image_handle.convert("RGB"))
            inputs = processor(
                images=images,
                return_tensors="pt",
                do_center_crop=False,
                size={"height": input_size, "width": input_size},
            )
            pixel_values = inputs["pixel_values"].to("cuda")
            output = model(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
            if output.shape[1] != grid_size * grid_size:
                raise ValidationError(
                    f"Unexpected DINOv2 patch count: {output.shape[1]} "
                    f"(expected {grid_size * grid_size})"
                )
            token_grid = output.reshape(len(batch), grid_size, grid_size, output.shape[-1])
            batches.append(_token_scores(token_grid).cpu().numpy().astype(np.float16))
            LOGGER.info(
                "DINOv2 ROI scores: %d/%d",
                min(start + len(batch), len(backgrounds)),
                len(backgrounds),
            )
    scores = np.concatenate(batches, axis=0)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        image_paths=np.asarray(expected_paths),
        scores=scores,
        model_revision=np.asarray(observed_revision),
        cache_key=np.asarray(cache_key),
    )
    del model
    torch.cuda.empty_cache()
    return dict(zip(expected_paths, scores.astype(np.float32), strict=True)), cache_file, revision


def _remove_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    result = np.zeros(mask.shape, dtype=np.uint8)
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= minimum_area:
            result[labels == component] = 1
    return result.astype(bool)


def dino_structured_roi(
    score_grid: np.ndarray,
    image_shape: tuple[int, int],
    *,
    reference_roi: np.ndarray | None,
    score_quantile: float,
    close_kernel: int,
    open_kernel: int,
    dilation_px: int,
    border_fraction: float,
    min_component_area_ratio: float,
) -> np.ndarray:
    height, width = image_shape
    score_map = cv2.resize(
        score_grid.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    values = score_map[reference_roi] if reference_roi is not None else score_map.reshape(-1)
    if not len(values):
        raise ImagingError("DINOv2 ROI reference is empty")
    threshold = float(np.quantile(values, score_quantile))
    structured = score_map >= threshold
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    opened = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    structured = cv2.morphologyEx(
        structured.astype(np.uint8),
        cv2.MORPH_CLOSE,
        close,
    )
    structured = cv2.morphologyEx(structured, cv2.MORPH_OPEN, opened)
    if dilation_px:
        dilation = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_px * 2 + 1, dilation_px * 2 + 1),
        )
        structured = cv2.dilate(structured, dilation)
    border_y = max(1, round(height * border_fraction))
    border_x = max(1, round(width * border_fraction))
    structured[:border_y, :] = 0
    structured[-border_y:, :] = 0
    structured[:, :border_x] = 0
    structured[:, -border_x:] = 0
    minimum_area = max(32, round(height * width * min_component_area_ratio))
    result = _remove_small_components(structured > 0, minimum_area)
    if int(result.sum()) < minimum_area:
        raise ImagingError("DINOv2 structured ROI is empty")
    return result


def legal_roi(
    image: np.ndarray,
    score_grid: np.ndarray,
    *,
    config: dict[str, Any],
    object_name: str,
    method: str,
) -> np.ndarray:
    foreground = detect_legal_roi(
        image,
        **object_section(config, "foreground", object_name),
    )
    if method == "otsu":
        return foreground

    structured = dino_structured_roi(
        score_grid,
        foreground.shape,
        reference_roi=foreground if method == "intersect" else None,
        **object_section(config, "dinov2", object_name),
    )
    if method == "dinov2":
        result = structured
    elif method == "intersect":
        result = foreground & structured
    else:
        raise ValidationError(f"Unknown ROI method: {method}")
    if not np.any(result):
        raise ImagingError(f"{method} legal ROI is empty")
    return result


def load_component_mask(
    paths: Paths,
    component: dict[str, Any],
) -> np.ndarray:
    mask_path = paths.visa_raw / component["mask_path"]
    cache_key = (str(mask_path.resolve()), int(component["component_id"]))
    cached = COMPONENT_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with Image.open(mask_path) as mask_handle:
        full_mask = np.asarray(mask_handle.convert("L")) > 0
    components = label(full_mask, connectivity=2)
    component_mask = components == int(component["component_id"]) + 1
    if int(component_mask.sum()) != int(component["area_px"]):
        raise ValidationError(
            f"Frozen component changed: {component['mask_path']} #{component['component_id']}"
        )
    x, y, width, height = (int(value) for value in component["bbox"])
    cropped = component_mask[y : y + height, x : x + width]
    COMPONENT_MASK_CACHE[cache_key] = cropped
    return cropped


def transform_mask(
    source_mask: np.ndarray,
    *,
    rotation_deg: float,
    scale: float,
    flip: bool,
) -> np.ndarray:
    mask = np.ascontiguousarray(np.fliplr(source_mask)) if flip else source_mask
    height, width = mask.shape
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, rotation_deg, scale)
    cosine = abs(float(matrix[0, 0]))
    sine = abs(float(matrix[0, 1]))
    output_width = max(1, math.ceil(height * sine + width * cosine))
    output_height = max(1, math.ceil(height * cosine + width * sine))
    matrix[0, 2] += output_width / 2 - center[0]
    matrix[1, 2] += output_height / 2 - center[1]
    transformed = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if not np.any(transformed):
        raise ImagingError("Mask affine produced an empty result")
    x, y, box_width, box_height = mask_bbox(transformed > 0)
    return transformed[y : y + box_height, x : x + box_width] > 0


def shape_bounds(
    paths: Paths,
    objects: tuple[str, ...],
    *,
    manifest_sha256: str,
) -> dict[str, dict[str, tuple[float, float]]]:
    stats = load_json(paths.reports / "real_mask_stats.json")
    if stats.get("manifest_sha256") != manifest_sha256:
        raise ValidationError("Real mask statistics point to another manifest")
    expected_selection_sha256, selection_filename = read_checksum_file(
        paths.splits / "FEWSHOT_SELECTION.sha256"
    )
    if sha256_file(paths.splits / selection_filename) != expected_selection_sha256:
        raise ValidationError("Few-shot selection checksum mismatch")
    if stats.get("selection_sha256") != expected_selection_sha256:
        raise ValidationError("Real mask statistics point to another selection")
    return {
        object_name: {
            metric: (
                float(stats["objects"][object_name]["summary"][metric]["p05"]),
                float(stats["objects"][object_name]["summary"][metric]["p95"]),
            )
            for metric in ("area_ratio", "aspect_ratio")
        }
        for object_name in objects
    }


def transformed_component(
    components: list[dict[str, Any]],
    rng: np.random.Generator,
    *,
    paths: Paths,
    image_shape: tuple[int, int],
    bounds: dict[str, tuple[float, float]],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], float, float, bool, int]:
    image_area = image_shape[0] * image_shape[1]
    transform_config = config["transform"]
    min_scale, max_scale = (float(value) for value in transform_config["scale"])
    failed = 0
    for _ in range(int(transform_config["max_transform_tries"])):
        component = components[int(rng.integers(0, len(components)))]
        source_area = float(component["area_px"])
        feasible_low = max(
            bounds["area_ratio"][0] * image_area,
            source_area * min_scale**2,
        )
        feasible_high = min(
            bounds["area_ratio"][1] * image_area,
            source_area * max_scale**2,
        )
        if feasible_high <= feasible_low:
            failed += 1
            continue
        target_area = log_uniform(rng, feasible_low, feasible_high)
        scale = float(np.sqrt(target_area / source_area))
        rotation = range_value(rng, transform_config["rotation_deg"])
        flip = bool(rng.random() < float(transform_config["flip_probability"]))
        mask = transform_mask(
            load_component_mask(paths, component),
            rotation_deg=rotation,
            scale=scale,
            flip=flip,
        )
        _, _, width, height = mask_bbox(mask)
        area_ratio = float(mask.sum() / image_area)
        aspect_ratio = float(width / height)
        if (
            bounds["area_ratio"][0] <= area_ratio <= bounds["area_ratio"][1]
            and bounds["aspect_ratio"][0] <= aspect_ratio <= bounds["aspect_ratio"][1]
        ):
            return mask, component, rotation, scale, flip, failed
        failed += 1
    raise ImagingError("Could not transform a source component into real-statistic bounds")


def existing_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                validate_placement_record(record)
            except (json.JSONDecodeError, ValidationError) as error:
                raise ValidationError(f"{path}:{line_number}: {error}") from error
            if record["placement_id"] in result:
                raise ValidationError(f"Duplicate placement_id: {record['placement_id']}")
            result[record["placement_id"]] = record
    return result


def output_mask_valid(
    path: Path,
    *,
    blocklist: set[str],
) -> np.ndarray:
    with Image.open(path) as mask_handle:
        mask = np.asarray(mask_handle.convert("L"))
    if not np.any(mask) or not {int(value) for value in np.unique(mask)}.issubset({0, 255}):
        raise ValidationError(f"Invalid placement mask: {path}")
    if sha256_file(path) in blocklist:
        raise ValidationError(f"Placement mask hit test blocklist: {path}")
    return mask > 0


def render_visualization(
    paths: Paths,
    object_name: str,
    records: list[dict[str, Any]],
    *,
    output_root: Path,
    score_maps: dict[str, np.ndarray],
    config: dict[str, Any],
    method: str,
    count: int,
    columns: int,
    destination: Path,
) -> None:
    selected_indices = np.linspace(
        0,
        len(records) - 1,
        num=min(count, len(records)),
        dtype=int,
    )
    selected = [records[int(index)] for index in selected_indices]
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.5 * columns, 2.8 * rows),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(-1)
    for axis, record in zip(axes_array, selected, strict=False):
        with Image.open(paths.visa_raw / record["background_image"]) as image_handle:
            image = np.asarray(image_handle.convert("RGB"))
        with Image.open(output_root / object_name / record["mask_path"]) as mask_handle:
            mask = np.asarray(mask_handle.convert("L")) > 0
        roi = legal_roi(
            image,
            score_maps[record["background_image"]],
            config=config,
            object_name=object_name,
            method=method,
        )
        axis.imshow(image)
        axis.contour(roi.astype(np.uint8), levels=[0.5], colors=["#00e5ff"], linewidths=0.8)
        axis.contour(mask.astype(np.uint8), levels=[0.5], colors=["#ff2d55"], linewidths=1.8)
        axis.set_title(
            f"{record['defect_type']} | v{record['variant_index']}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes_array[len(selected) :]:
        axis.axis("off")
    figure.suptitle(
        f"{object_name}: M9 {method} ROI (cyan) + placed mask (red)",
        fontsize=15,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, facecolor="white")
    plt.close(figure)


def write_report(
    destination: Path,
    *,
    records_by_object: dict[str, list[dict[str, Any]]],
    quotas_by_object: dict[str, dict[int, int]],
    roi_method: str,
    dino_revision: str,
) -> None:
    lines = [
        "# M9 Mask Placement Report",
        "",
        f"- ROI method: `{roi_method}`",
        f"- DINOv2 revision: `{dino_revision}`",
        "- Every mask is binary, non-empty, inside legal ROI, within frozen real-stat bounds,",
        "  test-blocklist clean, and non-overlapping with sibling variants on its background.",
        "",
        "| object | records | backgrounds | type quotas | failed transform/place attempts |",
        "|---|---:|---:|---|---:|",
    ]
    for object_name, records in records_by_object.items():
        backgrounds = {record["background_image"] for record in records}
        failures = sum(int(record["failed_attempts"]) for record in records)
        quota_text = ", ".join(
            f"type{cluster_id}={quota}"
            for cluster_id, quota in quotas_by_object[object_name].items()
        )
        lines.append(
            f"| {object_name} | {len(records)} | {len(backgrounds)} | {quota_text} | {failures} |"
        )
    lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.n_per_image < 1 or args.max_place_tries < 1:
            raise ValidationError("n-per-image and max-place-tries must be positive")
        if args.limit_backgrounds is not None and args.limit_backgrounds < 1:
            raise ValidationError("limit-backgrounds must be positive")
        if not OUTPUT_NAME.fullmatch(args.out_name):
            raise ValidationError(f"Unsafe out-name: {args.out_name!r}")
        paths = load_paths(args.paths)
        config = load_config(args.config)
        objects = tuple(args.objects or paths.objects)
        seed = paths.seed if args.seed is None else args.seed
        manifest, manifest_sha256 = verify_frozen_manifest(paths.splits)
        defect_types_path = paths.splits / "defect_types.json"
        defect_types_sha256 = sha256_file(defect_types_path)
        defect_types = load_json(defect_types_path)
        if defect_types.get("manifest_sha256") != manifest_sha256:
            raise ValidationError("Defect types point to another manifest")
        bounds_by_object = shape_bounds(
            paths,
            objects,
            manifest_sha256=manifest_sha256,
        )
        backgrounds_by_object: dict[str, list[dict[str, Any]]] = {}
        for object_name in objects:
            records = sorted(
                [
                    record
                    for record in manifest["images"]
                    if record["object"] == object_name
                    and record["set"] == "train"
                    and record["label"] == "good"
                ],
                key=lambda record: record["image_path"],
            )
            if args.limit_backgrounds is not None:
                records = records[: args.limit_backgrounds]
            backgrounds_by_object[object_name] = records
        components_by_object = {
            object_name: {
                int(type_record["cluster_id"]): list(type_record["components"])
                for type_record in defect_types["objects"][object_name]["types"]
            }
            for object_name in objects
        }
        input_files = [
            paths.visa_raw / record["image_path"]
            for records in backgrounds_by_object.values()
            for record in records
        ]
        input_files.extend(
            paths.visa_raw / component[key]
            for type_records in components_by_object.values()
            for components in type_records.values()
            for component in components
            for key in ("image_path", "mask_path")
        )
        assert_not_blocklisted(input_files, paths.splits / "test_blocklist.json")
        blocklist = set(load_json(paths.splits / "test_blocklist.json")["sha256"])
        quotas_by_object: dict[str, dict[int, int]] = {}
        schedules_by_object: dict[str, list[int]] = {}
        for object_name in objects:
            total = len(backgrounds_by_object[object_name]) * args.n_per_image
            sizes = {
                int(type_record["cluster_id"]): int(type_record["n_components"])
                for type_record in defect_types["objects"][object_name]["types"]
            }
            minimum = 50 if total >= 50 * len(sizes) else 1
            quotas_by_object[object_name] = allocate_quotas(sizes, total, minimum=minimum)
            schedules_by_object[object_name] = weighted_schedule(quotas_by_object[object_name])
        LOGGER.info("M9 quotas: %s", quotas_by_object)
        if args.dry_run:
            return 0

        output_root = paths.synthetic / args.out_name
        if output_root.exists() and not args.resume:
            raise ValidationError(
                f"Output exists; use --resume or choose another --out-name: {output_root}"
            )
        all_backgrounds = [
            record for object_name in objects for record in backgrounds_by_object[object_name]
        ]
        if args.roi_method == "otsu":
            score_maps = {
                str(record["image_path"]): np.zeros((1, 1), dtype=np.float32)
                for record in all_backgrounds
            }
            cache_file = None
            dino_revision = None
        else:
            score_maps, cache_file, dino_revision = load_or_extract_dino_scores(
                paths,
                all_backgrounds,
                manifest_sha256=manifest_sha256,
                config=config,
            )
        records_by_object: dict[str, list[dict[str, Any]]] = {}
        viz_n = int(args.viz_n or config["report"]["viz_n"])
        for object_name in objects:
            object_root = output_root / object_name
            mask_dir = object_root / "masks"
            metadata_path = object_root / "placements.jsonl"
            mask_dir.mkdir(parents=True, exist_ok=True)
            existing = existing_records(metadata_path)
            expected_count = len(backgrounds_by_object[object_name]) * args.n_per_image
            snapshot = {
                "seed": seed,
                "n_per_image": args.n_per_image,
                "roi_method": args.roi_method,
                "max_place_tries": args.max_place_tries,
                "backgrounds": len(backgrounds_by_object[object_name]),
                "expected_records": expected_count,
                "type_quotas": quotas_by_object[object_name],
                "manifest_sha256": manifest_sha256,
                "defect_types_sha256": defect_types_sha256,
                "dino_cache_file": str(cache_file) if cache_file is not None else None,
                "dino_revision": dino_revision,
                "placement_config": config,
            }
            run_config_path = object_root / "run_config.json"
            normalized_snapshot = json.loads(json.dumps(snapshot, sort_keys=True))
            if run_config_path.is_file():
                if load_json(run_config_path) != normalized_snapshot:
                    raise ValidationError(f"Resume parameters differ from {run_config_path}")
            elif existing:
                raise ValidationError(f"Cannot resume metadata without run config: {metadata_path}")
            else:
                run_config_path.write_text(
                    json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            schedule = schedules_by_object[object_name]
            types = {
                int(type_record["cluster_id"]): type_record
                for type_record in defect_types["objects"][object_name]["types"]
            }
            object_records: list[dict[str, Any]] = []
            with metadata_path.open("a", encoding="utf-8", newline="\n") as metadata_handle:
                for background_index, background_record in enumerate(
                    backgrounds_by_object[object_name]
                ):
                    with Image.open(
                        paths.visa_raw / background_record["image_path"]
                    ) as image_handle:
                        image = np.asarray(image_handle.convert("RGB"))
                    roi = legal_roi(
                        image,
                        score_maps[background_record["image_path"]],
                        config=config,
                        object_name=object_name,
                        method=args.roi_method,
                    )
                    occupied = np.zeros(roi.shape, dtype=bool)
                    background_stem = Path(background_record["image_path"]).stem
                    for variant_index in range(args.n_per_image):
                        global_index = background_index * args.n_per_image + variant_index
                        cluster_id = schedule[global_index]
                        placement_id = (
                            f"{object_name}__type{cluster_id}__{background_stem}"
                            f"__{variant_index:02d}"
                        )
                        sample_seed = placement_seed(seed, object_name, global_index)
                        if placement_id in existing:
                            record = existing[placement_id]
                            if (
                                record["object"] != object_name
                                or record["defect_type"] != f"type{cluster_id}"
                                or record["background_image"] != background_record["image_path"]
                                or int(record["variant_index"]) != variant_index
                                or record["roi_method"] != args.roi_method
                                or int(record["seed"]) != sample_seed
                            ):
                                raise ValidationError(
                                    f"Resumed placement metadata mismatch: {placement_id}"
                                )
                            mask = output_mask_valid(
                                object_root / record["mask_path"],
                                blocklist=blocklist,
                            )
                            if bool(np.any(mask & occupied)):
                                raise ValidationError(
                                    f"Resumed masks overlap on {background_record['image_path']}"
                                )
                            if not bool(np.all(roi[mask])):
                                raise ValidationError(f"Resumed mask escapes ROI: {placement_id}")
                            occupied |= mask
                            object_records.append(record)
                            continue

                        rng = placement_rng(seed, object_name, global_index)
                        failed_attempts = 0
                        placed: np.ndarray | None = None
                        component: dict[str, Any] | None = None
                        rotation = 0.0
                        scale = 1.0
                        flip = False
                        for _ in range(int(config["transform"]["max_transform_tries"])):
                            try:
                                (
                                    local_mask,
                                    candidate_component,
                                    candidate_rotation,
                                    candidate_scale,
                                    candidate_flip,
                                    transform_failures,
                                ) = transformed_component(
                                    components_by_object[object_name][cluster_id],
                                    rng,
                                    paths=paths,
                                    image_shape=roi.shape,
                                    bounds=bounds_by_object[object_name],
                                    config=config,
                                )
                                failed_attempts += transform_failures
                                clearance = int(config["transform"]["occupied_clearance_px"])
                                occupied_buffer = occupied.astype(np.uint8)
                                if clearance:
                                    kernel = cv2.getStructuringElement(
                                        cv2.MORPH_ELLIPSE,
                                        (clearance * 2 + 1, clearance * 2 + 1),
                                    )
                                    occupied_buffer = cv2.dilate(occupied_buffer, kernel)
                                available = roi & ~occupied_buffer.astype(bool)
                                x0, y0 = place_on_legal_roi(
                                    local_mask,
                                    available,
                                    rng,
                                    max_tries=args.max_place_tries,
                                )
                            except ImagingError:
                                failed_attempts += 1
                                continue
                            placed = np.zeros(roi.shape, dtype=np.uint8)
                            local_height, local_width = local_mask.shape
                            placed[
                                y0 : y0 + local_height,
                                x0 : x0 + local_width,
                            ] = local_mask.astype(np.uint8) * 255
                            component = candidate_component
                            rotation = candidate_rotation
                            scale = candidate_scale
                            flip = candidate_flip
                            break
                        if placed is None or component is None:
                            raise ValidationError(f"Could not place {placement_id} after retries")
                        placed_bool = placed > 0
                        if not bool(np.all(roi[placed_bool])) or bool(
                            np.any(placed_bool & occupied)
                        ):
                            raise ValidationError(f"Invalid final placement: {placement_id}")
                        destination_bbox = list(mask_bbox(placed_bool))
                        area = int(placed_bool.sum())
                        area_ratio = float(area / placed_bool.size)
                        aspect_ratio = float(destination_bbox[2] / destination_bbox[3])
                        bounds = bounds_by_object[object_name]
                        if not (
                            bounds["area_ratio"][0] <= area_ratio <= bounds["area_ratio"][1]
                            and bounds["aspect_ratio"][0]
                            <= aspect_ratio
                            <= bounds["aspect_ratio"][1]
                        ):
                            raise ValidationError(
                                f"Final placement is outside real-stat bounds: {placement_id}"
                            )
                        filename = f"{placement_id}.png"
                        mask_path = mask_dir / filename
                        Image.fromarray(placed).save(
                            mask_path,
                            format="PNG",
                            compress_level=6,
                        )
                        output_mask_valid(mask_path, blocklist=blocklist)
                        record = {
                            "placement_id": placement_id,
                            "object": object_name,
                            "defect_type": f"type{cluster_id}",
                            "trigger_token": types[cluster_id]["trigger_token"],
                            "variant_index": variant_index,
                            "background_image": background_record["image_path"],
                            "background_sha256": background_record["sha256"],
                            "source_image": component["image_path"],
                            "source_mask": component["mask_path"],
                            "source_component_id": component["component_id"],
                            "mask_path": f"masks/{filename}",
                            "roi_method": args.roi_method,
                            "roi_bbox": list(mask_bbox(roi)),
                            "mask_bbox": destination_bbox,
                            "affine": {
                                "dx": destination_bbox[0] - int(component["bbox"][0]),
                                "dy": destination_bbox[1] - int(component["bbox"][1]),
                                "rotation_deg": rotation,
                                "scale": scale,
                                "flip": flip,
                            },
                            "mask_area_px": area,
                            "mask_area_ratio": area_ratio,
                            "aspect_ratio": aspect_ratio,
                            "failed_attempts": failed_attempts,
                            "seed": sample_seed,
                            "pipeline_version": PIPELINE_VERSION,
                            "created_at": datetime.now(UTC).isoformat(),
                        }
                        validate_placement_record(record)
                        metadata_handle.write(json.dumps(record, sort_keys=True) + "\n")
                        metadata_handle.flush()
                        occupied |= placed_bool
                        object_records.append(record)
                    if (background_index + 1) % 25 == 0 or background_index + 1 == len(
                        backgrounds_by_object[object_name]
                    ):
                        LOGGER.info(
                            "%s: completed %d/%d backgrounds",
                            object_name,
                            background_index + 1,
                            len(backgrounds_by_object[object_name]),
                        )
            if len(object_records) != expected_count:
                raise ValidationError(
                    f"{object_name} placement count mismatch: "
                    f"{len(object_records)} != {expected_count}"
                )
            resumed_extras = set(existing) - {
                str(record["placement_id"]) for record in object_records
            }
            if resumed_extras:
                raise ValidationError(
                    f"{object_name} resume has {len(resumed_extras)} unexpected records"
                )
            observed_types = Counter(
                int(str(record["defect_type"]).removeprefix("type")) for record in object_records
            )
            if observed_types != Counter(quotas_by_object[object_name]):
                raise ValidationError(f"{object_name} placement quota mismatch: {observed_types}")
            records_by_object[object_name] = object_records
            render_visualization(
                paths,
                object_name,
                object_records,
                output_root=output_root,
                score_maps=score_maps,
                config=config,
                method=args.roi_method,
                count=viz_n,
                columns=int(config["report"]["columns"]),
                destination=paths.figures
                / (
                    f"placement_check_{object_name}.png"
                    if args.out_name == "placements"
                    else f"{args.out_name}_check_{object_name}.png"
                ),
            )
        write_report(
            paths.reports / f"{args.out_name}_report.md",
            records_by_object=records_by_object,
            quotas_by_object=quotas_by_object,
            roi_method=args.roi_method,
            dino_revision=dino_revision or "not used",
        )
        _, final_manifest_sha256 = verify_frozen_manifest(paths.splits)
        if final_manifest_sha256 != manifest_sha256:
            raise ValidationError("Frozen manifest changed during M9")
        if sha256_file(defect_types_path) != defect_types_sha256:
            raise ValidationError("Frozen defect types changed during M9")
        LOGGER.info(
            "M9 generated %d validated placements in %s",
            sum(len(records) for records in records_by_object.values()),
            output_root,
        )
        return 0
    except (
        ImagingError,
        IntegrityError,
        OSError,
        ValidationError,
        json.JSONDecodeError,
    ) as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
