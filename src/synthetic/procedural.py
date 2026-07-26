"""Generate deterministic procedural defects on frozen train-good backgrounds."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.imaging import (
    ImagingError,
    detect_legal_roi,
    mask_bbox,
    place_on_legal_roi,
)
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
    existing_records,
    load_config,
    object_code,
    object_roi_config,
    range_value,
    validate_output_pair,
)
from src.synthetic.metadata import MetadataError, validate_metadata

LOGGER = logging.getLogger("procedural")
PIPELINE_VERSION = "0.1.0"
SUPPORTED_SHAPES = ("perlin", "crack", "scratch", "spot")
SAFE_SHAPE = re.compile(r"^[a-z][a-z0-9_-]*$")


class StatsLeakageError(ValidationError):
    """The no-real-stats branch attempted to open real mask statistics."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument("--config", default="configs/stage_a.yaml", type=Path)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--shapes")
    parser.add_argument("--no-real-stats", action="store_true")
    parser.add_argument("--out-name")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def parse_shapes(value: str) -> tuple[str, ...]:
    shapes = tuple(dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip()))
    if not shapes:
        raise ValidationError("At least one procedural shape is required")
    unknown = sorted(set(shapes) - set(SUPPORTED_SHAPES))
    if unknown or not all(SAFE_SHAPE.fullmatch(shape) for shape in shapes):
        raise ValidationError(f"Unsupported procedural shapes: {unknown or shapes}")
    return shapes


def procedural_rng(seed: int, object_name: str, index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [seed, object_code(f"{object_name}:procedural"), index],
    )
    return np.random.Generator(np.random.PCG64(sequence))


def even_schedule(items: tuple[str, ...], total: int) -> list[str]:
    if total < len(items):
        raise ValidationError(
            f"Cannot allocate total={total} while covering {len(items)} procedural shapes"
        )
    base, remainder = divmod(total, len(items))
    return [
        item
        for item_index, item in enumerate(items)
        for _ in range(base + int(item_index < remainder))
    ]


def install_forbidden_stats_guard(path: Path) -> None:
    """Fail the process if Python attempts to open the forbidden stats file."""

    forbidden = path.resolve(strict=False)

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args or not isinstance(args[0], (str, bytes)):
            return
        candidate = Path(os.fsdecode(args[0])).resolve(strict=False)
        if candidate == forbidden:
            raise StatsLeakageError(f"Forbidden real-statistics access: {forbidden}")

    sys.addaudithook(audit)


def load_shape_bounds(
    *,
    paths: Paths,
    config: dict[str, Any],
    objects: tuple[str, ...],
    no_real_stats: bool,
    manifest_sha256: str,
) -> dict[str, dict[str, tuple[float, float]]]:
    if no_real_stats:
        fixed = config["procedural"]["fixed_bounds"]
        return {
            object_name: {
                metric: (float(values[0]), float(values[1]))
                for metric, values in fixed[object_name].items()
            }
            for object_name in objects
        }

    stats_path = paths.reports / "real_mask_stats.json"
    stats = load_json(stats_path)
    if stats.get("manifest_sha256") != manifest_sha256:
        raise ValidationError("Real mask statistics point to another manifest")
    expected_selection_sha256, selection_filename = read_checksum_file(
        paths.splits / "FEWSHOT_SELECTION.sha256"
    )
    if selection_filename != "fewshot_selection.json":
        raise ValidationError("Few-shot checksum sidecar names an unexpected file")
    if sha256_file(paths.splits / selection_filename) != expected_selection_sha256:
        raise ValidationError("Few-shot selection checksum mismatch")
    if stats.get("selection_sha256") != expected_selection_sha256:
        raise ValidationError("Real mask statistics point to another few-shot selection")
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


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        raise ImagingError("Procedural motif has no foreground")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _draw_polyline(
    canvas: np.ndarray,
    points: list[tuple[int, int]],
    *,
    thickness: int,
) -> None:
    cv2.polylines(
        canvas,
        [np.asarray(points, dtype=np.int32)],
        isClosed=False,
        color=255,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def base_motif(shape: str, rng: np.random.Generator, size: int = 128) -> np.ndarray:
    """Create one normalized binary motif before target-statistic resizing."""

    canvas = np.zeros((size, size), dtype=np.uint8)
    if shape == "perlin":
        noise = np.zeros((size, size), dtype=np.float32)
        weight_total = 0.0
        for grid_size, weight in ((4, 1.0), (8, 0.55), (16, 0.3), (32, 0.15)):
            grid = rng.normal(size=(grid_size, grid_size)).astype(np.float32)
            noise += cv2.resize(grid, (size, size), interpolation=cv2.INTER_CUBIC) * weight
            weight_total += weight
        noise /= weight_total
        threshold = float(np.quantile(noise, rng.uniform(0.62, 0.75)))
        canvas[noise >= threshold] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
        return _largest_component(canvas > 0)

    if shape == "crack":
        y = int(rng.integers(size // 4, size * 3 // 4))
        points = [(0, y)]
        segments = int(rng.integers(7, 12))
        for step in range(1, segments):
            x = round(step * (size - 1) / (segments - 1))
            y = int(np.clip(y + rng.integers(-18, 19), 2, size - 3))
            points.append((x, y))
        _draw_polyline(canvas, points, thickness=int(rng.integers(1, 4)))
        for _ in range(int(rng.integers(2, 5))):
            anchor_index = int(rng.integers(2, len(points) - 2))
            x0, y0 = points[anchor_index]
            branch = [(x0, y0)]
            for step in range(1, int(rng.integers(3, 6))):
                branch.append(
                    (
                        int(np.clip(x0 + step * rng.integers(5, 12), 0, size - 1)),
                        int(np.clip(y0 + step * rng.integers(-12, 13), 0, size - 1)),
                    )
                )
            _draw_polyline(canvas, branch, thickness=1)
        return canvas > 0

    if shape == "scratch":
        angle = float(rng.uniform(-0.35, 0.35))
        count = int(rng.integers(2, 6))
        center = size // 2
        for line_index in range(count):
            offset = round((line_index - (count - 1) / 2) * rng.uniform(4.0, 9.0))
            y0 = int(np.clip(center + offset - np.tan(angle) * center, 0, size - 1))
            y1 = int(np.clip(center + offset + np.tan(angle) * center, 0, size - 1))
            _draw_polyline(
                canvas,
                [(0, y0), (size - 1, y1)],
                thickness=int(rng.integers(1, 4)),
            )
        return canvas > 0

    if shape == "spot":
        center = (
            int(rng.integers(size * 2 // 5, size * 3 // 5)),
            int(rng.integers(size * 2 // 5, size * 3 // 5)),
        )
        axes = (
            int(rng.integers(size // 4, size * 2 // 5)),
            int(rng.integers(size // 4, size * 2 // 5)),
        )
        cv2.ellipse(
            canvas,
            center,
            axes,
            float(rng.uniform(0, 180)),
            0,
            360,
            255,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        for _ in range(int(rng.integers(3, 7))):
            satellite = (
                int(np.clip(center[0] + rng.normal(0, axes[0] * 0.7), 0, size - 1)),
                int(np.clip(center[1] + rng.normal(0, axes[1] * 0.7), 0, size - 1)),
            )
            radius = int(rng.integers(max(2, size // 24), max(3, size // 10)))
            cv2.circle(canvas, satellite, radius, 255, thickness=-1)
        return _largest_component(canvas > 0)

    raise ImagingError(f"Unknown procedural shape: {shape}")


def resize_motif(
    motif: np.ndarray,
    *,
    target_area: float,
    target_aspect: float,
    image_shape: tuple[int, int],
) -> np.ndarray:
    x, y, width, height = mask_bbox(motif)
    cropped = motif[y : y + height, x : x + width].astype(np.uint8)
    occupancy = float(cropped.mean())
    box_area = max(target_area / max(occupancy, 1e-6), 9.0)
    target_height = max(3, round(np.sqrt(box_area / target_aspect)))
    target_width = max(3, round(target_height * target_aspect))
    max_height, max_width = image_shape
    scale_limit = min(max_width / target_width, max_height / target_height, 1.0)
    target_width = max(3, round(target_width * scale_limit))
    target_height = max(3, round(target_height * scale_limit))
    resized = cropped
    for _ in range(4):
        resized = cv2.resize(
            cropped,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        observed_area = max(int(np.count_nonzero(resized)), 1)
        correction = float(np.sqrt(target_area / observed_area))
        if 0.97 <= correction <= 1.03:
            break
        target_width = int(np.clip(round(target_width * correction), 3, max_width))
        target_height = int(np.clip(round(target_height * correction), 3, max_height))
    return resized > 0


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    if low <= 0 or high <= low:
        raise ValidationError(f"Invalid positive range: {(low, high)}")
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def generate_local_mask(
    shape: str,
    rng: np.random.Generator,
    *,
    image_shape: tuple[int, int],
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    image_area = image_shape[0] * image_shape[1]
    for _ in range(24):
        area_ratio = log_uniform(rng, *bounds["area_ratio"])
        aspect_ratio = log_uniform(rng, *bounds["aspect_ratio"])
        motif = base_motif(shape, rng)
        mask = resize_motif(
            motif,
            target_area=area_ratio * image_area,
            target_aspect=aspect_ratio,
            image_shape=image_shape,
        )
        _, _, width, height = mask_bbox(mask)
        observed_area_ratio = float(mask.sum() / image_area)
        observed_aspect_ratio = float(width / height)
        if (
            bounds["area_ratio"][0] <= observed_area_ratio <= bounds["area_ratio"][1]
            and bounds["aspect_ratio"][0] <= observed_aspect_ratio <= bounds["aspect_ratio"][1]
        ):
            return mask
    raise ImagingError(f"Could not satisfy procedural bounds for shape={shape}")


def apply_procedural_effect(
    background: np.ndarray,
    placed_mask: np.ndarray,
    shape: str,
    rng: np.random.Generator,
    *,
    opacity: float,
    blur_sigma: float,
    dark_probability: float,
) -> np.ndarray:
    mask_float = (placed_mask > 0).astype(np.float32)
    alpha = cv2.GaussianBlur(mask_float, (0, 0), sigmaX=max(blur_sigma, 0.05))
    alpha = np.clip(alpha * opacity, 0.0, 1.0)[..., None]
    height, width = placed_mask.shape
    texture_small = rng.normal(size=(max(2, height // 32), max(2, width // 32))).astype(np.float32)
    texture = cv2.resize(texture_small, (width, height), interpolation=cv2.INTER_CUBIC)
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=3.0)
    texture /= max(float(texture.std()), 1e-6)
    original = background.astype(np.float32)
    dark = bool(rng.random() < dark_probability) or shape in {"crack", "scratch"}
    if dark:
        factor = np.clip(rng.uniform(0.18, 0.6) + texture * 0.06, 0.08, 0.72)[..., None]
        effected = original * factor
    else:
        tint = np.asarray(
            [
                rng.uniform(80, 220),
                rng.uniform(35, 180),
                rng.uniform(20, 150),
            ],
            dtype=np.float32,
        )
        mix = np.clip(rng.uniform(0.35, 0.75) + texture * 0.05, 0.2, 0.85)[..., None]
        effected = original * (1.0 - mix) + tint * mix
    result = original * (1.0 - alpha) + effected * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def generate_one(
    *,
    paths: Paths,
    object_name: str,
    index: int,
    shape: str,
    backgrounds: list[dict[str, Any]],
    bounds: dict[str, tuple[float, float]],
    config: dict[str, Any],
    seed: int,
    image_dir: Path,
    mask_dir: Path,
    blocklist: set[str],
    stats_mode: str,
) -> dict[str, Any]:
    rng = procedural_rng(seed, object_name, index)
    procedural_config = config["procedural"]
    selected_background: dict[str, Any] | None = None
    background: np.ndarray | None = None
    legal_roi: np.ndarray | None = None
    local_mask: np.ndarray | None = None
    placement: tuple[int, int] | None = None
    for _ in range(int(procedural_config["max_shape_tries"])):
        candidate = backgrounds[int(rng.integers(0, len(backgrounds)))]
        with Image.open(paths.visa_raw / candidate["image_path"]) as image_handle:
            candidate_image = np.asarray(image_handle.convert("RGB"))
        try:
            candidate_roi = detect_legal_roi(
                candidate_image,
                **object_roi_config(config, object_name),
            )
            candidate_mask = generate_local_mask(
                shape,
                rng,
                image_shape=candidate_roi.shape,
                bounds=bounds,
            )
            candidate_placement = place_on_legal_roi(
                candidate_mask,
                candidate_roi,
                rng,
                max_tries=int(procedural_config["max_place_tries"]),
            )
        except ImagingError:
            continue
        selected_background = candidate
        background = candidate_image
        legal_roi = candidate_roi
        local_mask = candidate_mask
        placement = candidate_placement
        break
    if (
        selected_background is None
        or background is None
        or legal_roi is None
        or local_mask is None
        or placement is None
    ):
        raise ValidationError(f"Could not place procedural {shape} for {object_name} #{index}")

    x0, y0 = placement
    placed_mask = np.zeros(legal_roi.shape, dtype=np.uint8)
    local_height, local_width = local_mask.shape
    placed_mask[y0 : y0 + local_height, x0 : x0 + local_width] = local_mask.astype(np.uint8) * 255
    if not bool(np.all(legal_roi[placed_mask > 0])):
        raise ValidationError("Procedural mask is not 100% contained in legal ROI")
    synthetic = apply_procedural_effect(
        background,
        placed_mask,
        shape,
        rng,
        opacity=range_value(rng, procedural_config["opacity"]),
        blur_sigma=range_value(rng, procedural_config["blur_sigma"]),
        dark_probability=float(procedural_config["dark_probability"]),
    )

    sample_id = f"{object_name}__procedural_{shape}__{index:05d}"
    filename = f"{sample_id}.png"
    image_path = image_dir / filename
    mask_path = mask_dir / filename
    Image.fromarray(synthetic).save(image_path, format="PNG", compress_level=6)
    Image.fromarray(placed_mask).save(mask_path, format="PNG", compress_level=6)
    validate_output_pair(image_path, mask_path, blocklist=blocklist)
    destination_bbox = list(mask_bbox(placed_mask > 0))
    area = int(np.count_nonzero(placed_mask))
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "object": object_name,
        "defect_type": shape,
        "trigger_token": f"<{object_name}-procedural-{shape}>",
        "generator": "stageA_procedural",
        "bucket": stats_mode,
        "image_path": f"images/{filename}",
        "mask_path": f"masks/{filename}",
        "source": {
            "background_image": selected_background["image_path"],
            "background_sha256": selected_background["sha256"],
            "defect_source_image": None,
            "defect_source_mask": None,
            "defect_source_component_id": None,
        },
        "placement": {
            "roi_bbox": list(mask_bbox(legal_roi)),
            "mask_bbox": destination_bbox,
            "affine": {
                "dx": x0,
                "dy": y0,
                "rotation_deg": 0.0,
                "scale": 1.0,
                "flip": False,
            },
            "mask_area_px": area,
            "mask_area_ratio": area / placed_mask.size,
        },
        "generation": {
            "seed": int(
                np.random.SeedSequence(
                    [seed, object_code(f"{object_name}:procedural"), index]
                ).generate_state(1, dtype=np.uint64)[0]
            ),
            "base_model": "numpy-opencv-procedural",
            "lora_path": None,
            "prompt": None,
            "negative_prompt": None,
            "guidance_scale": None,
            "num_inference_steps": None,
            "strength": None,
            "crop_ratio": None,
            "crop_bbox": None,
            "model_resolution": None,
            "blend": "procedural_alpha",
        },
        "filter": None,
        "pipeline_version": PIPELINE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
    }
    validate_metadata(record)
    return record


def render_contact_sheet(
    output_root: Path,
    object_name: str,
    records: list[dict[str, Any]],
    destination: Path,
    *,
    count: int,
    columns: int,
    stats_mode: str,
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
        figsize=(3.4 * columns, 2.7 * rows),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(-1)
    for axis, record in zip(axes_array, selected, strict=False):
        with Image.open(output_root / record["image_path"]) as image_handle:
            image = np.asarray(image_handle.convert("RGB"))
        with Image.open(output_root / record["mask_path"]) as mask_handle:
            mask = np.asarray(mask_handle.convert("L")) > 0
        axis.imshow(image)
        axis.contour(mask.astype(np.uint8), levels=[0.5], colors=["#ff2d55"], linewidths=1.5)
        axis.set_title(record["defect_type"], fontsize=9)
        axis.axis("off")
    for axis in axes_array[len(selected) :]:
        axis.axis("off")
    figure.suptitle(
        f"{object_name}: Stage A procedural / {stats_mode} (red = mask)",
        fontsize=15,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, facecolor="white")
    plt.close(figure)


def write_report(
    destination: Path,
    *,
    output_name: str,
    records: list[dict[str, Any]],
    shapes: tuple[str, ...],
    stats_mode: str,
    bounds: dict[str, dict[str, tuple[float, float]]],
    stats_guard_enabled: bool,
) -> None:
    lines = [
        "# Stage A Procedural Generation Report",
        "",
        f"- Output: `${{synthetic}}/{output_name}`",
        f"- Stats mode: `{stats_mode}`",
        f"- Runtime real-stats access guard: `{stats_guard_enabled}`",
        f"- Validated samples: `{len(records)}`",
        "",
    ]
    if stats_mode == "real_stats":
        lines.extend(
            [
                "> **Zero real defect pixels.** The procedural-only group never sees a single real",
                "> defect pixel. It does use *aggregate shape statistics* (area ratio and aspect",
                "> ratio percentiles) computed from the 10 few-shot training masks — that is the",
                "> entire leakage surface, and it is disclosed here.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- This control uses only the fixed hand-tuned bounds in `configs/stage_a.yaml`.",
                "- A Python audit hook made any access to `real_mask_stats.json` fatal.",
                "",
            ]
        )
    lines.extend(
        [
            "| object | shape | count | area-ratio range | aspect-ratio range |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for object_name in sorted({str(record["object"]) for record in records}):
        for shape in shapes:
            selected = [
                record
                for record in records
                if record["object"] == object_name and record["defect_type"] == shape
            ]
            areas = [float(record["placement"]["mask_area_ratio"]) for record in selected]
            aspects = [
                float(record["placement"]["mask_bbox"][2] / record["placement"]["mask_bbox"][3])
                for record in selected
            ]
            lines.append(
                f"| {object_name} | {shape} | {len(selected)} | "
                f"{min(areas):.8f}–{max(areas):.8f} | "
                f"{min(aspects):.5f}–{max(aspects):.5f} |"
            )
        lines.append(
            f"| {object_name} | bounds | — | "
            f"{bounds[object_name]['area_ratio'][0]:.8f}–"
            f"{bounds[object_name]['area_ratio'][1]:.8f} | "
            f"{bounds[object_name]['aspect_ratio'][0]:.5f}–"
            f"{bounds[object_name]['aspect_ratio'][1]:.5f} |"
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
        if args.n < 1:
            raise ValidationError("n must be positive")
        paths = load_paths(args.paths)
        config = load_config(args.config)
        objects = tuple(args.objects or paths.objects)
        configured_shapes = tuple(config["procedural"]["shapes"])
        shapes = parse_shapes(args.shapes) if args.shapes else configured_shapes
        if any(shape not in configured_shapes for shape in shapes):
            raise ValidationError("Requested shapes are not enabled in stage_a.yaml")
        output_name = args.out_name or (
            "stageA_procedural_norealstats" if args.no_real_stats else "stageA_procedural"
        )
        if not OUTPUT_NAME.fullmatch(output_name):
            raise ValidationError(f"Unsafe out-name: {output_name!r}")
        seed = paths.seed if args.seed is None else args.seed
        manifest, manifest_sha256 = verify_frozen_manifest(paths.splits)
        stats_path = paths.reports / "real_mask_stats.json"
        if args.no_real_stats:
            install_forbidden_stats_guard(stats_path)
        bounds = load_shape_bounds(
            paths=paths,
            config=config,
            objects=objects,
            no_real_stats=args.no_real_stats,
            manifest_sha256=manifest_sha256,
        )
        backgrounds_by_object = {
            object_name: sorted(
                [
                    record
                    for record in manifest["images"]
                    if record["object"] == object_name
                    and record["set"] == "train"
                    and record["label"] == "good"
                ],
                key=lambda record: record["image_path"],
            )
            for object_name in objects
        }
        input_files = [
            paths.visa_raw / record["image_path"]
            for records in backgrounds_by_object.values()
            for record in records
        ]
        assert_not_blocklisted(input_files, paths.splits / "test_blocklist.json")
        blocklist = set(load_json(paths.splits / "test_blocklist.json")["sha256"])
        schedules = {object_name: even_schedule(shapes, args.n) for object_name in objects}
        LOGGER.info(
            "Procedural quotas: %s",
            {object_name: dict(Counter(schedule)) for object_name, schedule in schedules.items()},
        )
        if args.dry_run:
            return 0

        output_root = paths.synthetic / output_name
        image_dir = output_root / "images"
        mask_dir = output_root / "masks"
        metadata_path = output_root / "metadata.jsonl"
        if output_root.exists() and not args.resume:
            raise ValidationError(
                f"Output exists; use --resume or choose a new --out-name: {output_root}"
            )
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        existing = existing_records(metadata_path)
        stats_mode = "no_real_stats" if args.no_real_stats else "real_stats"
        records: list[dict[str, Any]] = []
        with metadata_path.open("a", encoding="utf-8", newline="\n") as metadata_handle:
            for object_name in objects:
                for index, shape in enumerate(schedules[object_name]):
                    sample_id = f"{object_name}__procedural_{shape}__{index:05d}"
                    if sample_id in existing:
                        record = existing[sample_id]
                        if record["bucket"] != stats_mode:
                            raise ValidationError(f"Stats mode mismatch in {sample_id}")
                        validate_output_pair(
                            output_root / record["image_path"],
                            output_root / record["mask_path"],
                            blocklist=blocklist,
                        )
                    else:
                        record = generate_one(
                            paths=paths,
                            object_name=object_name,
                            index=index,
                            shape=shape,
                            backgrounds=backgrounds_by_object[object_name],
                            bounds=bounds[object_name],
                            config=config,
                            seed=seed,
                            image_dir=image_dir,
                            mask_dir=mask_dir,
                            blocklist=blocklist,
                            stats_mode=stats_mode,
                        )
                        metadata_handle.write(json.dumps(record, sort_keys=True) + "\n")
                        metadata_handle.flush()
                    records.append(record)
                    if (index + 1) % 25 == 0 or index + 1 == args.n:
                        LOGGER.info("%s: completed %d/%d", object_name, index + 1, args.n)

        for object_name in objects:
            object_records = [record for record in records if record["object"] == object_name]
            observed = Counter(str(record["defect_type"]) for record in object_records)
            expected = Counter(schedules[object_name])
            if observed != expected:
                raise ValidationError(
                    f"{object_name} shape quota mismatch: {observed} != {expected}"
                )
            outside = [
                record["sample_id"]
                for record in object_records
                if not (
                    bounds[object_name]["area_ratio"][0]
                    <= float(record["placement"]["mask_area_ratio"])
                    <= bounds[object_name]["area_ratio"][1]
                    and bounds[object_name]["aspect_ratio"][0]
                    <= float(
                        record["placement"]["mask_bbox"][2] / record["placement"]["mask_bbox"][3]
                    )
                    <= bounds[object_name]["aspect_ratio"][1]
                )
            ]
            if len(outside) / len(object_records) >= 0.1:
                raise ValidationError(
                    f"{object_name} procedural statistic outlier rate >=10%: {len(outside)}"
                )
            render_contact_sheet(
                output_root,
                object_name,
                object_records,
                paths.figures / f"{output_name}_grid_{object_name}.png",
                count=int(config["report"]["contact_sheet_n"]),
                columns=int(config["report"]["contact_sheet_columns"]),
                stats_mode=stats_mode,
            )
        snapshot = {
            "seed": seed,
            "n_per_object": args.n,
            "objects": list(objects),
            "shapes": list(shapes),
            "stats_mode": stats_mode,
            "real_stats_access_guard": args.no_real_stats,
            "bounds": bounds,
            "manifest_sha256": manifest_sha256,
            "stage_a_config": config,
        }
        (output_root / "run_config.json").write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_report(
            paths.reports / f"{output_name}_report.md",
            output_name=output_name,
            records=records,
            shapes=shapes,
            stats_mode=stats_mode,
            bounds=bounds,
            stats_guard_enabled=args.no_real_stats,
        )
        _, final_manifest_sha256 = verify_frozen_manifest(paths.splits)
        if final_manifest_sha256 != manifest_sha256:
            raise ValidationError("Frozen manifest changed during M8")
        LOGGER.info("M8 generated %d validated samples in %s", len(records), output_root)
        return 0
    except (
        ImagingError,
        IntegrityError,
        MetadataError,
        OSError,
        ValidationError,
        json.JSONDecodeError,
    ) as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
