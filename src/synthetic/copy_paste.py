"""Generate deterministic copy-paste defects from frozen few-shot components."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from skimage.measure import label

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.imaging import (  # isort: skip
    ImagingError,
    blend_component,
    detect_legal_roi,
    mask_bbox,
    place_on_legal_roi,
    transform_component,
)
from src.common.integrity import (  # isort: skip
    IntegrityError,
    assert_not_blocklisted,
    load_json,
    sha256_file,
    verify_frozen_manifest,
)
from src.common.paths import Paths, load_paths  # isort: skip
from src.synthetic.metadata import MetadataError, validate_metadata  # isort: skip


LOGGER = logging.getLogger("copy_paste")
PIPELINE_VERSION = "0.1.0"
OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class ValidationError(RuntimeError):
    """An M7 assertion failed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument("--config", default="configs/stage_a.yaml", type=Path)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument(
        "--blend",
        choices=("poisson", "feather", "mixed"),
        default="mixed",
    )
    parser.add_argument("--out-name", default="stageA_copypaste")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValidationError(f"Expected config object in {path}")
    return value


def allocate_quotas(type_sizes: dict[int, int], total: int, minimum: int = 50) -> dict[int, int]:
    """Allocate exactly ``total`` by largest remainder while honoring a minimum."""

    if not type_sizes or total < minimum * len(type_sizes):
        raise ValidationError(
            f"Cannot allocate total={total} with minimum={minimum} across {len(type_sizes)} types"
        )
    size_total = sum(type_sizes.values())
    raw = {key: total * value / size_total for key, value in type_sizes.items()}
    quotas = {key: max(minimum, int(np.floor(value))) for key, value in raw.items()}
    while sum(quotas.values()) < total:
        candidate = max(
            quotas,
            key=lambda key: (raw[key] - quotas[key], type_sizes[key], -key),
        )
        quotas[candidate] += 1
    while sum(quotas.values()) > total:
        candidates = [key for key, value in quotas.items() if value > minimum]
        if not candidates:
            raise ValidationError("Minimum quotas exceed requested total")
        candidate = min(
            candidates,
            key=lambda key: (raw[key] - quotas[key], -type_sizes[key], key),
        )
        quotas[candidate] -= 1
    return dict(sorted(quotas.items()))


def object_code(object_name: str) -> int:
    return int.from_bytes(object_name.encode("utf-8"), byteorder="little") & 0xFFFFFFFF


def sample_rng(seed: int, object_name: str, index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([seed, object_code(object_name), index])
    return np.random.Generator(np.random.PCG64(sequence))


def range_value(
    rng: np.random.Generator,
    values: list[float],
) -> float:
    if len(values) != 2:
        raise ValidationError(f"Expected [min, max], observed {values!r}")
    return float(rng.uniform(float(values[0]), float(values[1])))


def load_source_component(
    paths: Paths,
    component: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    image_path = paths.visa_raw / component["image_path"]
    mask_path = paths.visa_raw / component["mask_path"]
    with Image.open(image_path) as image_handle:
        image = np.asarray(image_handle.convert("RGB"))
    with Image.open(mask_path) as mask_handle:
        full_mask = np.asarray(mask_handle.convert("L")) > 0
    components = label(full_mask, connectivity=2)
    component_mask = components == int(component["component_id"]) + 1
    if int(component_mask.sum()) != int(component["area_px"]):
        raise ValidationError(
            f"Frozen component area changed for {component['image_path']} "
            f"#{component['component_id']}"
        )
    x0, y0, x1, y1 = (int(value) for value in component["crop_bbox"])
    return image[y0:y1, x0:x1], component_mask[y0:y1, x0:x1]


def roi_bbox(roi: np.ndarray) -> list[int]:
    return list(mask_bbox(roi))


def choose_blend(
    requested: str,
    rng: np.random.Generator,
    poisson_probability: float,
) -> str:
    if requested == "mixed":
        return "poisson" if rng.random() < poisson_probability else "feather"
    return requested


def object_roi_config(config: dict[str, Any], object_name: str) -> dict[str, Any]:
    roi_config = config["roi"]
    result = dict(roi_config["default"])
    result.update(roi_config.get("objects", {}).get(object_name, {}))
    return result


def existing_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                validate_metadata(record)
            except (json.JSONDecodeError, MetadataError) as error:
                raise ValidationError(f"{path}:{line_number}: {error}") from error
            result[record["sample_id"]] = record
    return result


def type_schedule(quotas: dict[int, int]) -> list[int]:
    return [cluster_id for cluster_id, quota in quotas.items() for _ in range(quota)]


def validate_output_pair(
    image_path: Path,
    mask_path: Path,
    *,
    blocklist: set[str],
) -> tuple[str, str]:
    with Image.open(image_path) as image:
        image_size = image.size
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image.convert("L"))
        mask_size = mask_image.size
    if image_size != mask_size:
        raise ValidationError(f"Output image/mask size mismatch: {image_path}")
    if not np.any(mask):
        raise ValidationError(f"Output mask is empty: {mask_path}")
    image_digest = sha256_file(image_path)
    mask_digest = sha256_file(mask_path)
    if image_digest in blocklist or mask_digest in blocklist:
        raise ValidationError(f"Generated output unexpectedly hit test blocklist: {image_path}")
    return image_digest, mask_digest


def generate_one(
    *,
    paths: Paths,
    object_name: str,
    index: int,
    cluster_id: int,
    type_record: dict[str, Any],
    background_records: list[dict[str, Any]],
    config: dict[str, Any],
    requested_blend: str,
    seed: int,
    image_dir: Path,
    mask_dir: Path,
    blocklist: set[str],
) -> dict[str, Any]:
    rng = sample_rng(seed, object_name, index)
    component = type_record["components"][int(rng.integers(0, len(type_record["components"])))]
    source_patch, source_mask = load_source_component(paths, component)
    transform_config = config["copy_paste"]
    selected_background: dict[str, Any] | None = None
    background: np.ndarray | None = None
    legal_roi: np.ndarray | None = None
    placement: tuple[int, int] | None = None
    patch: np.ndarray | None = None
    transformed_mask: np.ndarray | None = None
    rotation = 0.0
    scale = 1.0
    flip = False
    for _ in range(int(transform_config["max_transform_tries"])):
        rotation = range_value(rng, transform_config["rotation_deg"])
        scale = range_value(rng, transform_config["scale"])
        flip = bool(rng.random() < float(transform_config["flip_probability"]))
        patch, transformed_mask = transform_component(
            source_patch,
            source_mask,
            rotation_deg=rotation,
            scale=scale,
            flip=flip,
            brightness=range_value(rng, transform_config["brightness"]),
            contrast=range_value(rng, transform_config["contrast"]),
            saturation=range_value(rng, transform_config["saturation"]),
            hue_shift=range_value(rng, transform_config["hue_shift"]),
        )
        for _ in range(int(transform_config["max_background_tries"])):
            candidate = background_records[int(rng.integers(0, len(background_records)))]
            with Image.open(paths.visa_raw / candidate["image_path"]) as image_handle:
                candidate_image = np.asarray(image_handle.convert("RGB"))
            try:
                candidate_roi = detect_legal_roi(
                    candidate_image,
                    **object_roi_config(config, object_name),
                )
                candidate_placement = place_on_legal_roi(
                    transformed_mask,
                    candidate_roi,
                    rng,
                    max_tries=int(transform_config["max_place_tries"]),
                )
            except ImagingError:
                continue
            selected_background = candidate
            background = candidate_image
            legal_roi = candidate_roi
            placement = candidate_placement
            break
        if selected_background is not None:
            break
    if (
        selected_background is None
        or background is None
        or legal_roi is None
        or placement is None
        or patch is None
        or transformed_mask is None
    ):
        raise ValidationError(
            f"Could not place {object_name} sample {index} after transform/background retries"
        )

    x0, y0 = placement
    method = choose_blend(
        requested_blend,
        rng,
        float(transform_config["poisson_probability"]),
    )
    opacity = range_value(rng, transform_config["opacity"])
    feather_radius = range_value(rng, transform_config["feather_radius"])
    synthetic, actual_blend = blend_component(
        background,
        patch,
        transformed_mask,
        x0=x0,
        y0=y0,
        method=method,
        opacity=opacity,
        feather_radius=feather_radius,
    )
    placed_mask = np.zeros(legal_roi.shape, dtype=np.uint8)
    patch_height, patch_width = transformed_mask.shape
    placed_mask[y0 : y0 + patch_height, x0 : x0 + patch_width] = (
        transformed_mask.astype(np.uint8) * 255
    )
    if not bool(np.all(legal_roi[placed_mask > 0])):
        raise ValidationError("Placed mask is not 100% contained in legal ROI")

    source_stem = Path(component["image_path"]).stem
    defect_type = f"type{cluster_id}"
    sample_id = f"{object_name}__{defect_type}__{source_stem}__{index:05d}"
    filename = f"{sample_id}.png"
    image_path = image_dir / filename
    mask_path = mask_dir / filename
    Image.fromarray(synthetic).save(image_path, format="PNG", compress_level=6)
    Image.fromarray(placed_mask).save(mask_path, format="PNG", compress_level=6)
    validate_output_pair(image_path, mask_path, blocklist=blocklist)

    destination_bbox = list(mask_bbox(placed_mask > 0))
    mask_area = int(np.count_nonzero(placed_mask))
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "object": object_name,
        "defect_type": defect_type,
        "trigger_token": type_record["trigger_token"],
        "generator": "stageA_copypaste",
        "bucket": None,
        "image_path": f"images/{filename}",
        "mask_path": f"masks/{filename}",
        "source": {
            "background_image": selected_background["image_path"],
            "background_sha256": selected_background["sha256"],
            "defect_source_image": component["image_path"],
            "defect_source_mask": component["mask_path"],
            "defect_source_component_id": component["component_id"],
        },
        "placement": {
            "roi_bbox": roi_bbox(legal_roi),
            "mask_bbox": destination_bbox,
            "affine": {
                "dx": x0 - int(component["crop_bbox"][0]),
                "dy": y0 - int(component["crop_bbox"][1]),
                "rotation_deg": rotation,
                "scale": scale,
                "flip": flip,
            },
            "mask_area_px": mask_area,
            "mask_area_ratio": mask_area / placed_mask.size,
        },
        "generation": {
            "seed": int(
                np.random.SeedSequence([seed, object_code(object_name), index]).generate_state(
                    1,
                    dtype=np.uint64,
                )[0]
            ),
            "base_model": None,
            "lora_path": None,
            "prompt": None,
            "negative_prompt": None,
            "guidance_scale": None,
            "num_inference_steps": None,
            "strength": None,
            "crop_ratio": None,
            "crop_bbox": None,
            "model_resolution": None,
            "blend": actual_blend,
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
        axis.set_title(
            f"{record['defect_type']} | {record['generation']['blend']}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes_array[len(selected) :]:
        axis.axis("off")
    figure.suptitle(
        f"{object_name}: Stage A copy-paste (red = synthetic mask)",
        fontsize=15,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, facecolor="white")
    plt.close(figure)


def write_generation_report(
    destination: Path,
    *,
    out_name: str,
    records: list[dict[str, Any]],
    quotas_by_object: dict[str, dict[int, int]],
) -> None:
    lines = [
        "# Stage A Generation Report",
        "",
        "## M7 copy-paste",
        "",
        f"- Output: `${{synthetic}}/{out_name}`",
        f"- Validated samples: `{len(records)}`",
        "- Quotas use largest-remainder proportional allocation and sum exactly to n.",
        "- Every output image/mask pair passed size, non-empty, ROI containment, schema,",
        "  and test-blocklist assertions.",
        "",
        "| object | type | quota | poisson | feather |",
        "|---|---|---:|---:|---:|",
    ]
    for object_name, quotas in quotas_by_object.items():
        for cluster_id, quota in quotas.items():
            selected = [
                record
                for record in records
                if record["object"] == object_name and record["defect_type"] == f"type{cluster_id}"
            ]
            blends = Counter(record["generation"]["blend"] for record in selected)
            lines.append(
                f"| {object_name} | type{cluster_id} | {quota} | "
                f"{blends['poisson']} | {blends['feather_alpha']} |"
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
        source_components = [
            component
            for object_name in objects
            for type_record in defect_types["objects"][object_name]["types"]
            for component in type_record["components"]
        ]
        input_files = {
            paths.visa_raw / record["image_path"]
            for records in backgrounds_by_object.values()
            for record in records
        }
        input_files.update(
            paths.visa_raw / component[key]
            for component in source_components
            for key in ("image_path", "mask_path")
        )
        assert_not_blocklisted(
            sorted(input_files),
            paths.splits / "test_blocklist.json",
        )

        blocklist = set(load_json(paths.splits / "test_blocklist.json")["sha256"])
        quotas_by_object: dict[str, dict[int, int]] = {}
        for object_name in objects:
            type_sizes = {
                int(type_record["cluster_id"]): int(type_record["n_components"])
                for type_record in defect_types["objects"][object_name]["types"]
            }
            minimum = 50 if args.n >= 50 * len(type_sizes) else 1
            quotas_by_object[object_name] = allocate_quotas(
                type_sizes,
                args.n,
                minimum=minimum,
            )
        LOGGER.info("Generation quotas: %s", quotas_by_object)
        if args.dry_run:
            return 0

        output_root = paths.synthetic / args.out_name
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

        generated_records: list[dict[str, Any]] = []
        with metadata_path.open("a", encoding="utf-8", newline="\n") as metadata_handle:
            for object_name in objects:
                types = {
                    int(type_record["cluster_id"]): type_record
                    for type_record in defect_types["objects"][object_name]["types"]
                }
                schedule = type_schedule(quotas_by_object[object_name])
                for index, cluster_id in enumerate(schedule):
                    prefix = f"{object_name}__type{cluster_id}__"
                    already = [
                        record
                        for sample_id, record in existing.items()
                        if sample_id.startswith(prefix) and sample_id.endswith(f"__{index:05d}")
                    ]
                    if already:
                        record = already[0]
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
                            cluster_id=cluster_id,
                            type_record=types[cluster_id],
                            background_records=backgrounds_by_object[object_name],
                            config=config,
                            requested_blend=args.blend,
                            seed=seed,
                            image_dir=image_dir,
                            mask_dir=mask_dir,
                            blocklist=blocklist,
                        )
                        metadata_handle.write(json.dumps(record, sort_keys=True) + "\n")
                        metadata_handle.flush()
                    generated_records.append(record)
                    if (index + 1) % 25 == 0 or index + 1 == len(schedule):
                        LOGGER.info(
                            "%s: completed %d/%d",
                            object_name,
                            index + 1,
                            len(schedule),
                        )

        for object_name in objects:
            object_records = [
                record for record in generated_records if record["object"] == object_name
            ]
            observed = Counter(record["defect_type"] for record in object_records)
            expected = {
                f"type{cluster_id}": quota
                for cluster_id, quota in quotas_by_object[object_name].items()
            }
            if observed != expected:
                raise ValidationError(
                    f"{object_name} quota mismatch: observed={observed}, expected={expected}"
                )
            render_contact_sheet(
                output_root,
                object_name,
                object_records,
                paths.figures / f"{args.out_name}_grid_{object_name}.png",
                count=int(config["report"]["contact_sheet_n"]),
                columns=int(config["report"]["contact_sheet_columns"]),
            )
        write_generation_report(
            paths.reports / "generation_report.md",
            out_name=args.out_name,
            records=generated_records,
            quotas_by_object=quotas_by_object,
        )

        _, final_manifest_sha256 = verify_frozen_manifest(paths.splits)
        if final_manifest_sha256 != manifest_sha256:
            raise ValidationError("Frozen manifest changed during M7")
        if sha256_file(defect_types_path) != defect_types_sha256:
            raise ValidationError("Defect types changed during M7")
        LOGGER.info("M7 generated %d validated samples in %s", len(generated_records), output_root)
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
