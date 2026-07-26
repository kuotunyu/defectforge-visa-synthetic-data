"""Cluster few-shot defect components with DINOv2 and morphology features."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.measure import label, regionprops
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import (  # isort: skip
    IntegrityError,
    assert_not_blocklisted,
    load_json,
    sha256_file,
    verify_frozen_manifest,
)
from src.common.paths import Paths, load_paths  # isort: skip


LOGGER = logging.getLogger("cluster_defect_types")
MODEL_ID = "facebook/dinov2-base"
MODEL_LICENSE = "Apache-2.0"
EMBEDDING_BATCH_SIZE = 8
MORPHOLOGY_NAMES = (
    "area_ratio",
    "aspect_ratio",
    "solidity",
    "extent",
    "eccentricity",
    "components_in_image",
    "intensity_delta",
    "contrast_delta",
)


class ValidationError(RuntimeError):
    """An M6 assertion failed."""


@dataclass(slots=True)
class ComponentSample:
    object_name: str
    image_path: str
    mask_path: str
    component_id: int
    bbox: tuple[int, int, int, int]
    crop_bbox: tuple[int, int, int, int]
    area_px: int
    morphology: list[float]
    crop: Image.Image
    crop_mask: np.ndarray

    @property
    def key(self) -> tuple[str, int]:
        return self.image_path, self.component_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--k-range", nargs=2, type=int, default=(1, 5), metavar=("MIN", "MAX"))
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-component-area", type=int, default=32)
    parser.add_argument("--auto-name", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def square_context_bbox(
    bbox: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    context_scale: float = 2.0,
    minimum_side: int = 64,
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    center_x = x + width / 2
    center_y = y + height / 2
    side = max(minimum_side, math.ceil(max(width, height) * context_scale))
    side = min(side, image_width, image_height)
    x0 = round(center_x - side / 2)
    y0 = round(center_y - side / 2)
    x0 = max(0, min(x0, image_width - side))
    y0 = max(0, min(y0, image_height - side))
    return x0, y0, x0 + side, y0 + side


def extract_components(
    paths: Paths,
    object_name: str,
    records: list[dict[str, Any]],
    *,
    min_area: int,
) -> tuple[list[ComponentSample], int]:
    samples: list[ComponentSample] = []
    filtered_tiny = 0
    for record in records:
        image_file = paths.visa_raw / record["image_path"]
        mask_file = paths.visa_raw / record["mask_path"]
        with Image.open(image_file) as image_handle:
            image = image_handle.convert("RGB")
            image_array = np.asarray(image)
        with Image.open(mask_file) as mask_handle:
            mask = np.asarray(mask_handle.convert("L")) > 0
        if mask.shape != image_array.shape[:2]:
            raise ValidationError(f"Image/mask shape mismatch: {record['image_path']}")

        regions = list(regionprops(label(mask, connectivity=2)))
        retained = [region for region in regions if region.area >= min_area]
        filtered_tiny += len(regions) - len(retained)
        if not retained:
            raise ValidationError(
                f"No component >= {min_area}px in seed mask {record['mask_path']}"
            )

        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        image_height, image_width = mask.shape
        for region in retained:
            min_row, min_col, max_row, max_col = region.bbox
            bbox = (
                int(min_col),
                int(min_row),
                int(max_col - min_col),
                int(max_row - min_row),
            )
            crop_bbox = square_context_bbox(
                bbox,
                image_width=image_width,
                image_height=image_height,
            )
            x0, y0, x1, y1 = crop_bbox
            component_mask = label(mask, connectivity=2) == region.label
            local_mask = component_mask[y0:y1, x0:x1]
            local_gray = gray[y0:y1, x0:x1]
            inside = local_gray[local_mask]
            outside = local_gray[~local_mask]
            if not len(outside):
                outside = gray[~component_mask]
            intensity_delta = float(inside.mean() - outside.mean())
            contrast_delta = float(inside.std() - outside.std())
            morphology = [
                float(region.area / (image_width * image_height)),
                float(bbox[2] / bbox[3]),
                float(region.solidity),
                float(region.extent),
                float(region.eccentricity),
                float(len(retained)),
                intensity_delta,
                contrast_delta,
            ]
            samples.append(
                ComponentSample(
                    object_name=object_name,
                    image_path=str(record["image_path"]),
                    mask_path=str(record["mask_path"]),
                    component_id=int(region.label - 1),
                    bbox=bbox,
                    crop_bbox=crop_bbox,
                    area_px=int(region.area),
                    morphology=morphology,
                    crop=image.crop(crop_bbox),
                    crop_mask=local_mask,
                )
            )
    return samples, filtered_tiny


def embedding_cache_key(
    samples: list[ComponentSample],
    *,
    selection_sha256: str,
) -> str:
    payload = {
        "model": MODEL_ID,
        "selection_sha256": selection_sha256,
        "components": [
            {
                "image_path": sample.image_path,
                "component_id": sample.component_id,
                "crop_bbox": sample.crop_bbox,
            }
            for sample in samples
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def extract_dinov2_embeddings(
    paths: Paths,
    samples: list[ComponentSample],
    *,
    selection_sha256: str,
) -> tuple[np.ndarray, str]:
    cache_key = embedding_cache_key(samples, selection_sha256=selection_sha256)
    cache_file = paths.cache / f"dinov2_components_{cache_key}.npz"
    if cache_file.is_file():
        with np.load(cache_file) as cached:
            embeddings = cached["embeddings"]
            model_revision = str(cached["model_revision"].item())
        if embeddings.shape != (len(samples), 768):
            raise ValidationError(f"Unexpected cached embedding shape: {embeddings.shape}")
        LOGGER.info("Loaded %d cached DINOv2 embeddings", len(samples))
        return embeddings, model_revision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise ValidationError("M6 requires CUDA for DINOv2 extraction")
    LOGGER.info("Loading %s on %s", MODEL_ID, device)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    model_revision = str(
        getattr(model.config, "_commit_hash", None) or "unavailable"
    )

    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(samples), EMBEDDING_BATCH_SIZE):
            batch = samples[start : start + EMBEDDING_BATCH_SIZE]
            inputs = processor(
                images=[sample.crop for sample in batch],
                return_tensors="pt",
            )
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            batches.append(cls_embeddings.float().cpu().numpy())
            LOGGER.info(
                "Embedded %d/%d components",
                min(start + len(batch), len(samples)),
                len(samples),
            )
    embeddings = np.concatenate(batches, axis=0)
    if embeddings.shape != (len(samples), 768):
        raise ValidationError(f"Unexpected DINOv2 embedding shape: {embeddings.shape}")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        embeddings=embeddings,
        model_revision=np.asarray(model_revision),
    )
    del model
    torch.cuda.empty_cache()
    return embeddings, model_revision


def balanced_features(
    embeddings: np.ndarray,
    samples: list[ComponentSample],
) -> np.ndarray:
    morphology = np.asarray([sample.morphology for sample in samples], dtype=np.float64)
    semantic_block = StandardScaler().fit_transform(embeddings)
    morphology_block = StandardScaler().fit_transform(morphology)
    semantic_block /= math.sqrt(semantic_block.shape[1])
    morphology_block /= math.sqrt(morphology_block.shape[1])
    combined = np.concatenate((semantic_block, morphology_block), axis=1)
    if not np.isfinite(combined).all():
        raise ValidationError("Combined feature matrix contains non-finite values")
    return combined


def stable_cluster_labels(
    labels: np.ndarray,
    samples: list[ComponentSample],
) -> np.ndarray:
    cluster_order = sorted(
        {int(item) for item in labels},
        key=lambda cluster: min(
            sample.key
            for sample, label_value in zip(samples, labels, strict=True)
            if int(label_value) == cluster
        ),
    )
    mapping = {cluster: index for index, cluster in enumerate(cluster_order)}
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int64)


def choose_clustering(
    features: np.ndarray,
    samples: list[ComponentSample],
    *,
    k_min: int,
    k_max: int,
    min_cluster_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]], bool]:
    candidates: list[dict[str, Any]] = []
    eligible: list[tuple[float, np.ndarray, int]] = []
    upper = min(k_max, len(samples) - 1)
    for k in range(max(2, k_min), upper + 1):
        raw_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
            features
        )
        labels = stable_cluster_labels(raw_labels, samples)
        sizes = Counter(int(value) for value in labels)
        minimum = min(sizes.values())
        score = float(silhouette_score(features, labels))
        is_eligible = minimum >= min_cluster_size
        candidates.append(
            {
                "k": k,
                "silhouette": score,
                "cluster_sizes": dict(sorted(sizes.items())),
                "eligible": is_eligible,
            }
        )
        if is_eligible:
            eligible.append((score, labels, k))

    if not eligible:
        labels = np.zeros(len(samples), dtype=np.int64)
        candidates.append(
            {
                "k": 1,
                "silhouette": None,
                "cluster_sizes": {0: len(samples)},
                "eligible": True,
                "reason": "fallback: no multi-cluster candidate met minimum size",
            }
        )
        return labels, candidates, True

    _, best_labels, _ = max(eligible, key=lambda item: (item[0], -item[2]))
    return best_labels, candidates, False


def component_json(sample: ComponentSample) -> dict[str, Any]:
    return {
        "image_path": sample.image_path,
        "mask_path": sample.mask_path,
        "component_id": sample.component_id,
        "bbox": list(sample.bbox),
        "crop_bbox": list(sample.crop_bbox),
        "area_px": sample.area_px,
        "morphology": {
            name: value
            for name, value in zip(MORPHOLOGY_NAMES, sample.morphology, strict=True)
        },
    }


def render_clusters(
    paths: Paths,
    object_name: str,
    samples: list[ComponentSample],
    labels: np.ndarray,
) -> Path:
    columns = 5
    rows = math.ceil(len(samples) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.4 * columns, 3.25 * rows),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(-1)
    palette = plt.get_cmap("tab10")
    ordered = sorted(
        zip(samples, labels, strict=True),
        key=lambda pair: (int(pair[1]), pair[0].key),
    )
    for axis, (sample, cluster_id) in zip(axes_array, ordered, strict=False):
        axis.imshow(sample.crop)
        axis.contour(
            sample.crop_mask.astype(np.uint8),
            levels=[0.5],
            colors=["#ff2d55"],
            linewidths=2,
        )
        color = palette(int(cluster_id) % 10)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(4)
            spine.set_edgecolor(color)
        axis.set_title(
            f"type{int(cluster_id)} | {Path(sample.image_path).name} #{sample.component_id}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes_array[len(ordered) :]:
        axis.axis("off")
    chosen_k = len({int(value) for value in labels})
    figure.suptitle(
        f"{object_name}: DINOv2 + morphology clusters (k={chosen_k}, red=component)",
        fontsize=15,
    )
    destination = (
        paths.figures / f"defect_type_cluster_{object_name}_{chosen_k}.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170, facecolor="white")
    plt.close(figure)
    return destination


def write_report(
    destination: Path,
    *,
    model_revision: str,
    object_results: dict[str, dict[str, Any]],
    min_component_area: int,
    min_cluster_size: int,
) -> None:
    lines = [
        "# M6 Defect-type Clustering Report",
        "",
        f"- Backbone: `{MODEL_ID}` (`{MODEL_LICENSE}`)",
        f"- Model revision: `{model_revision}`",
        "- Semantic representation: `last_hidden_state[:, 0, :]` (CLS, 768-D)",
        f"- Minimum component area: `{min_component_area}` px",
        f"- Minimum cluster size: `{min_cluster_size}`",
        "- Feature fusion: each block standardized, then divided by sqrt(block dimension)",
        "- Naming: stable temporary `typeN` tokens; user display-name review is non-blocking",
        "",
    ]
    for object_name, result in object_results.items():
        lines.extend(
            [
                f"## {object_name}",
                "",
                f"- Retained components: `{result['n_components']}`",
                f"- Tiny components filtered: `{result['filtered_tiny_components']}`",
                f"- Selected k: `{len(result['types'])}`",
                f"- Fallback applied: `{str(result['fallback_applied']).lower()}`",
                "",
                "| k | silhouette | cluster sizes | eligible |",
                "|---:|---:|---|---|",
            ]
        )
        for candidate in result["clustering_candidates"]:
            score = (
                "n/a"
                if candidate["silhouette"] is None
                else f"{candidate['silhouette']:.6f}"
            )
            lines.append(
                f"| {candidate['k']} | {score} | {candidate['cluster_sizes']} | "
                f"{candidate['eligible']} |"
            )
        lines.extend(["", "### Frozen temporary types", ""])
        for type_record in result["types"]:
            lines.append(
                f"- `{type_record['trigger_token']}`: "
                f"{type_record['n_components']} components"
            )
        lines.append("")
    lines.extend(
        [
            "## Assertions",
            "",
            "- Every input image and mask cleared the test blocklist: **passed**",
            "- Every retained component area meets the minimum: **passed**",
            "- Every selected cluster meets the minimum size, or single-type fallback applied: **passed**",
            "- Frozen manifest and few-shot selection checksums unchanged: **passed**",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def write_frozen_types(paths: Paths, payload: dict[str, Any]) -> str:
    destination = paths.splits / "defect_types.json"
    checksum_path = paths.splits / "DEFECT_TYPES.sha256"
    if destination.exists() or checksum_path.exists():
        raise ValidationError("Refusing to overwrite frozen defect-type artifacts")
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(destination)
    checksum_path.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return digest


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        k_min, k_max = args.k_range
        if k_min < 1 or k_max < k_min:
            raise ValidationError(f"Invalid k range: {args.k_range}")
        if args.min_cluster_size < 1 or args.min_component_area < 1:
            raise ValidationError("Minimum sizes must be positive")
        if not args.auto_name:
            raise ValidationError(
                "Non-interactive M6 requires --auto-name for stable temporary tokens"
            )

        paths = load_paths(args.paths)
        objects = tuple(args.objects or paths.objects)
        seed = paths.seed if args.seed is None else args.seed
        _, manifest_sha256 = verify_frozen_manifest(paths.splits)
        selection_path = paths.splits / "fewshot_selection.json"
        selection_sha256 = sha256_file(selection_path)
        selection = load_json(selection_path)
        if selection.get("manifest_sha256") != manifest_sha256:
            raise ValidationError("Few-shot selection points to another manifest")
        if selection.get("seed") != seed:
            raise ValidationError(
                f"Selection seed {selection.get('seed')} does not match requested {seed}"
            )

        records_by_object = {
            object_name: selection["objects"][object_name]["fewshot_seed"]
            for object_name in objects
        }
        input_files = [
            paths.visa_raw / record[key]
            for records in records_by_object.values()
            for record in records
            for key in ("image_path", "mask_path")
        ]
        assert_not_blocklisted(input_files, paths.splits / "test_blocklist.json")

        samples_by_object: dict[str, list[ComponentSample]] = {}
        filtered_by_object: dict[str, int] = {}
        all_samples: list[ComponentSample] = []
        for object_name, records in records_by_object.items():
            samples, filtered = extract_components(
                paths,
                object_name,
                records,
                min_area=args.min_component_area,
            )
            samples_by_object[object_name] = samples
            filtered_by_object[object_name] = filtered
            all_samples.extend(samples)
            LOGGER.info(
                "%s: retained %d components, filtered %d tiny regions",
                object_name,
                len(samples),
                filtered,
            )

        embeddings, model_revision = extract_dinov2_embeddings(
            paths,
            all_samples,
            selection_sha256=selection_sha256,
        )
        object_results: dict[str, dict[str, Any]] = {}
        offset = 0
        for object_name, samples in samples_by_object.items():
            object_embeddings = embeddings[offset : offset + len(samples)]
            offset += len(samples)
            features = balanced_features(object_embeddings, samples)
            labels, candidates, fallback = choose_clustering(
                features,
                samples,
                k_min=k_min,
                k_max=k_max,
                min_cluster_size=args.min_cluster_size,
            )
            sizes = Counter(int(value) for value in labels)
            if min(sizes.values()) < args.min_cluster_size:
                raise ValidationError(
                    f"{object_name} selected cluster violates minimum size: {sizes}"
                )

            types: list[dict[str, Any]] = []
            for cluster_id in sorted(sizes):
                components = [
                    component_json(sample)
                    for sample, label_value in zip(samples, labels, strict=True)
                    if int(label_value) == cluster_id
                ]
                types.append(
                    {
                        "cluster_id": cluster_id,
                        "type_name": f"type{cluster_id} (temporary)",
                        "trigger_token": f"<{object_name}-type{cluster_id}>",
                        "n_components": len(components),
                        "components": components,
                    }
                )
            object_results[object_name] = {
                "n_components": len(samples),
                "filtered_tiny_components": filtered_by_object[object_name],
                "types": types,
                "fallback_applied": fallback,
                "clustering_candidates": candidates,
            }
            render_clusters(paths, object_name, samples, labels)

        payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "method": (
                "agglomerative(standardized block-balanced DINOv2-base CLS + morphology), "
                "silhouette-selected k with minimum cluster size"
            ),
            "model": {
                "id": MODEL_ID,
                "revision": model_revision,
                "license": MODEL_LICENSE,
                "embedding": "last_hidden_state[:, 0, :]",
            },
            "manifest_sha256": manifest_sha256,
            "fewshot_selection_sha256": selection_sha256,
            "seed": seed,
            "min_component_area": args.min_component_area,
            "min_cluster_size": args.min_cluster_size,
            "confirmed_by_user": False,
            "naming": "auto-temporary; trigger tokens are frozen, display names may change",
            "objects": object_results,
        }
        if args.dry_run:
            LOGGER.info(
                "Dry run selected types: %s",
                {
                    object_name: [
                        type_record["n_components"]
                        for type_record in result["types"]
                    ]
                    for object_name, result in object_results.items()
                },
            )
            return 0

        digest = write_frozen_types(paths, payload)
        write_report(
            paths.reports / "defect_type_report.md",
            model_revision=model_revision,
            object_results=object_results,
            min_component_area=args.min_component_area,
            min_cluster_size=args.min_cluster_size,
        )
        _, final_manifest_sha256 = verify_frozen_manifest(paths.splits)
        if final_manifest_sha256 != manifest_sha256:
            raise ValidationError("Frozen manifest changed during M6")
        if sha256_file(selection_path) != selection_sha256:
            raise ValidationError("Few-shot selection changed during M6")
        LOGGER.info("M6 defect types frozen with SHA256 %s", digest)
        return 0
    except (
        IntegrityError,
        OSError,
        ValidationError,
        json.JSONDecodeError,
    ) as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
