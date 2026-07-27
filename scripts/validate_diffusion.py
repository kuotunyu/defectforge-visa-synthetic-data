"""Independently validate Stage B diffusion generation and refine artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import (  # isort: skip
    load_json,
    read_checksum_file,
    sha256_file,
    verify_frozen_manifest,
)
from src.common.paths import Paths, load_paths  # isort: skip
from src.synthetic.generate_diffusion import (  # isort: skip
    load_config,
    pipeline_version,
    read_placements,
)
from src.synthetic.copy_paste import object_code  # isort: skip
from src.synthetic.metadata import validate_metadata  # isort: skip

LOGGER = logging.getLogger("validate_diffusion")


class DiffusionValidationError(RuntimeError):
    """A Stage B artifact violated the locked generation contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument("--config", default="configs/generate_sd2.yaml", type=Path)
    parser.add_argument("--out-name")
    parser.add_argument("--bucket", choices=("original", "searched"), required=True)
    parser.add_argument("--n", type=int)
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiffusionValidationError(message)


def configured_adapter(paths: Paths, object_config: dict[str, Any]) -> tuple[Path, str]:
    root_name = str(object_config.get("adapter_root", "runs"))
    if root_name == "runs":
        adapter = paths.runs / str(object_config["adapter"])
        logical = Path("runs") / str(object_config["adapter"])
    elif root_name == "colab_results":
        adapter = paths.colab_results / str(object_config["adapter"])
        try:
            logical = adapter.relative_to(paths.project_root)
        except ValueError as error:
            raise DiffusionValidationError(
                f"Colab adapter is outside the project root: {adapter}"
            ) from error
    else:
        raise DiffusionValidationError(f"Unsupported adapter_root: {root_name}")
    return adapter.resolve(strict=False), logical.as_posix()


def expected_candidate_seed(
    seed: int,
    object_name: str,
    placement_index: int,
    candidate_index: int,
) -> int:
    return int(
        np.random.SeedSequence(
            [
                seed,
                object_code(f"{object_name}:stageB-diffusion"),
                placement_index,
                candidate_index,
            ]
        ).generate_state(1, dtype=np.uint64)[0]
    )


def validate_candidate_schedule(
    candidates: list[dict[str, Any]],
    *,
    bucket: str,
    config: dict[str, Any],
    object_name: str,
    placement_index: int,
    seed: int,
    sample_id: str,
) -> None:
    generation = config["generation"]
    refine = config["refine"]
    baseline = (
        float(generation["guidance_scale"]),
        float(generation["crop_ratio"]),
    )
    indices = [int(candidate["candidate_index"]) for candidate in candidates]
    require(
        indices == list(range(len(candidates))),
        f"{sample_id}: candidate indices are not contiguous",
    )
    pairs = [
        (float(candidate["guidance_scale"]), float(candidate["crop_ratio"]))
        for candidate in candidates
    ]
    require(len(set(pairs)) == len(pairs), f"{sample_id}: duplicate candidate parameters")
    require(pairs[0] == baseline, f"{sample_id}: candidate 0 is not the original baseline")
    for candidate in candidates:
        candidate_index = int(candidate["candidate_index"])
        require(
            int(candidate["generator_seed"])
            == expected_candidate_seed(seed, object_name, placement_index, candidate_index),
            f"{sample_id}: candidate seed mismatch",
        )
    if bucket != "searched":
        require(pairs == [baseline], f"{sample_id}: original schedule mismatch")
        return
    guidance_grid = {float(value) for value in refine["guidance_grid"]}
    crop_grid = {float(value) for value in refine["crop_ratio_grid"]}
    require(
        all(guidance in guidance_grid and crop in crop_grid for guidance, crop in pairs),
        f"{sample_id}: candidate parameter is outside the refine grid",
    )
    if len(candidates) >= len(guidance_grid):
        require(
            {guidance for guidance, _ in pairs} == guidance_grid,
            f"{sample_id}: refine schedule does not cover the guidance grid",
        )
    if len(candidates) >= len(crop_grid):
        require(
            {crop for _, crop in pairs} == crop_grid,
            f"{sample_id}: refine schedule does not cover the crop grid",
        )


def validate_search_baseline(
    *,
    searched_sidecar: dict[str, Any],
    original_sidecar: dict[str, Any],
    sample_id: str,
) -> None:
    original_candidates = original_sidecar.get("candidates")
    searched_candidates = searched_sidecar.get("candidates")
    require(
        isinstance(original_candidates, list) and len(original_candidates) == 1,
        f"{sample_id}: malformed original baseline",
    )
    require(
        isinstance(searched_candidates, list) and searched_candidates,
        f"{sample_id}: malformed searched candidates",
    )
    require(
        searched_candidates[0] == original_candidates[0],
        f"{sample_id}: candidate 0 does not reproduce original evidence",
    )
    require(
        float(searched_candidates[int(searched_sidecar["selected_candidate_index"])]["score"])
        >= float(original_candidates[0]["score"]),
        f"{sample_id}: searched score regressed below original",
    )
    if int(searched_sidecar["selected_candidate_index"]) == 0:
        require(
            searched_sidecar["image_sha256"] == original_sidecar["image_sha256"],
            f"{sample_id}: selected baseline image differs from original",
        )


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                require(isinstance(value, dict), "metadata row is not an object")
                validate_metadata(value)
            except (json.JSONDecodeError, ValueError, DiffusionValidationError) as error:
                raise DiffusionValidationError(f"{path}:{line_number}: {error}") from error
            records.append(value)
    return records


def safe_png(root: Path, relative: str, expected_parent: str) -> Path:
    path = (root / relative).resolve(strict=True)
    parent = (root / expected_parent).resolve(strict=True)
    require(
        path.is_relative_to(parent) and path.suffix.lower() == ".png",
        f"Unsafe output path: {relative}",
    )
    return path


def expected_placement_map(
    *,
    paths: Any,
    config: dict[str, Any],
    objects: tuple[str, ...],
    n: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for object_name in objects:
        object_config = config["objects"][object_name]
        placements, _ = read_placements(
            paths=paths,
            object_name=object_name,
            expected_sha256=str(object_config["placements_sha256"]),
            n=n,
            defect_types=None,
        )
        for index, placement in enumerate(placements):
            placement["_validation_index"] = index
            sample_id = str(placement["placement_id"])
            require(sample_id not in result, f"Duplicate expected sample ID: {sample_id}")
            result[sample_id] = placement
    return result


def compare_record_to_placement(
    record: dict[str, Any],
    placement: dict[str, Any],
) -> None:
    sample_id = str(record["sample_id"])
    require(record["object"] == placement["object"], f"{sample_id}: object mismatch")
    require(
        record["defect_type"] == placement["defect_type"],
        f"{sample_id}: defect type mismatch",
    )
    require(
        record["trigger_token"] == placement["trigger_token"],
        f"{sample_id}: trigger token mismatch",
    )
    source = record["source"]
    expected_source = {
        "background_image": placement["background_image"],
        "background_sha256": placement["background_sha256"],
        "defect_source_component_id": placement["source_component_id"],
        "defect_source_image": placement["source_image"],
        "defect_source_mask": placement["source_mask"],
    }
    require(source == expected_source, f"{sample_id}: source provenance mismatch")
    expected_placement = {
        "affine": {
            key: placement["affine"][key] for key in ("dx", "dy", "rotation_deg", "scale", "flip")
        },
        "mask_area_px": placement["mask_area_px"],
        "mask_area_ratio": placement["mask_area_ratio"],
        "mask_bbox": placement["mask_bbox"],
        "roi_bbox": placement["roi_bbox"],
    }
    require(
        record["placement"] == expected_placement,
        f"{sample_id}: placement provenance mismatch",
    )


def expected_blend_support(
    mask: np.ndarray,
    crop_bbox: list[int],
    *,
    dilation_px: int,
    sigma: float,
) -> np.ndarray:
    x, y, width, height = (int(value) for value in crop_bbox)
    support = np.zeros_like(mask, dtype=bool)
    crop_mask = (mask[y : y + height, x : x + width] > 0).astype(np.uint8)
    if dilation_px:
        side = dilation_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
        crop_mask = cv2.dilate(crop_mask, kernel)
    if sigma:
        crop_mask = (
            cv2.GaussianBlur(
                crop_mask.astype(np.float32),
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
            )
            > 1e-7
        )
    support[y : y + height, x : x + width] = crop_mask > 0
    return support


def validate_dataset(
    *,
    paths_config: Path,
    generation_config: Path,
    out_name: str | None,
    bucket: str,
    n: int | None,
    requested_objects: list[str] | None,
) -> dict[str, Any]:
    paths = load_paths(paths_config)
    config = load_config(generation_config)
    sample_count = int(n if n is not None else config["generation"]["n"])
    require(sample_count > 0, "n must be positive")
    objects = tuple(requested_objects or paths.objects)
    require(len(set(objects)) == len(objects), "Duplicate --object")
    require(
        all(name in paths.objects and name in config["objects"] for name in objects),
        "Unknown object requested",
    )
    root = paths.synthetic / (out_name or str(config["output"]["name"])) / bucket
    records = read_records(root / "metadata.jsonl")
    selected_records = [record for record in records if record["object"] in objects]
    require(
        len(selected_records) == sample_count * len(objects),
        f"Expected {sample_count * len(objects)} records, found {len(selected_records)}",
    )
    expected = expected_placement_map(
        paths=paths,
        config=config,
        objects=objects,
        n=sample_count,
    )
    observed_ids = [str(record["sample_id"]) for record in selected_records]
    require(len(set(observed_ids)) == len(observed_ids), "Duplicate sample ID")
    require(set(observed_ids) == set(expected), "Sample ID inventory differs from M9 placements")

    expected_image_names = {f"{sample_id}.png" for sample_id in expected}
    expected_sidecar_names = {f"{sample_id}.json" for sample_id in expected}
    observed_image_names = {
        path.name
        for path in (root / "images").glob("*.png")
        if path.name.startswith(tuple(f"{name}__" for name in objects))
    }
    observed_mask_names = {
        path.name
        for path in (root / "masks").glob("*.png")
        if path.name.startswith(tuple(f"{name}__" for name in objects))
    }
    observed_sidecar_names = {
        path.name
        for path in (root / ".records").glob("*.json")
        if path.name.startswith(tuple(f"{name}__" for name in objects))
    }
    require(observed_image_names == expected_image_names, "Image tree inventory mismatch")
    require(observed_mask_names == expected_image_names, "Mask tree inventory mismatch")
    require(observed_sidecar_names == expected_sidecar_names, "Sidecar inventory mismatch")
    require(not list(root.rglob("*.tmp")), "Temporary files remain in output tree")

    manifest, manifest_sha256 = verify_frozen_manifest(paths.splits)
    selection_sha256, selection_name = read_checksum_file(paths.splits / "FEWSHOT_SELECTION.sha256")
    defect_types_sha256, defect_types_name = read_checksum_file(
        paths.splits / "DEFECT_TYPES.sha256"
    )
    require(
        sha256_file(paths.splits / selection_name) == selection_sha256,
        "Few-shot selection checksum mismatch",
    )
    require(
        sha256_file(paths.splits / defect_types_name) == defect_types_sha256,
        "Defect types checksum mismatch",
    )
    train_good = {
        str(record["image_path"]): record
        for record in manifest["images"]
        if record["set"] == "train" and record["label"] == "good"
    }
    blocklist = set(load_json(paths.splits / "test_blocklist.json")["sha256"])
    image_hashes: set[str] = set()
    mask_hashes: set[str] = set()
    source_hash_cache: dict[str, str] = {}
    score_values: list[float] = []
    visible_values: list[float] = []
    original_baselines_compared = 0
    score_deltas: list[float] = []
    selected_candidate_indices: Counter[str] = Counter()
    selected_guidance: Counter[str] = Counter()
    selected_crop_ratio: Counter[str] = Counter()
    type_counts: dict[str, Counter[str]] = {object_name: Counter() for object_name in objects}

    for index, record in enumerate(selected_records, start=1):
        sample_id = str(record["sample_id"])
        placement = expected[sample_id]
        compare_record_to_placement(record, placement)
        require(record["bucket"] == bucket, f"{sample_id}: bucket mismatch")
        require(
            record["generator"] == str(config["output"]["name"]),
            f"{sample_id}: generator mismatch",
        )
        require(
            record["pipeline_version"] == pipeline_version(bucket),
            f"{sample_id}: pipeline version mismatch",
        )
        try:
            datetime.fromisoformat(str(record["created_at"]))
        except ValueError as error:
            raise DiffusionValidationError(f"{sample_id}: invalid timestamp") from error

        object_name = str(record["object"])
        object_config = config["objects"][object_name]
        generation = record["generation"]
        require(
            generation["base_model"] == config["model"]["id"],
            f"{sample_id}: base model mismatch",
        )
        require(
            generation["model_resolution"] == config["model"]["resolution"],
            f"{sample_id}: model resolution mismatch",
        )
        _, expected_adapter_path = configured_adapter(paths, object_config)
        require(
            generation["lora_path"] == expected_adapter_path,
            f"{sample_id}: adapter path mismatch",
        )
        require(
            generation["num_inference_steps"] == int(config["generation"]["num_inference_steps"]),
            f"{sample_id}: inference steps mismatch",
        )
        require(
            generation["blend"] == config["generation"]["blend"],
            f"{sample_id}: blend mismatch",
        )

        sidecar_path = root / ".records" / f"{sample_id}.json"
        sidecar = load_json(sidecar_path)
        require(sidecar.get("record") == record, f"{sample_id}: sidecar record mismatch")
        require(
            sidecar.get("config_sha256") == sha256_file(generation_config),
            f"{sample_id}: generation config changed",
        )
        require(
            sidecar.get("model_revision") == config["model"]["revision"],
            f"{sample_id}: model revision mismatch",
        )
        require(
            sidecar.get("placement_index") == placement["_validation_index"],
            f"{sample_id}: placement index mismatch",
        )
        require(
            sidecar.get("placements_sha256") == object_config["placements_sha256"],
            f"{sample_id}: placement checksum mismatch",
        )
        candidates = sidecar.get("candidates")
        require(isinstance(candidates, list) and candidates, f"{sample_id}: no candidates")
        expected_candidates = int(config["refine"]["num_search_run"]) if bucket == "searched" else 1
        require(
            len(candidates) == expected_candidates,
            f"{sample_id}: candidate count mismatch",
        )
        require(
            int(sidecar.get("seed", -1)) == paths.seed,
            f"{sample_id}: root seed mismatch",
        )
        validate_candidate_schedule(
            candidates,
            bucket=bucket,
            config=config,
            object_name=object_name,
            placement_index=int(sidecar["placement_index"]),
            seed=int(sidecar["seed"]),
            sample_id=sample_id,
        )
        if bucket == "searched":
            original_sidecar_path = root.parent / "original" / ".records" / f"{sample_id}.json"
            require(
                original_sidecar_path.is_file(),
                f"{sample_id}: missing original baseline sidecar",
            )
            original_sidecar = load_json(original_sidecar_path)
            require(
                original_sidecar.get("pipeline_version") == pipeline_version("original"),
                f"{sample_id}: original baseline pipeline version mismatch",
            )
            validate_search_baseline(
                searched_sidecar=sidecar,
                original_sidecar=original_sidecar,
                sample_id=sample_id,
            )
            original_baselines_compared += 1
        selected_index = int(sidecar["selected_candidate_index"])
        selected = [
            candidate
            for candidate in candidates
            if int(candidate["candidate_index"]) == selected_index
        ]
        require(len(selected) == 1, f"{sample_id}: selected candidate missing")
        selected_candidate = selected[0]
        require(
            int(selected_candidate["generator_seed"]) == int(generation["seed"])
            and float(selected_candidate["guidance_scale"]) == float(generation["guidance_scale"])
            and float(selected_candidate["crop_ratio"]) == float(generation["crop_ratio"]),
            f"{sample_id}: selected candidate metadata mismatch",
        )
        require(
            float(selected_candidate["score"])
            == max(float(candidate["score"]) for candidate in candidates),
            f"{sample_id}: selected candidate is not the recorded optimum",
        )
        score_values.append(float(selected_candidate["score"]))
        visible_values.append(float(selected_candidate["visible_change"]))
        if bucket == "searched":
            original_score = float(original_sidecar["candidates"][0]["score"])
            score_deltas.append(float(selected_candidate["score"]) - original_score)
            selected_candidate_indices[str(selected_index)] += 1
            selected_guidance[f"{float(selected_candidate['guidance_scale']):g}"] += 1
            selected_crop_ratio[f"{float(selected_candidate['crop_ratio']):g}"] += 1

        background_relative = str(record["source"]["background_image"])
        background_record = train_good.get(background_relative)
        require(background_record is not None, f"{sample_id}: background is not train-good")
        background_path = paths.visa_raw / background_relative
        for relative in (
            background_relative,
            str(record["source"]["defect_source_image"]),
            str(record["source"]["defect_source_mask"]),
        ):
            if relative not in source_hash_cache:
                source_hash_cache[relative] = sha256_file(paths.visa_raw / relative)
            require(
                source_hash_cache[relative] not in blocklist,
                f"{sample_id}: source hit test blocklist",
            )
        require(
            source_hash_cache[background_relative]
            == record["source"]["background_sha256"]
            == background_record["sha256"],
            f"{sample_id}: background checksum mismatch",
        )

        image_path = safe_png(root, str(record["image_path"]), "images")
        mask_path = safe_png(root, str(record["mask_path"]), "masks")
        placement_mask_path = (
            paths.synthetic / "placements" / object_name / str(placement["mask_path"])
        )
        image_sha256 = sha256_file(image_path)
        mask_sha256 = sha256_file(mask_path)
        require(
            image_sha256 == sidecar["image_sha256"] and mask_sha256 == sidecar["mask_sha256"],
            f"{sample_id}: output hash mismatch",
        )
        require(
            mask_sha256 == sha256_file(placement_mask_path),
            f"{sample_id}: GT mask differs from M9 placement",
        )
        require(
            image_sha256 not in blocklist and mask_sha256 not in blocklist,
            f"{sample_id}: output hit test blocklist",
        )
        image_hashes.add(image_sha256)
        mask_hashes.add(mask_sha256)

        with Image.open(background_path) as handle:
            background = np.asarray(handle.convert("RGB"))
        with Image.open(image_path) as handle:
            generated = np.asarray(handle.convert("RGB"))
        with Image.open(mask_path) as handle:
            mask = np.asarray(handle.convert("L"))
        require(
            generated.shape == background.shape and mask.shape == background.shape[:2],
            f"{sample_id}: full-resolution shape mismatch",
        )
        require(
            {int(value) for value in np.unique(mask)}.issubset({0, 255}) and np.any(mask),
            f"{sample_id}: mask is not non-empty binary PNG",
        )
        require(
            int((mask > 0).sum()) == int(record["placement"]["mask_area_px"]),
            f"{sample_id}: mask area mismatch",
        )
        support = expected_blend_support(
            mask,
            generation["crop_bbox"],
            dilation_px=int(config["generation"]["blend_dilation_px"]),
            sigma=float(config["generation"]["blend_feather_sigma"]),
        )
        require(
            bool(np.array_equal(generated[~support], background[~support])),
            f"{sample_id}: pixels changed outside declared blend support",
        )
        type_counts[object_name][str(record["defect_type"])] += 1
        if index % 100 == 0:
            LOGGER.info("Validated %d/%d Stage B records", index, len(selected_records))

    require(
        len(image_hashes) == len(selected_records),
        "Two Stage B records produced identical image bytes",
    )
    require(
        len(mask_hashes) == len(selected_records),
        "Two Stage B records unexpectedly share identical masks",
    )
    for object_name in objects:
        object_config = config["objects"][object_name]
        adapter, _ = configured_adapter(paths, object_config)
        require(
            sha256_file(adapter / "unet_adapter" / "adapter_model.safetensors")
            == object_config["unet_adapter_sha256"],
            f"{object_name}: UNet adapter changed",
        )
        require(
            sha256_file(adapter / "text_token_adapter" / "adapter_model.safetensors")
            == object_config["text_token_adapter_sha256"],
            f"{object_name}: text adapter changed",
        )
        if "text_token_adapter_2_sha256" in object_config:
            require(
                sha256_file(adapter / "text_token_adapter_2" / "adapter_model.safetensors")
                == object_config["text_token_adapter_2_sha256"],
                f"{object_name}: second text adapter changed",
            )

    _, final_manifest_sha256 = verify_frozen_manifest(paths.splits)
    require(final_manifest_sha256 == manifest_sha256, "Frozen manifest changed during validation")
    return {
        "bucket": bucket,
        "config_sha256": sha256_file(generation_config),
        "defect_types_sha256": defect_types_sha256,
        "manifest_sha256": manifest_sha256,
        "mean_refine_score": float(np.mean(score_values)),
        "mean_visible_change": float(np.mean(visible_values)),
        "original_baselines_compared": original_baselines_compared,
        "refine_selection": (
            {
                "crop_ratio_counts": dict(sorted(selected_crop_ratio.items())),
                "guidance_scale_counts": dict(sorted(selected_guidance.items())),
                "score_delta": {
                    "equal": sum(delta == 0 for delta in score_deltas),
                    "improved": sum(delta > 0 for delta in score_deltas),
                    "maximum": max(score_deltas),
                    "mean": float(np.mean(score_deltas)),
                    "minimum": min(score_deltas),
                    "regressed": sum(delta < 0 for delta in score_deltas),
                },
                "selected_candidate_index_counts": dict(sorted(selected_candidate_indices.items())),
            }
            if score_deltas
            else None
        ),
        "objects": {
            object_name: dict(sorted(type_counts[object_name].items())) for object_name in objects
        },
        "output": str(root),
        "records": len(selected_records),
        "selection_sha256": selection_sha256,
        "status": "passed",
        "test_blocklist_hits": 0,
        "unique_image_sha256": len(image_hashes),
        "unique_mask_sha256": len(mask_hashes),
        "unique_source_files_hashed": len(source_hash_cache),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        result = validate_dataset(
            paths_config=args.paths,
            generation_config=args.config,
            out_name=args.out_name,
            bucket=args.bucket,
            n=args.n,
            requested_objects=args.objects,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            destination = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except (
        DiffusionValidationError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
