"""Independently validate one object's eight formal M18 SegFormer run directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from safetensors import safe_open
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import load_json, sha256_file, verify_frozen_manifest
from src.common.paths import load_paths

FORMAL_GROUPS = (
    "real_only",
    "std_aug",
    "unfiltered_syn",
    "filtered_syn",
    "full_real",
    "procedural_only",
    "copypaste_only",
    "diffusion_only",
)
METRICS = ("dice", "miou", "pixel_auroc", "aupro")


class SegmenterValidationError(RuntimeError):
    """Raised when returned M18 results violate their frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SegmenterValidationError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON mapping: {path}")
    return value


def _frozen_test_inventory(
    paths: Any,
    object_name: str,
) -> tuple[set[str], dict[str, str | None], str]:
    manifest, manifest_sha256 = verify_frozen_manifest(paths.splits)
    rows = [
        row
        for row in manifest["images"]
        if row["object"] == object_name and row["set"] == "test"
    ]
    images = {str(row["sha256"]) for row in rows}
    masks = {
        str(row["sha256"]): (
            None if row.get("mask_sha256") is None else str(row["mask_sha256"])
        )
        for row in rows
    }
    return images, masks, manifest_sha256


def _validate_data_manifest(
    data: Mapping[str, Any],
    *,
    group_name: str,
    object_name: str,
    expected_test_images: set[str],
    expected_test_masks: Mapping[str, str | None],
    blocklist: set[str],
) -> dict[str, int]:
    require(data["canonical_group"] == group_name, f"Canonical group changed: {group_name}")
    require(data["object"] == object_name, f"Object changed: {group_name}")
    require(data["mode"] == "final", f"Non-final data manifest: {group_name}")
    require(data["validation"] == [], f"Final validation is not empty: {group_name}")
    train = data["train"]
    test = data["test"]
    require(isinstance(train, list) and isinstance(test, list), "Invalid data partitions")
    train_hashes = {str(row["image_sha256"]) for row in train}
    test_hashes = {str(row["image_sha256"]) for row in test}
    require(len(train_hashes) == len(train), f"Duplicate train image: {group_name}")
    require(test_hashes == expected_test_images, f"Frozen test inventory changed: {group_name}")
    require(not (train_hashes & test_hashes), f"Train/test overlap: {group_name}")
    require(not (train_hashes & blocklist), f"Training image hits blocklist: {group_name}")
    for row in train:
        for reference in row["manifest_refs"]:
            require(isinstance(reference, str) and reference, "Empty provenance reference")
    for row in test:
        require(row["kind"] == "real", f"Synthetic sample entered test: {group_name}")
        expected_mask = expected_test_masks[str(row["image_sha256"])]
        require(row["mask_sha256"] == expected_mask, f"Frozen test mask changed: {group_name}")
        require(str(row["image_sha256"]) in blocklist, "Test image is absent from blocklist")
        if expected_mask is not None:
            require(expected_mask in blocklist, "Test mask is absent from blocklist")
    real_defects = sum(row["kind"] == "real" and row["has_defect"] for row in train)
    synthetic_defects = sum(row["kind"] == "synthetic" for row in train)
    if group_name == "procedural_only":
        require(real_defects == 0, "procedural_only contains real defect pixels")
        require(synthetic_defects == 500, "procedural_only synthetic count changed")
    return {
        "train": len(train),
        "test": len(test),
        "real_defects": real_defects,
        "synthetic_defects": synthetic_defects,
    }


def _validate_model(
    run_dir: Path,
    *,
    report: Mapping[str, Any],
    reload_model: bool,
) -> None:
    final_dir = run_dir / "final"
    weight_path = final_dir / "model.safetensors"
    config_path = final_dir / "config.json"
    require(weight_path.is_file() and config_path.is_file(), f"Final model is incomplete: {run_dir}")
    require(sha256_file(weight_path) == report["model_sha256"], "Final model hash changed")
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    require("decode_head.classifier.weight" in keys, "SegFormer classifier weight is missing")
    require("decode_head.classifier.bias" in keys, "SegFormer classifier bias is missing")
    config = _load_mapping(config_path)
    require(config["id2label"] in ({"0": "defect"}, {0: "defect"}), "SegFormer label map changed")
    if reload_model:
        model = SegformerForSemanticSegmentation.from_pretrained(
            final_dir,
            local_files_only=True,
            use_safetensors=True,
        )
        require(model.config.num_labels == 1, "Fresh SegFormer reload changed num_labels")


def _validate_training_budget(
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    run_name: str,
) -> None:
    expected = int(config["training"]["total_steps"])
    require(
        int(report["requested_total_steps"]) == expected,
        f"Requested optimizer-step budget changed: {run_name}",
    )
    require(
        int(report["executed_steps"]) == expected,
        f"Executed optimizer-step budget changed: {run_name}",
    )


def validate(
    *,
    paths: Any,
    config: Mapping[str, Any],
    run_root: Path,
    object_name: str,
    reload_model: bool,
) -> dict[str, Any]:
    expected_test_images, expected_test_masks, manifest_sha256 = _frozen_test_inventory(
        paths,
        object_name,
    )
    blocklist_payload = load_json(paths.splits / "test_blocklist.json")
    blocklist = {str(value) for value in blocklist_payload["sha256"]}
    expected_selection_sha, selection_name = (
        paths.splits / "FEWSHOT_SELECTION.sha256"
    ).read_text(encoding="utf-8").strip().split(maxsplit=1)
    require(
        sha256_file(paths.splits / selection_name) == expected_selection_sha,
        "Few-shot selection checksum changed",
    )
    summaries: dict[str, Any] = {}
    observed_run_dirs = {
        path.name
        for path in run_root.iterdir()
        if path.is_dir() and path.name.startswith("m18_") and f"_{object_name}_" in path.name
    }
    expected_run_dirs = {
        f"m18_{group}_{object_name}_seed42" for group in FORMAL_GROUPS
    }
    require(expected_run_dirs <= observed_run_dirs, "One or more formal M18 runs are missing")
    require(
        f"m18_all_mixed_{object_name}_seed42" not in observed_run_dirs,
        "all_mixed alias was rerun",
    )

    for group_name in FORMAL_GROUPS:
        run_name = f"m18_{group_name}_{object_name}_seed42"
        run_dir = run_root / run_name
        report = _load_mapping(run_dir / "training_report.json")
        run_config = _load_mapping(run_dir / "run_config.json")
        data = _load_mapping(run_dir / "data_manifest.json")
        signature_payload = dict(run_config)
        signature = str(signature_payload.pop("run_signature"))
        require(canonical_sha256(signature_payload) == signature, f"Run signature changed: {run_name}")
        require(report["run_signature"] == signature, f"Report signature changed: {run_name}")
        require(report["status"] == "passed", f"Run did not pass: {run_name}")
        require(report["mode"] == "final" and not report["smoke"], f"Run is not formal: {run_name}")
        require(report["canonical_group"] == group_name, f"Report group changed: {run_name}")
        require(report["object"] == object_name and report["seed"] == 42, "Run identity changed")
        require(report["model_repository"] == config["model"]["repository"], "Model repo changed")
        require(report["model_revision"] == config["model"]["revision"], "Model revision changed")
        require(report["base_weight_sha256"] == config["model"]["sha256"], "Base hash changed")
        require(report["input_size"] == config["model"]["input_size"], "Input size changed")
        _validate_training_budget(report, config, run_name=run_name)
        for metric_name in METRICS:
            value = float(report["metrics"][metric_name])
            require(math.isfinite(value) and 0.0 <= value <= 1.0, f"Invalid {metric_name}")
        data_sha = canonical_sha256(data)
        require(run_config["data_manifest_sha256"] == data_sha, "Data manifest hash changed")
        require(report["data_manifest_sha256"] == data_sha, "Report data hash changed")
        require(report["split_manifest_sha256"] == manifest_sha256, "Split manifest changed")
        require(report["selection_sha256"] == expected_selection_sha, "Selection changed")
        counts = _validate_data_manifest(
            data,
            group_name=group_name,
            object_name=object_name,
            expected_test_images=expected_test_images,
            expected_test_masks=expected_test_masks,
            blocklist=blocklist,
        )
        _validate_model(run_dir, report=report, reload_model=reload_model)
        summaries[group_name] = {
            **counts,
            "metrics": {name: float(report["metrics"][name]) for name in METRICS},
            "model_sha256": report["model_sha256"],
            "run_signature": signature,
        }

    return {
        "status": "passed",
        "schema_version": 1,
        "object": object_name,
        "runs": len(summaries),
        "all_mixed_alias_of": "filtered_syn",
        "alias_reruns": 0,
        "split_manifest_sha256": manifest_sha256,
        "selection_sha256": expected_selection_sha,
        "fresh_model_reload": reload_model,
        "groups": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/segmenter.yaml"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--object", dest="object_name", required=True)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = load_paths(args.paths)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "Invalid segmenter config")
    require(args.object_name in paths.objects, f"Unsupported object: {args.object_name}")
    payload = validate(
        paths=paths,
        config=config,
        run_root=args.run_root.resolve(strict=True),
        object_name=args.object_name,
        reload_model=args.reload,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
