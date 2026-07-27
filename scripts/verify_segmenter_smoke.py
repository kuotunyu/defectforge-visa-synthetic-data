"""Verify both M18 one-step development smoke runs without allocating CUDA."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file, verify_frozen_manifest
from src.common.paths import load_paths


class SegmenterSmokeError(RuntimeError):
    """Raised when an M18 smoke artefact violates its development contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SegmenterSmokeError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON mapping: {path}")
    return value


def verify_run(
    *,
    run_dir: Path,
    object_name: str,
    config: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    report = load_mapping(run_dir / "training_report.json")
    run_config = load_mapping(run_dir / "run_config.json")
    data = load_mapping(run_dir / "data_manifest.json")
    signature_payload = dict(run_config)
    signature = str(signature_payload.pop("run_signature"))
    require(canonical_sha256(signature_payload) == signature, "Smoke run signature changed")
    require(report["run_signature"] == signature, "Smoke report signature changed")
    require(report["status"] == "passed", "Smoke run did not pass")
    require(report["smoke"] is True and report["mode"] == "development", "Smoke mode changed")
    require(report["object"] == object_name, "Smoke object changed")
    require(
        report["requested_group"] == "real_only"
        and report["canonical_group"] == "real_only",
        "Smoke group changed",
    )
    require(
        report["requested_total_steps"] == 1 and report["executed_steps"] == 1,
        "Smoke must execute exactly one optimizer step",
    )
    require(report["model_repository"] == config["model"]["repository"], "Model repo changed")
    require(report["model_revision"] == config["model"]["revision"], "Model revision changed")
    require(report["base_weight_sha256"] == config["model"]["sha256"], "Base hash changed")
    require(report["input_size"] == 512, "Smoke input size changed")
    require(math.isfinite(float(report["peak_vram_gib"])), "Smoke peak VRAM is invalid")
    require(0.0 < float(report["peak_vram_gib"]) < 14.0, "Smoke exceeds T4 VRAM budget")
    require(data["mode"] == "development", "Smoke data mode changed")
    require(data["test"] == [], "Development smoke loaded the frozen test")
    require(data["validation"], "Development smoke has no validation")
    require(
        all(row["kind"] == "real" for row in data["validation"]),
        "Development validation contains synthetic data",
    )
    require(data["split_manifest_sha256"] == manifest_sha256, "Split manifest changed")
    data_sha = canonical_sha256(data)
    require(run_config["data_manifest_sha256"] == data_sha, "Smoke data hash changed")
    require(report["data_manifest_sha256"] == data_sha, "Smoke report data hash changed")
    for name in ("dice", "miou", "pixel_auroc", "aupro"):
        value = float(report["metrics"][name])
        require(math.isfinite(value) and 0.0 <= value <= 1.0, f"Invalid smoke metric: {name}")
    model_path = run_dir / "final" / "model.safetensors"
    require(model_path.is_file(), "Smoke final SafeTensors is missing")
    require(sha256_file(model_path) == report["model_sha256"], "Smoke model hash changed")
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        require(
            "decode_head.classifier.weight" in set(handle.keys()),
            "Smoke classifier weight is missing",
        )
    return {
        "status": "passed",
        "run_signature": signature,
        "model_sha256": report["model_sha256"],
        "peak_vram_gib": float(report["peak_vram_gib"]),
        "wall_clock_seconds": float(report["wall_clock_seconds"]),
        "metrics": {name: float(report["metrics"][name]) for name in report["metrics"]},
        "train": len(data["train"]),
        "validation": len(data["validation"]),
        "test": len(data["test"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/segmenter.yaml"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/segmenter_smoke_validation.json"))
    args = parser.parse_args()
    paths = load_paths(args.paths)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "Invalid segmenter config")
    _, manifest_sha256 = verify_frozen_manifest(paths.splits)
    run_root = (args.run_root or paths.runs / "seg_smoke").resolve(strict=True)
    runs = {
        object_name: verify_run(
            run_dir=run_root / f"m18_smoke_{object_name}_v1",
            object_name=object_name,
            config=config,
            manifest_sha256=manifest_sha256,
        )
        for object_name in paths.objects
    }
    payload = {
        "status": "passed",
        "schema_version": 1,
        "pipeline_version": config["pipeline_version"],
        "model_repository": config["model"]["repository"],
        "model_revision": config["model"]["revision"],
        "base_weight_sha256": config["model"]["sha256"],
        "objects": runs,
        "test_loaded": 0,
        "validation_is_real_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
