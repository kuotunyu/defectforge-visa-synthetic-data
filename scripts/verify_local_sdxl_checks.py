"""Verify M11 local SDXL smoke and checkpoint-resume evidence without loading models."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_lora_run import validate_adapter_configs
from src.common.integrity import (
    assert_not_blocklisted,
    read_checksum_file,
    sha256_file,
)
from src.common.paths import Paths, load_paths
from src.training.train_inpaint_lora import load_config

REQUIRED_BUNDLE_FILES = (
    "trainer_state.json",
    "training_state.pt",
    "unet_adapter/adapter_config.json",
    "unet_adapter/adapter_model.safetensors",
    "text_token_adapter/adapter_config.json",
    "text_token_adapter/adapter_model.safetensors",
    "text_token_adapter_2/adapter_config.json",
    "text_token_adapter_2/adapter_model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer_2/tokenizer.json",
)
LOCK_FIELDS = (
    "alpha",
    "base_model",
    "base_model_revision",
    "component_samples",
    "defect_types_sha256",
    "learning_rate",
    "manifest_sha256",
    "model_family",
    "object",
    "pipeline_version",
    "rank",
    "resolution",
    "seed",
    "selection_sha256",
    "source_images",
    "token_ids",
    "token_learning_rate",
    "training_config_sha256",
    "type_counts",
)


class LocalSdxlValidationError(RuntimeError):
    """Raised when local M11 evidence violates the locked contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalSdxlValidationError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalSdxlValidationError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise LocalSdxlValidationError(f"Expected JSON object: {path}")
    return value


def compare_fields(
    actual: dict[str, Any],
    expected: dict[str, Any],
    fields: tuple[str, ...],
    *,
    label: str,
) -> None:
    for field in fields:
        require(actual.get(field) == expected.get(field), f"{label}: {field} mismatch")


def frozen_hashes(paths: Paths) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, checksum_name in (
        ("manifest_sha256", "MANIFEST.sha256"),
        ("selection_sha256", "FEWSHOT_SELECTION.sha256"),
        ("defect_types_sha256", "DEFECT_TYPES.sha256"),
    ):
        expected, filename = read_checksum_file(paths.splits / checksum_name)
        require(
            sha256_file(paths.splits / filename) == expected,
            f"Frozen checksum mismatch: {filename}",
        )
        result[key] = expected
    return result


def validate_model_cache(
    cache_root: Path,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    model_id = str(config["model"]["id"])
    revision = str(config["model"]["revision"])
    model_cache = cache_root / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshot = model_cache / "snapshots" / revision
    require(snapshot.is_dir(), f"Locked model cache is missing: {snapshot}")
    evidence: dict[str, dict[str, Any]] = {}
    for relative, expected_sha256 in config["model"]["expected_files"].items():
        path = snapshot / str(relative)
        require(path.is_file(), f"Cached model file is missing: {relative}")
        actual_sha256 = sha256_file(path)
        require(actual_sha256 == expected_sha256, f"Cached model hash mismatch: {relative}")
        evidence[str(relative)] = {
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
        }
    return evidence


def adapter_hashes(bundle: Path) -> dict[str, str]:
    return {
        "unet_adapter_sha256": sha256_file(bundle / "unet_adapter" / "adapter_model.safetensors"),
        "text_token_adapter_sha256": sha256_file(
            bundle / "text_token_adapter" / "adapter_model.safetensors"
        ),
        "text_token_adapter_2_sha256": sha256_file(
            bundle / "text_token_adapter_2" / "adapter_model.safetensors"
        ),
    }


def validate_bundle(
    bundle: Path,
    *,
    config: dict[str, Any],
    expected_state: dict[str, Any],
    label: str,
) -> dict[str, str]:
    for relative in REQUIRED_BUNDLE_FILES:
        path = bundle / relative
        require(path.is_file() and path.stat().st_size > 0, f"{label}: missing {relative}")
    state = load_json(bundle / "trainer_state.json")
    compare_fields(state, expected_state, LOCK_FIELDS, label=f"{label} trainer state")
    require(
        state.get("global_step") == expected_state.get("global_step")
        and state.get("micro_step") == expected_state.get("micro_step"),
        f"{label}: progress mismatch",
    )
    validate_adapter_configs(bundle, config)
    observed_hashes = adapter_hashes(bundle)
    for key, digest in observed_hashes.items():
        if key in expected_state:
            require(expected_state[key] == digest, f"{label}: {key} mismatch")
    return observed_hashes


def validate_sample(
    object_root: Path,
    *,
    step: int,
    object_name: str,
    paths: Paths,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    png_path = object_root / "samples" / f"step_{step:06d}.png"
    sidecar = load_json(png_path.with_suffix(".json"))
    require(png_path.is_file(), f"{object_name}: sample PNG is missing")
    with Image.open(png_path) as image:
        image.verify()
        size = image.size
    resolution = int(config["model"]["resolution"])
    require(size == (resolution * 3, resolution), f"{object_name}: panel size mismatch")
    require(sidecar.get("step") == step, f"{object_name}: sample step mismatch")
    require(
        sidecar.get("model_id") == config["model"]["id"]
        and sidecar.get("model_revision") == config["model"]["revision"],
        f"{object_name}: sample model lock mismatch",
    )
    require(
        sidecar.get("panel_sha256") == sha256_file(png_path),
        f"{object_name}: sample panel hash mismatch",
    )
    background = paths.visa_raw / str(sidecar["background_image"])
    require(
        sidecar.get("background_sha256") == sha256_file(background),
        f"{object_name}: sample background hash mismatch",
    )
    return (
        {
            "file": png_path.relative_to(object_root).as_posix(),
            "panel_sha256": sidecar["panel_sha256"],
            "placement_id": sidecar["placement_id"],
            "trigger_token": sidecar["trigger_token"],
        },
        background,
    )


def validate_report_locks(
    report: dict[str, Any],
    *,
    object_name: str,
    config: dict[str, Any],
    config_sha256: str,
    frozen: dict[str, str],
) -> None:
    require(report.get("status") == "passed", f"{object_name}: report did not pass")
    require(report.get("object") == object_name, f"{object_name}: object mismatch")
    require(report.get("model_family") == "sdxl", f"{object_name}: family mismatch")
    require(report.get("pipeline_version") == "0.3.0", f"{object_name}: pipeline mismatch")
    require(
        report.get("base_model") == config["model"]["id"]
        and report.get("base_model_revision") == config["model"]["revision"],
        f"{object_name}: base model lock mismatch",
    )
    require(
        report.get("training_config_sha256") == config_sha256,
        f"{object_name}: training config changed",
    )
    for key, digest in frozen.items():
        require(report.get(key) == digest, f"{object_name}: frozen {key} mismatch")
    require(report.get("reload_class") == "PeftModel", f"{object_name}: reload failed")
    require(float(report.get("peak_vram_gib", 0.0)) > 0.0, f"{object_name}: VRAM missing")


def validate_smoke(
    *,
    smoke_root: Path,
    paths: Paths,
    config: dict[str, Any],
    config_sha256: str,
    frozen: dict[str, str],
) -> tuple[dict[str, Any], list[Path]]:
    evidence: dict[str, Any] = {}
    backgrounds: list[Path] = []
    for object_name in paths.objects:
        object_root = smoke_root / object_name
        report = load_json(object_root / "training_report.json")
        validate_report_locks(
            report,
            object_name=object_name,
            config=config,
            config_sha256=config_sha256,
            frozen=frozen,
        )
        require(
            report.get("global_step") == report.get("micro_step") == 1
            and report.get("max_train_steps") == 1,
            f"{object_name}: smoke did not complete exactly one step",
        )
        checkpoint_names = sorted(path.name for path in object_root.glob("checkpoint-*"))
        require(
            checkpoint_names == ["checkpoint-000001"],
            f"{object_name}: smoke checkpoint mismatch",
        )
        final_hashes = validate_bundle(
            object_root / "final",
            config=config,
            expected_state=report,
            label=f"{object_name} smoke final",
        )
        for key, digest in final_hashes.items():
            require(report.get(key) == digest, f"{object_name}: report {key} mismatch")
        sample, background = validate_sample(
            object_root,
            step=1,
            object_name=object_name,
            paths=paths,
            config=config,
        )
        require(
            report.get("sample_files") == [sample["file"]],
            f"{object_name}: sample inventory mismatch",
        )
        backgrounds.append(background)
        evidence[object_name] = {
            "adapter_hashes": final_hashes,
            "checkpoint_names": checkpoint_names,
            "last_loss": report["last_loss"],
            "peak_vram_gib": report["peak_vram_gib"],
            "reload_class": report["reload_class"],
            "sample": sample,
            "steps": report["global_step"],
            "training_elapsed_seconds": report["training_elapsed_seconds"],
            "wall_clock_seconds": report["wall_clock_seconds"],
        }
    return evidence, backgrounds


def validate_resume_progression(
    checkpoint_state: dict[str, Any],
    final_state: dict[str, Any],
) -> None:
    compare_fields(
        final_state,
        checkpoint_state,
        (*LOCK_FIELDS, "run_signature"),
        label="Resume progression",
    )
    require(
        checkpoint_state.get("global_step") == checkpoint_state.get("micro_step") == 1,
        "Resume source is not step 1",
    )
    require(
        final_state.get("global_step") == final_state.get("micro_step") == 2,
        "Resume final is not step 2",
    )
    require(
        checkpoint_state.get("max_train_steps") == final_state.get("max_train_steps") == 2,
        "Resume max-step contract mismatch",
    )
    require(
        float(final_state.get("training_elapsed_seconds", 0.0))
        > float(checkpoint_state.get("training_elapsed_seconds", 0.0)),
        "Resume did not accumulate training time",
    )
    require(
        str(final_state.get("created_at")) > str(checkpoint_state.get("created_at")),
        "Resume final was not created after its source checkpoint",
    )


def validate_resume(
    *,
    resume_root: Path,
    paths: Paths,
    config: dict[str, Any],
    config_sha256: str,
    frozen: dict[str, str],
) -> tuple[dict[str, Any], Path]:
    report = load_json(resume_root / "training_report.json")
    validate_report_locks(
        report,
        object_name="pcb1",
        config=config,
        config_sha256=config_sha256,
        frozen=frozen,
    )
    checkpoint_names = sorted(path.name for path in resume_root.glob("checkpoint-*"))
    require(
        checkpoint_names == ["checkpoint-000001", "checkpoint-000002"],
        "Resume checkpoint inventory mismatch",
    )
    checkpoint_state = load_json(resume_root / "checkpoint-000001" / "trainer_state.json")
    final_state = load_json(resume_root / "final" / "trainer_state.json")
    validate_resume_progression(checkpoint_state, final_state)
    compare_fields(report, final_state, (*LOCK_FIELDS, "run_signature"), label="Resume report")
    require(
        report.get("global_step") == report.get("micro_step") == 2,
        "Resume report progress mismatch",
    )
    checkpoint_hashes = validate_bundle(
        resume_root / "checkpoint-000001",
        config=config,
        expected_state=checkpoint_state,
        label="Resume checkpoint 1",
    )
    final_hashes = validate_bundle(
        resume_root / "final",
        config=config,
        expected_state=report,
        label="Resume final",
    )
    for key, digest in final_hashes.items():
        require(report.get(key) == digest, f"Resume report {key} mismatch")
    sample, background = validate_sample(
        resume_root,
        step=2,
        object_name="pcb1",
        paths=paths,
        config=config,
    )
    require(report.get("sample_files") == [sample["file"]], "Resume sample mismatch")
    return (
        {
            "checkpoint_names": checkpoint_names,
            "final_adapter_hashes": final_hashes,
            "final_global_step": report["global_step"],
            "final_loss": report["last_loss"],
            "peak_vram_gib": report["peak_vram_gib"],
            "reload_class": report["reload_class"],
            "run_signature": report["run_signature"],
            "sample": sample,
            "source_adapter_hashes": checkpoint_hashes,
            "source_checkpoint": "checkpoint-000001",
            "source_global_step": checkpoint_state["global_step"],
            "training_elapsed_seconds": report["training_elapsed_seconds"],
            "wall_clock_seconds": report["wall_clock_seconds"],
        },
        background,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/lora_sdxl.yaml"))
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--smoke-root", type=Path)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/lora_sdxl_local_validation.json"),
    )
    args = parser.parse_args()

    paths = load_paths(args.paths)
    config = load_config(args.config)
    require(config["model"]["family"] == "sdxl", "Expected an SDXL config")
    cache_root = (args.cache_root or paths.cache / "huggingface").resolve(strict=True)
    smoke_root = (args.smoke_root or paths.runs / "lora_sdxl_smoke").resolve(strict=True)
    resume_root = (args.resume_root or paths.runs / "lora_sdxl_resume_check" / "pcb1").resolve(
        strict=True
    )
    config_sha256 = sha256_file(args.config)
    frozen = frozen_hashes(paths)
    model_files = validate_model_cache(cache_root, config)
    smoke, backgrounds = validate_smoke(
        smoke_root=smoke_root,
        paths=paths,
        config=config,
        config_sha256=config_sha256,
        frozen=frozen,
    )
    resume, resume_background = validate_resume(
        resume_root=resume_root,
        paths=paths,
        config=config,
        config_sha256=config_sha256,
        frozen=frozen,
    )
    assert_not_blocklisted(
        [*backgrounds, resume_background],
        paths.splits / "test_blocklist.json",
    )
    result = {
        "base_model": config["model"]["id"],
        "base_model_revision": config["model"]["revision"],
        "frozen_sha256": frozen,
        "model_cache": {
            "bytes": sum(int(item["bytes"]) for item in model_files.values()),
            "files": model_files,
        },
        "pipeline_version": "0.3.0",
        "resume": resume,
        "smoke": smoke,
        "status": "passed",
        "test_blocklist_hits": 0,
        "validated_at": datetime.now(UTC).isoformat(),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
