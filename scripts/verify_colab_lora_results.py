"""Verify downloaded M11 Colab artifacts without loading SDXL base weights."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_lora_run import validate_adapter_configs
from src.common.integrity import read_checksum_file, sha256_file
from src.common.paths import Paths, load_paths
from src.training.train_inpaint_lora import load_config


class ColabResultValidationError(RuntimeError):
    """Raised when a downloaded Colab result violates the M11 contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ColabResultValidationError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ColabResultValidationError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ColabResultValidationError(f"Expected JSON object: {path}")
    return value


def validate_archive(path: Path) -> dict[str, Any]:
    """Validate the user-downloaded ZIP before treating it as evidence."""

    require(path.is_file(), f"Downloaded archive is missing: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            require(archive.testzip() is None, "Downloaded archive failed CRC validation")
    except zipfile.BadZipFile as exc:
        raise ColabResultValidationError(f"Invalid ZIP archive: {path}") from exc
    require(names, "Downloaded archive is empty")
    require(len(names) == len(set(names)), "Downloaded archive contains duplicate members")
    for name in names:
        member = PurePosixPath(name)
        require(
            not member.is_absolute() and ".." not in member.parts,
            f"Unsafe archive member: {name}",
        )
    return {
        "bytes": path.stat().st_size,
        "file": path.name,
        "members": len(names),
        "sha256": sha256_file(path),
    }


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


def compare_fields(
    actual: dict[str, Any],
    expected: dict[str, Any],
    fields: tuple[str, ...],
    *,
    label: str,
) -> None:
    for field in fields:
        require(
            actual.get(field) == expected.get(field),
            f"{label}: {field} mismatch",
        )


def validate_object(
    *,
    object_name: str,
    results_root: Path,
    paths: Paths,
    config: dict[str, Any],
    config_sha256: str,
    frozen: dict[str, str],
    upstream: dict[str, Any],
    blocked: set[str],
) -> dict[str, Any]:
    object_root = results_root / object_name
    final_dir = object_root / "final"
    report = load_json(object_root / "training_report.json")
    trainer_state = load_json(final_dir / "trainer_state.json")
    max_steps = int(config["training"]["max_train_steps"])
    expected_steps = list(
        range(
            int(config["training"]["sample_every"]),
            max_steps + 1,
            int(config["training"]["sample_every"]),
        )
    )
    require(report.get("status") == "passed", f"{object_name}: training report did not pass")
    require(report.get("object") == object_name, f"{object_name}: object mismatch")
    require(report.get("model_family") == "sdxl", f"{object_name}: family mismatch")
    require(report.get("pipeline_version") == "0.3.0", f"{object_name}: pipeline mismatch")
    require(report.get("global_step") == max_steps, f"{object_name}: step mismatch")
    require(report.get("micro_step") == max_steps, f"{object_name}: micro-step mismatch")
    require(
        report.get("training_config_sha256") == config_sha256,
        f"{object_name}: training config changed",
    )
    require(
        report.get("base_model") == config["model"]["id"]
        and report.get("base_model_revision") == config["model"]["revision"],
        f"{object_name}: base model lock mismatch",
    )
    for key, expected in frozen.items():
        require(report.get(key) == expected, f"{object_name}: frozen {key} mismatch")
    compare_fields(
        trainer_state,
        report,
        (
            "alpha",
            "base_model",
            "base_model_revision",
            "component_samples",
            "defect_types_sha256",
            "global_step",
            "manifest_sha256",
            "max_train_steps",
            "micro_step",
            "model_family",
            "object",
            "pipeline_version",
            "rank",
            "resolution",
            "run_signature",
            "seed",
            "selection_sha256",
            "source_images",
            "token_ids",
            "training_config_sha256",
            "type_counts",
        ),
        label=f"{object_name} trainer state",
    )
    compare_fields(
        upstream,
        report,
        (
            "component_samples",
            "peak_vram_gib",
            "source_images",
            "type_counts",
        ),
        label=f"{object_name} Colab validation",
    )
    require(upstream.get("steps") == max_steps, f"{object_name}: upstream step mismatch")
    require(
        upstream.get("reload_class") == "PeftModel",
        f"{object_name}: fresh PEFT reload is missing",
    )
    require(
        upstream.get("checkpoint_names") == [f"checkpoint-{step:06d}" for step in expected_steps],
        f"{object_name}: checkpoint inventory mismatch",
    )

    required_files = (
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
    for relative in required_files:
        path = final_dir / relative
        require(path.is_file() and path.stat().st_size > 0, f"{object_name}: missing {relative}")
    validate_adapter_configs(final_dir, config)
    load_json(final_dir / "tokenizer" / "tokenizer.json")
    load_json(final_dir / "tokenizer_2" / "tokenizer.json")

    adapter_hashes = {
        "unet_adapter_sha256": sha256_file(
            final_dir / "unet_adapter" / "adapter_model.safetensors"
        ),
        "text_token_adapter_sha256": sha256_file(
            final_dir / "text_token_adapter" / "adapter_model.safetensors"
        ),
        "text_token_adapter_2_sha256": sha256_file(
            final_dir / "text_token_adapter_2" / "adapter_model.safetensors"
        ),
    }
    require(
        upstream.get("adapter_hashes") == adapter_hashes,
        f"{object_name}: upstream adapter hashes mismatch",
    )
    for key, digest in adapter_hashes.items():
        require(report.get(key) == digest, f"{object_name}: report {key} mismatch")

    upstream_samples = {
        str(item["file"]): item for item in upstream.get("samples", []) if isinstance(item, dict)
    }
    sample_evidence: list[dict[str, Any]] = []
    backgrounds: set[str] = set()
    for index, step in enumerate(expected_steps):
        relative = f"samples/step_{step:06d}.png"
        png_path = object_root / relative
        sidecar = load_json(png_path.with_suffix(".json"))
        require(png_path.is_file(), f"{object_name}: missing {relative}")
        with Image.open(png_path) as image:
            image.verify()
            size = image.size
        resolution = int(config["model"]["resolution"])
        require(size == (resolution * 3, resolution), f"{object_name}: panel size mismatch")
        panel_sha256 = sha256_file(png_path)
        expected_token = f"<{object_name}-type{index % 2}>"
        require(sidecar.get("step") == step, f"{object_name}: sidecar step mismatch")
        require(
            sidecar.get("trigger_token") == expected_token,
            f"{object_name}: trigger-token rotation mismatch",
        )
        require(expected_token in str(sidecar.get("prompt")), f"{object_name}: prompt mismatch")
        require(
            sidecar.get("model_id") == config["model"]["id"]
            and sidecar.get("model_revision") == config["model"]["revision"],
            f"{object_name}: sample model lock mismatch",
        )
        require(
            sidecar.get("panel_sha256") == panel_sha256,
            f"{object_name}: panel hash mismatch",
        )
        background = paths.visa_raw / str(sidecar["background_image"])
        background_sha256 = sha256_file(background)
        require(
            sidecar.get("background_sha256") == background_sha256,
            f"{object_name}: background hash mismatch",
        )
        require(background_sha256 not in blocked, f"{object_name}: test blocklist hit")
        placement_mask = paths.synthetic / "placements" / object_name / str(sidecar["mask_path"])
        require(placement_mask.is_file(), f"{object_name}: placement mask is missing")
        evidence = {
            "file": relative,
            "panel_sha256": panel_sha256,
            "placement_id": sidecar["placement_id"],
            "trigger_token": expected_token,
        }
        require(
            upstream_samples.get(relative) == evidence,
            f"{object_name}: upstream sample evidence mismatch",
        )
        sample_evidence.append(evidence)
        backgrounds.add(background_sha256)

    require(
        report.get("sample_files") == [item["file"] for item in sample_evidence],
        f"{object_name}: report sample inventory mismatch",
    )
    return {
        "adapter_hashes": adapter_hashes,
        "backgrounds_checked": len(backgrounds),
        "checkpoint_names_verified_by_colab": upstream["checkpoint_names"],
        "component_samples": report["component_samples"],
        "peak_vram_gib": report["peak_vram_gib"],
        "reload_class_verified_by_colab": upstream["reload_class"],
        "samples": sample_evidence,
        "source_images": report["source_images"],
        "steps": report["global_step"],
        "training_elapsed_seconds": report["training_elapsed_seconds"],
        "wall_clock_seconds": report["wall_clock_seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/lora_sdxl.yaml"))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/colab/lora_sdxl"),
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/lora_sdxl_import_validation.json"),
    )
    args = parser.parse_args()

    paths = load_paths(args.paths)
    config = load_config(args.config)
    require(config["model"]["family"] == "sdxl", "Expected an SDXL training config")
    results_root = args.results_root.resolve(strict=True)
    validation_path = results_root / "lora_sdxl_validation.json"
    upstream = load_json(validation_path)
    frozen = frozen_hashes(paths)
    require(upstream.get("status") == "passed", "Colab validation did not pass")
    require(upstream.get("base_model") == config["model"]["id"], "Base model mismatch")
    require(
        upstream.get("base_model_revision") == config["model"]["revision"],
        "Base model revision mismatch",
    )
    require(upstream.get("pipeline_version") == "0.3.0", "Pipeline version mismatch")
    require(upstream.get("frozen_sha256") == frozen, "Frozen checksum set mismatch")
    require(set(upstream.get("objects", {})) == set(paths.objects), "Object set mismatch")
    blocklist = load_json(paths.splits / "test_blocklist.json")
    blocked = set(blocklist.get("sha256", []))
    require(
        len(blocked) == int(blocklist.get("unique_sha256_count", -1)) == 803,
        "Test blocklist cardinality mismatch",
    )
    config_sha256 = sha256_file(args.config)
    objects = {
        object_name: validate_object(
            object_name=object_name,
            results_root=results_root,
            paths=paths,
            config=config,
            config_sha256=config_sha256,
            frozen=frozen,
            upstream=upstream["objects"][object_name],
            blocked=blocked,
        )
        for object_name in paths.objects
    }
    result = {
        "archive": validate_archive(args.archive) if args.archive is not None else None,
        "base_model": config["model"]["id"],
        "base_model_revision": config["model"]["revision"],
        "colab_validation_sha256": sha256_file(validation_path),
        "frozen_sha256": frozen,
        "objects": objects,
        "pipeline_version": "0.3.0",
        "results_files": sum(1 for path in results_root.rglob("*") if path.is_file()),
        "results_root": results_root.relative_to(paths.project_root).as_posix(),
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
