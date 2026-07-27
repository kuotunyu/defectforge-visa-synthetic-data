"""Validate SD2 or SDXL LoRA run bundles, checkpoints, samples, and frozen inputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import (
    assert_not_blocklisted,
    read_checksum_file,
    sha256_file,
)
from src.common.paths import load_paths
from src.training.train_inpaint_lora import load_config, validate_saved_bundle


class LoraRunValidationError(RuntimeError):
    """Raised when a formal M10 artifact violates its contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument("--config", default="configs/lora_sd2.yaml", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LoraRunValidationError(f"Expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LoraRunValidationError(message)


def expected_checkpoint_names(max_steps: int, checkpoint_every: int) -> list[str]:
    steps = list(range(checkpoint_every, max_steps + 1, checkpoint_every))
    if not steps or steps[-1] != max_steps:
        steps.append(max_steps)
    return [f"checkpoint-{step:06d}" for step in steps]


def validate_adapter_configs(final_dir: Path, config: dict[str, Any]) -> None:
    unet = load_json(final_dir / "unet_adapter" / "adapter_config.json")
    text = load_json(final_dir / "text_token_adapter" / "adapter_config.json")
    require(unet["peft_type"] == "LORA", "UNet adapter is not PEFT LoRA")
    require(unet["r"] == int(config["training"]["rank"]), "LoRA rank mismatch")
    require(
        unet["lora_alpha"] == int(config["training"]["alpha"]),
        "LoRA alpha mismatch",
    )
    require(
        set(unet["target_modules"]) == {"to_k", "to_q", "to_v", "to_out.0"},
        "LoRA target modules mismatch",
    )
    require(
        text["peft_type"] == "TRAINABLE_TOKENS",
        "Text adapter is not PEFT TrainableTokens",
    )
    require(len(text["token_indices"]) == 2, "Expected exactly two trainable tokens")
    # Treat pre-family validation fixtures as the original SD2 contract.
    if config.get("model", {}).get("family", "sd2") == "sdxl":
        text_2 = load_json(final_dir / "text_token_adapter_2" / "adapter_config.json")
        require(
            text_2["peft_type"] == "TRAINABLE_TOKENS",
            "Second text adapter is not PEFT TrainableTokens",
        )
        require(
            len(text_2["token_indices"]) == 2,
            "Expected exactly two trainable tokens in text encoder 2",
        )


def validate_object(
    *,
    object_name: str,
    run_root: Path,
    paths: Any,
    config: dict[str, Any],
    config_sha256: str,
    frozen_sha256: dict[str, str],
    reload_adapters: bool,
) -> dict[str, Any]:
    seed = paths.seed
    object_root = run_root / object_name / f"seed_{seed}"
    report = load_json(object_root / "training_report.json")
    max_steps = int(config["training"]["max_train_steps"])
    checkpoint_every = int(config["training"]["checkpoint_every"])
    require(report["status"] == "passed", f"{object_name}: run did not pass")
    require(report["object"] == object_name, f"{object_name}: report object mismatch")
    require(report["seed"] == seed, f"{object_name}: seed mismatch")
    require(report["global_step"] == max_steps, f"{object_name}: step mismatch")
    require(report["micro_step"] == max_steps, f"{object_name}: micro-step mismatch")
    expected_pipeline = {"sd2": "0.2.0", "sdxl": "0.3.0"}[
        str(config["model"]["family"])
    ]
    require(
        report["pipeline_version"] == expected_pipeline,
        f"{object_name}: pipeline mismatch",
    )
    require(
        report["training_config_sha256"] == config_sha256,
        f"{object_name}: training config changed",
    )
    for key, expected in frozen_sha256.items():
        require(report[key] == expected, f"{object_name}: frozen {key} mismatch")
    require(
        report["base_model"] == config["model"]["id"]
        and report["base_model_revision"] == config["model"]["revision"],
        f"{object_name}: base model lock mismatch",
    )

    checkpoint_names = sorted(path.name for path in object_root.glob("checkpoint-*"))
    require(
        checkpoint_names == expected_checkpoint_names(max_steps, checkpoint_every),
        f"{object_name}: checkpoint set mismatch",
    )
    required_bundle_files = [
        "trainer_state.json",
        "training_state.pt",
        "unet_adapter/adapter_config.json",
        "unet_adapter/adapter_model.safetensors",
        "text_token_adapter/adapter_config.json",
        "text_token_adapter/adapter_model.safetensors",
        "tokenizer/tokenizer.json",
    ]
    if config["model"]["family"] == "sdxl":
        required_bundle_files.extend(
            [
                "text_token_adapter_2/adapter_config.json",
                "text_token_adapter_2/adapter_model.safetensors",
                "tokenizer_2/tokenizer.json",
            ]
        )
    for bundle_name in [*checkpoint_names, "final"]:
        bundle = object_root / bundle_name
        for relative in required_bundle_files:
            require((bundle / relative).is_file(), f"Missing {bundle_name}/{relative}")

    final_dir = object_root / "final"
    validate_adapter_configs(final_dir, config)
    adapter_hashes = {
        "unet_adapter_sha256": sha256_file(
            final_dir / "unet_adapter" / "adapter_model.safetensors"
        ),
        "text_token_adapter_sha256": sha256_file(
            final_dir / "text_token_adapter" / "adapter_model.safetensors"
        ),
    }
    if config["model"]["family"] == "sdxl":
        adapter_hashes["text_token_adapter_2_sha256"] = sha256_file(
            final_dir / "text_token_adapter_2" / "adapter_model.safetensors"
        )
    for key, actual in adapter_hashes.items():
        require(report[key] == actual, f"{object_name}: {key} mismatch")

    sample_dir = object_root / "samples"
    png_paths = sorted(sample_dir.glob("step_*.png"))
    sidecar_paths = sorted(sample_dir.glob("step_*.json"))
    expected_samples = max_steps // int(config["training"]["sample_every"])
    require(
        len(png_paths) == expected_samples == len(sidecar_paths),
        f"{object_name}: sample count mismatch",
    )
    trigger_tokens = [f"<{object_name}-type0>", f"<{object_name}-type1>"]
    expected_tokens = [
        trigger_tokens[index % len(trigger_tokens)] for index in range(expected_samples)
    ]
    backgrounds: list[Path] = []
    sample_evidence: list[dict[str, Any]] = []
    for index, sidecar_path in enumerate(sidecar_paths):
        sidecar = load_json(sidecar_path)
        png_path = sidecar_path.with_suffix(".png")
        require(png_path in png_paths, f"{object_name}: sample PNG missing")
        require(
            sidecar["trigger_token"] == expected_tokens[index],
            f"{object_name}: trigger-token rotation mismatch",
        )
        require(
            sidecar["panel_sha256"] == sha256_file(png_path),
            f"{object_name}: sample panel hash mismatch",
        )
        background = paths.visa_raw / sidecar["background_image"]
        require(
            sidecar["background_sha256"] == sha256_file(background),
            f"{object_name}: sample background hash mismatch",
        )
        backgrounds.append(background)
        sample_evidence.append(
            {
                "file": png_path.relative_to(object_root).as_posix(),
                "panel_sha256": sidecar["panel_sha256"],
                "placement_id": sidecar["placement_id"],
                "trigger_token": sidecar["trigger_token"],
            }
        )
    assert_not_blocklisted(backgrounds, paths.splits / "test_blocklist.json")

    reload_evidence: dict[str, Any] | None = None
    if reload_adapters:
        reload_evidence = validate_saved_bundle(
            final_dir,
            model_id=str(config["model"]["id"]),
            revision=str(config["model"]["revision"]),
            family=str(config["model"]["family"]),
        )
        for key, expected in adapter_hashes.items():
            require(
                reload_evidence[key] == expected,
                f"{object_name}: reloaded {key} mismatch",
            )

    return {
        "adapter_hashes": adapter_hashes,
        "checkpoint_names": checkpoint_names,
        "component_samples": report["component_samples"],
        "peak_vram_gib": report["peak_vram_gib"],
        "reload_class": (
            reload_evidence["reload_class"] if reload_evidence else report["reload_class"]
        ),
        "samples": sample_evidence,
        "source_images": report["source_images"],
        "steps": report["global_step"],
        "training_elapsed_seconds": report["training_elapsed_seconds"],
        "type_counts": report["type_counts"],
    }


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths)
    config = load_config(args.config)
    run_root = args.run_root or paths.runs / str(config["output"]["name"])
    manifest_sha256, _ = read_checksum_file(paths.splits / "MANIFEST.sha256")
    selection_sha256, _ = read_checksum_file(paths.splits / "FEWSHOT_SELECTION.sha256")
    defect_types_sha256, _ = read_checksum_file(paths.splits / "DEFECT_TYPES.sha256")
    frozen_sha256 = {
        "manifest_sha256": manifest_sha256,
        "selection_sha256": selection_sha256,
        "defect_types_sha256": defect_types_sha256,
    }
    result = {
        "base_model": config["model"]["id"],
        "base_model_revision": config["model"]["revision"],
        "frozen_sha256": frozen_sha256,
        "objects": {
            object_name: validate_object(
                object_name=object_name,
                run_root=run_root,
                paths=paths,
                config=config,
                config_sha256=sha256_file(args.config),
                frozen_sha256=frozen_sha256,
                reload_adapters=args.reload,
            )
            for object_name in paths.objects
        },
        "pipeline_version": {"sd2": "0.2.0", "sdxl": "0.3.0"}[
            str(config["model"]["family"])
        ],
        "run_root": str(run_root),
        "status": "passed",
        "validated_at": datetime.now(UTC).isoformat(),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
