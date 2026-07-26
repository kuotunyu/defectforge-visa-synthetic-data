import json
from pathlib import Path

import pytest

from scripts.validate_lora_run import (
    LoraRunValidationError,
    expected_checkpoint_names,
    load_json,
    validate_adapter_configs,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_expected_checkpoint_names_includes_final_partial_interval() -> None:
    assert expected_checkpoint_names(400, 100) == [
        "checkpoint-000100",
        "checkpoint-000200",
        "checkpoint-000300",
        "checkpoint-000400",
    ]
    assert expected_checkpoint_names(250, 100) == [
        "checkpoint-000100",
        "checkpoint-000200",
        "checkpoint-000250",
    ]


def test_load_json_rejects_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    write_json(path, [])

    with pytest.raises(LoraRunValidationError, match="Expected JSON object"):
        load_json(path)


def test_validate_adapter_configs_enforces_adapter_contract(tmp_path: Path) -> None:
    final_dir = tmp_path / "final"
    write_json(
        final_dir / "unet_adapter" / "adapter_config.json",
        {
            "peft_type": "LORA",
            "r": 4,
            "lora_alpha": 4,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        },
    )
    write_json(
        final_dir / "text_token_adapter" / "adapter_config.json",
        {
            "peft_type": "TRAINABLE_TOKENS",
            "token_indices": [49408, 49409],
        },
    )
    config = {"training": {"rank": 4, "alpha": 4}}

    validate_adapter_configs(final_dir, config)

    config["training"]["rank"] = 8
    with pytest.raises(LoraRunValidationError, match="LoRA rank mismatch"):
        validate_adapter_configs(final_dir, config)
