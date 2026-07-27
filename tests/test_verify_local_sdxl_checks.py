import pytest

from scripts.verify_local_sdxl_checks import (
    LocalSdxlValidationError,
    validate_resume_progression,
)


def _state(step: int, *, signature: str = "locked") -> dict[str, object]:
    state: dict[str, object] = {
        "created_at": f"2026-07-27T00:00:0{step}+00:00",
        "global_step": step,
        "max_train_steps": 2,
        "micro_step": step,
        "run_signature": signature,
        "training_elapsed_seconds": float(step),
    }
    for field in (
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
    ):
        state[field] = field
    return state


def test_validate_resume_progression_accepts_exact_step_transition() -> None:
    validate_resume_progression(_state(1), _state(2))


def test_validate_resume_progression_rejects_signature_change() -> None:
    with pytest.raises(LocalSdxlValidationError, match="run_signature mismatch"):
        validate_resume_progression(_state(1), _state(2, signature="changed"))
