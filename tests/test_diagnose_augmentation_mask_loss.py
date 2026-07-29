from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.diagnose_augmentation_mask_loss import (
    AugmentationDiagnosisError,
    compare_augmentation,
    defect_samples,
    loss_trajectory,
)

_DEFECT = {
    "sample_id": "capsules/train/000",
    "object_name": "capsules",
    "kind": "real",
    "source_name": "real_train",
    "root": "visa_raw",
    "image_path": "capsules/Data/Images/Anomaly/000.JPG",
    "image_sha256": "a" * 64,
    "mask_path": "capsules/Data/Masks/Anomaly/000.png",
    "mask_sha256": "b" * 64,
    "has_defect": True,
    "manifest_refs": [],
    "defect_type": None,
}
_NORMAL = {**_DEFECT, "sample_id": "capsules/train/001", "has_defect": False, "mask_path": None}
_SYNTHETIC = {**_DEFECT, "sample_id": "capsules/train/002", "kind": "synthetic"}


def _write_report(
    runs_root: Path,
    *,
    object_name: str,
    group: str,
    seed: int,
    dice_start: float,
    dice_end: float,
    steps: int = 100,
) -> None:
    run_dir = runs_root / object_name / "runs" / f"m18_{group}_{object_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for step in range(steps):
        # Linear ramp so the first and last windows are unambiguous.
        share = step / max(1, steps - 1)
        history.append(
            {
                "step": step + 1,
                "dice_loss": dice_start + (dice_end - dice_start) * share,
                "bce_loss": 0.5 - 0.3 * share,
            }
        )
    (run_dir / "training_report.json").write_text(
        json.dumps(
            {
                "run_name": run_dir.name,
                "canonical_group": group,
                "standard_augmentation": group == "std_aug",
                "metrics": {"dice": 0.0 if group == "std_aug" else 0.5},
                "loss_history": history,
            }
        ),
        encoding="utf-8",
    )


def test_only_real_defect_records_are_replayed() -> None:
    samples = defect_samples({"train": [_DEFECT, _NORMAL, _SYNTHETIC]})

    assert [sample.sample_id for sample in samples] == ["capsules/train/000"]


def test_a_manifest_without_real_defects_is_rejected() -> None:
    with pytest.raises(AugmentationDiagnosisError, match="No real defect images"):
        defect_samples({"train": [_NORMAL, _SYNTHETIC]})


def test_loss_trajectory_measures_improvement_between_windows(tmp_path: Path) -> None:
    _write_report(tmp_path, object_name="capsules", group="real_only", seed=42,
                  dice_start=0.99, dice_end=0.90)
    run_dir = tmp_path / "capsules" / "runs" / "m18_real_only_capsules_seed42"

    trajectory = loss_trajectory(run_dir, window=50)

    assert len(trajectory["windows"]) == 2
    assert trajectory["dice_loss_improvement"] == pytest.approx(0.045, abs=1e-3)
    assert trajectory["bce_loss_improvement"] > 0.0


def test_flat_dice_loss_under_augmentation_is_reported_as_not_engaged(tmp_path: Path) -> None:
    """capsules/std_aug: BCE still falls, but the Dice term never engages."""
    _write_report(tmp_path, object_name="capsules", group="real_only", seed=42,
                  dice_start=0.99, dice_end=0.90)
    _write_report(tmp_path, object_name="capsules", group="std_aug", seed=42,
                  dice_start=0.996, dice_end=0.995)

    comparison = compare_augmentation("capsules", runs_root=tmp_path, seed=42, window=50)

    assert comparison["dice_term_engaged_without_augmentation"] is True
    assert comparison["dice_term_engaged_with_augmentation"] is False
    # BCE improves in both, which is exactly why it cannot be the discriminator.
    assert comparison["trajectories"]["std_aug"]["bce_loss_improvement"] > 0.0


def test_augmentation_that_does_not_block_learning_is_reported_as_engaged(tmp_path: Path) -> None:
    _write_report(tmp_path, object_name="pcb1", group="real_only", seed=42,
                  dice_start=0.99, dice_end=0.90)
    _write_report(tmp_path, object_name="pcb1", group="std_aug", seed=42,
                  dice_start=0.99, dice_end=0.90)

    comparison = compare_augmentation("pcb1", runs_root=tmp_path, seed=42, window=50)

    assert comparison["dice_term_engaged_without_augmentation"] is True
    assert comparison["dice_term_engaged_with_augmentation"] is True
