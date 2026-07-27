from __future__ import annotations

import numpy as np
import pytest
import torch

from src.training.segmenter_data import SegmentationSample
from src.training.train_segmenter import (
    FixedDrawSampler,
    balanced_draw_indices,
    binary_dice_bce_loss,
    segmentation_metrics,
)


def _sample(name: str, has_defect: bool) -> SegmentationSample:
    return SegmentationSample(
        sample_id=name,
        object_name="pcb1",
        kind="synthetic" if has_defect else "real",
        source_name="test",
        root="synthetic/test" if has_defect else "visa_raw",
        image_path=f"{name}.png",
        image_sha256=name * 64,
        mask_path=f"{name}_mask.png" if has_defect else None,
        mask_sha256=name.upper() * 64 if has_defect else None,
        has_defect=has_defect,
        manifest_refs=(f"{name}.png",),
    )


def test_balanced_draw_indices_are_deterministic_and_cover_both_classes() -> None:
    samples = [_sample("a", False), _sample("b", False), _sample("c", True)]
    first = balanced_draw_indices(samples, num_samples=20, seed=42)
    second = balanced_draw_indices(samples, num_samples=20, seed=42)
    assert first == second
    assert {samples[index].has_defect for index in first} == {False, True}


def test_fixed_draw_sampler_preserves_global_draw_index_on_resume() -> None:
    assert list(FixedDrawSampler([2, 0, 1, 2], offset=2)) == [(1, 2), (2, 3)]


def test_dice_bce_prefers_correct_logits() -> None:
    masks = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    correct = torch.tensor([[[[-8.0, 8.0], [-8.0, 8.0]]]])
    wrong = -correct
    correct_loss, correct_parts = binary_dice_bce_loss(
        correct,
        masks,
        dice_weight=1.0,
        bce_weight=1.0,
        smooth=1.0,
    )
    wrong_loss, _ = binary_dice_bce_loss(
        wrong,
        masks,
        dice_weight=1.0,
        bce_weight=1.0,
        smooth=1.0,
    )
    assert correct_loss < wrong_loss
    assert correct_parts["dice_loss"] < 0.01


def test_segmentation_metrics_are_one_for_perfect_scores() -> None:
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, 2:5, 2:5] = True
    probabilities = np.where(masks, 0.99, 0.01).astype(np.float32)
    metrics = segmentation_metrics(
        probabilities,
        masks,
        threshold=0.5,
        au_pro_max_fpr=0.3,
        au_pro_thresholds=64,
    )
    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["miou"] == pytest.approx(1.0)
    assert metrics["pixel_auroc"] == pytest.approx(1.0)
    assert metrics["aupro"] == pytest.approx(1.0)
