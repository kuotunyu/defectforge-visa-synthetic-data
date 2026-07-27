from pathlib import Path

import pytest

from src.training.classifier_data import ClassificationSample
from src.training.train_classifier import (
    ClassifierTrainingError,
    _exposure_counts,
    classification_metrics,
    cosine_multiplier,
)


def _sample(name: str, *, label: int, kind: str) -> ClassificationSample:
    return ClassificationSample(
        sample_id=name,
        object_name="pcb1",
        label=label,
        kind=kind,
        source_name=kind,
        root="visa_raw" if kind == "real" else "synthetic/filtered",
        relative_path=str(Path(f"{name}.png")),
        sha256=name * 64,
        manifest_refs=("source.png",),
    )


def test_classification_metrics_reports_operational_false_positive_rate() -> None:
    metrics = classification_metrics(
        [0, 0, 1, 1],
        [0.1, 0.9, 0.8, 0.2],
    )
    assert metrics["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["macro_f1"] == pytest.approx(0.5)
    assert metrics["anomaly_f1"] == pytest.approx(0.5)
    assert metrics["auroc"] == pytest.approx(0.5)
    assert metrics["normal_false_positive_rate"] == pytest.approx(0.5)


def test_classification_metrics_requires_both_classes() -> None:
    with pytest.raises(ClassifierTrainingError, match="both classes"):
        classification_metrics([0, 0], [0.1, 0.2])


def test_cosine_multiplier_warms_up_then_decays() -> None:
    assert cosine_multiplier(0, total_steps=10, warmup_steps=2) == pytest.approx(0.5)
    assert cosine_multiplier(1, total_steps=10, warmup_steps=2) == pytest.approx(1.0)
    assert cosine_multiplier(10, total_steps=10, warmup_steps=2) == pytest.approx(0.0)


def test_exposure_counts_separates_real_and_synthetic_bad() -> None:
    samples = [
        _sample("good", label=0, kind="real"),
        _sample("real_bad", label=1, kind="real"),
        _sample("synthetic_bad", label=1, kind="synthetic"),
    ]
    assert _exposure_counts(samples, [0, 1, 2, 2]) == {
        "real_good": 1,
        "real_bad": 1,
        "synthetic_bad": 2,
    }
