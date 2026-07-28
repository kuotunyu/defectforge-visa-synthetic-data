from pathlib import Path

import pytest

from src.training.classifier_data import ClassificationSample
from src.training.train_classifier import (
    ClassifierTrainingError,
    _exposure_counts,
    build_sampler,
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


def test_domain_balanced_sampler_preserves_real_anomaly_exposure() -> None:
    samples = [
        *[_sample(f"good_{index}", label=0, kind="real") for index in range(100)],
        *[_sample(f"real_bad_{index}", label=1, kind="real") for index in range(10)],
        *[_sample(f"synthetic_bad_{index}", label=1, kind="synthetic") for index in range(500)],
    ]
    sampler = build_sampler(
        samples,
        strategy="domain_balanced",
        num_samples=10_000,
        seed=42,
        real_bad_share=0.75,
    )
    exposure = _exposure_counts(samples, list(sampler))
    assert exposure["real_good"] / 10_000 == pytest.approx(0.50, abs=0.02)
    assert exposure["real_bad"] / 10_000 == pytest.approx(0.375, abs=0.02)
    assert exposure["synthetic_bad"] / 10_000 == pytest.approx(0.125, abs=0.02)


def test_domain_balanced_sampler_is_deterministic() -> None:
    samples = [
        _sample("good", label=0, kind="real"),
        _sample("real_bad", label=1, kind="real"),
        _sample("synthetic_bad", label=1, kind="synthetic"),
    ]
    kwargs = {
        "strategy": "domain_balanced",
        "num_samples": 100,
        "seed": 42,
        "real_bad_share": 0.5,
    }
    assert list(build_sampler(samples, **kwargs)) == list(build_sampler(samples, **kwargs))


def test_class_balanced_sampler_rejects_misleading_domain_share() -> None:
    samples = [
        _sample("good", label=0, kind="real"),
        _sample("real_bad", label=1, kind="real"),
    ]
    with pytest.raises(ClassifierTrainingError, match="only applies"):
        build_sampler(
            samples,
            strategy="class_balanced",
            num_samples=10,
            seed=42,
            real_bad_share=0.75,
        )
