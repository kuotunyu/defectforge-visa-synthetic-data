import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.common.paths import load_paths
from src.training.classifier_data import (
    ClassificationDataError,
    ClassificationSample,
    _load_synthetic_records,
    _resolve_group,
    stratified_select,
    summarize_samples,
)


def _synthetic(
    sample_id: str,
    source_name: str,
    defect_type: str,
) -> tuple[dict[str, object], Path, str]:
    return (
        {
            "sample_id": sample_id,
            "defect_type": defect_type,
        },
        Path(f"{sample_id}.png"),
        source_name,
    )


def test_resolve_group_follows_alias_without_changing_request() -> None:
    groups = {
        "full_real": {"real_bad": "full"},
        "real_60": {"alias_of": "full_real"},
    }
    canonical, config = _resolve_group(groups, "real_60")
    assert canonical == "full_real"
    assert config["real_bad"] == "full"


def test_resolve_group_rejects_alias_cycle() -> None:
    groups = {"a": {"alias_of": "b"}, "b": {"alias_of": "a"}}
    with pytest.raises(ClassificationDataError, match="alias cycle"):
        _resolve_group(groups, "a")


def test_stratified_select_is_exact_and_deterministic() -> None:
    records = [
        _synthetic(f"a{i}", "source_a", "type0")
        for i in range(6)
    ] + [
        _synthetic(f"b{i}", "source_b", "type1")
        for i in range(4)
    ]
    first = stratified_select(
        records,
        count=5,
        seed=42,
        group_name="filtered_syn",
        object_name="pcb1",
    )
    second = stratified_select(
        records,
        count=5,
        seed=42,
        group_name="filtered_syn",
        object_name="pcb1",
    )
    assert [item[0]["sample_id"] for item in first] == [
        item[0]["sample_id"] for item in second
    ]
    assert sum(item[2] == "source_a" for item in first) == 3
    assert sum(item[2] == "source_b" for item in first) == 2


def test_stratified_select_rejects_short_source() -> None:
    with pytest.raises(ClassificationDataError, match=r"has 1 < 2"):
        stratified_select(
            [_synthetic("only", "source", "type0")],
            count=2,
            seed=42,
            group_name="filtered_syn",
            object_name="pcb1",
        )


def test_load_synthetic_records_filters_packaged_ids_before_path_resolution(
    tmp_path: Path,
) -> None:
    view = tmp_path / "synthetic" / "m18_pool"
    images = view / "images"
    images.mkdir(parents=True)
    (images / "keep.png").write_bytes(b"kept")
    records = [
        {
            "sample_id": "keep",
            "object": "pcb1",
            "generator": "stageA_procedural",
            "bucket": None,
            "defect_type": "type0",
            "image_path": "images/keep.png",
        },
        {
            "sample_id": "not-packaged",
            "object": "pcb1",
            "generator": "stageA_procedural",
            "bucket": None,
            "defect_type": "type0",
            "image_path": "images/missing.png",
        },
    ]
    (view / "metadata.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    paths = replace(load_paths(), synthetic=tmp_path / "synthetic")
    selected = _load_synthetic_records(
        paths,
        view="m18_pool",
        object_name="pcb1",
        inputs=None,
        sample_ids={"keep"},
    )
    assert [record[0]["sample_id"] for record in selected] == ["keep"]


def test_summarize_samples_reports_real_and_synthetic_counts() -> None:
    samples = [
        ClassificationSample(
            sample_id="good",
            object_name="pcb1",
            label=0,
            kind="real",
            source_name="real_train",
            root="visa_raw",
            relative_path="good.png",
            sha256="a" * 64,
            manifest_refs=("good.png",),
        ),
        ClassificationSample(
            sample_id="bad",
            object_name="pcb1",
            label=1,
            kind="synthetic",
            source_name="stageB_sd2/searched",
            root="synthetic/filtered",
            relative_path="bad.png",
            sha256="b" * 64,
            manifest_refs=("background.png",),
            defect_type="type0",
        ),
    ]
    assert summarize_samples(samples) == {
        "total": 2,
        "labels": {"bad": 1, "good": 1},
        "kinds": {"real": 1, "synthetic": 1},
        "sources": {"real_train": 1, "stageB_sd2/searched": 1},
    }
