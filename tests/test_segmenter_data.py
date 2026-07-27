from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.common.paths import load_paths
from src.training import segmenter_data


def test_resolve_group_maps_all_mixed_to_filtered() -> None:
    canonical, group = segmenter_data._resolve_group(
        {
            "filtered_syn": {"real_bad": "seed10"},
            "all_mixed": {"alias_of": "filtered_syn"},
        },
        "all_mixed",
    )
    assert canonical == "filtered_syn"
    assert group["real_bad"] == "seed10"


def test_resolve_group_rejects_alias_cycle() -> None:
    with pytest.raises(segmenter_data.SegmentationDataError, match="cycle"):
        segmenter_data._resolve_group(
            {"a": {"alias_of": "b"}, "b": {"alias_of": "a"}},
            "a",
        )


def test_resolve_mask_path_supports_real_and_synthetic(tmp_path: Path) -> None:
    data = tmp_path / "data"
    paths = replace(
        load_paths(),
        data_root=data,
        visa_raw=data / "raw" / "VisA",
        synthetic=data / "synthetic",
    )
    real_mask = data / "raw" / "VisA" / "pcb1" / "mask.png"
    synthetic_mask = data / "synthetic" / "filtered" / "masks" / "sample.png"
    real_mask.parent.mkdir(parents=True)
    synthetic_mask.parent.mkdir(parents=True)
    real_mask.write_bytes(b"real")
    synthetic_mask.write_bytes(b"synthetic")

    base = {
        "sample_id": "sample",
        "object_name": "pcb1",
        "kind": "real",
        "source_name": "source",
        "image_path": "image.png",
        "image_sha256": "0" * 64,
        "mask_sha256": "1" * 64,
        "has_defect": True,
        "manifest_refs": (),
        "defect_type": None,
    }
    real = segmenter_data.SegmentationSample(
        root="visa_raw",
        mask_path="pcb1/mask.png",
        **base,
    )
    synthetic = segmenter_data.SegmentationSample(
        root="synthetic/filtered",
        mask_path="masks/sample.png",
        **{**base, "kind": "synthetic"},
    )
    assert segmenter_data.resolve_mask_path(paths, real) == real_mask.resolve()
    assert segmenter_data.resolve_mask_path(paths, synthetic) == synthetic_mask.resolve()


def test_group_payload_sha_is_stable() -> None:
    sample = segmenter_data.SegmentationSample(
        sample_id="real::x",
        object_name="pcb1",
        kind="real",
        source_name="real_train",
        root="visa_raw",
        image_path="x.png",
        image_sha256="a" * 64,
        mask_path=None,
        mask_sha256=None,
        has_defect=False,
        manifest_refs=("x.png",),
    )
    group = segmenter_data.SegmentationGroup(
        requested_group="real_only",
        canonical_group="real_only",
        object_name="pcb1",
        mode="development",
        standard_augmentation=False,
        train=(sample,),
        validation=(sample,),
        test=(),
        manifest_sha256="b" * 64,
        selection_sha256="c" * 64,
    )
    first = segmenter_data.group_payload_sha256(group)
    second = segmenter_data.group_payload_sha256(group)
    assert first == second
    assert len(first) == 64
    assert group.payload()["split_manifest_sha256"] == "b" * 64
