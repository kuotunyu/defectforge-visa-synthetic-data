import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.common.integrity import sha256_file
from src.evaluation.quality_data import (
    QualityDataError,
    audit_sources_against_blocklist,
    crop_cache_key,
    load_defect_types,
)


def test_crop_cache_key_changes_for_every_provenance_axis() -> None:
    base = {
        "metadata_sha256": "a" * 64,
        "defect_types_sha256": "b" * 64,
        "source_audit_sha256": "c" * 64,
        "ratio": 2.5,
    }
    observed = crop_cache_key(**base)
    assert len(observed) == 24
    for name, replacement in {
        "metadata_sha256": "d" * 64,
        "defect_types_sha256": "e" * 64,
        "source_audit_sha256": "f" * 64,
        "ratio": 3.5,
    }.items():
        changed = dict(base)
        changed[name] = replacement
        assert crop_cache_key(**changed) != observed


def test_load_defect_types_rejects_non_object_payload(tmp_path: Path) -> None:
    path = tmp_path / "types.json"
    path.write_text(json.dumps({"objects": []}), encoding="utf-8")
    with pytest.raises(QualityDataError, match="Invalid defect"):
        load_defect_types(path)


def test_source_audit_fails_closed_on_real_component_blocklist_hit(
    tmp_path: Path,
) -> None:
    visa_root = tmp_path / "visa"
    image = visa_root / "pcb1" / "image.png"
    mask = visa_root / "pcb1" / "mask.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    defect_types = {
        "objects": {
            "pcb1": {
                "types": [
                    {
                        "components": [
                            {
                                "image_path": "pcb1/image.png",
                                "mask_path": "pcb1/mask.png",
                            }
                        ]
                    }
                ]
            }
        }
    }
    blocklist = tmp_path / "blocklist.json"
    blocklist.write_text(
        json.dumps({"sha256": [sha256_file(image)]}),
        encoding="utf-8",
    )
    with pytest.raises(QualityDataError, match="hit test blocklist"):
        audit_sources_against_blocklist(
            SimpleNamespace(visa_raw=visa_root),
            [],
            defect_types,
            blocklist,
        )
