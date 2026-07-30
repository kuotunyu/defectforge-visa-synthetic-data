from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.diagnose_placement_geometry import (
    RING_WIDTH_PX,
    PlacementDiagnosisError,
    assert_sources_are_not_test,
    context_ring,
    derive_findings,
    load_placements,
    quantile,
)


def _square_mask(size: int = 200, side: int = 40) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    start = size // 2 - side // 2
    mask[start : start + side, start : start + side] = 255
    return mask


def test_context_ring_never_includes_the_mask_itself() -> None:
    # The whole point: real masks already contain the defect, so the band must sit outside.
    mask = _square_mask()
    ring = context_ring(mask)
    assert not (ring & (mask > 0)).any()
    assert ring.any()


def test_context_ring_hugs_the_mask_boundary() -> None:
    mask = _square_mask(side=40)
    ring = context_ring(mask, width_px=3)
    rows = np.where(ring.any(axis=1))[0]
    inner = np.where((mask > 0).any(axis=1))[0]
    # The band starts within the requested width of the mask edge, not somewhere arbitrary.
    assert 0 < inner[0] - rows[0] <= 3 + 1


def test_a_wider_ring_covers_more_pixels() -> None:
    mask = _square_mask()
    assert context_ring(mask, width_px=4).sum() < context_ring(mask, width_px=12).sum()


def test_an_empty_mask_yields_an_empty_ring() -> None:
    assert not context_ring(np.zeros((50, 50), dtype=np.uint8)).any()


def test_default_ring_width_is_used_when_unspecified() -> None:
    mask = _square_mask()
    assert context_ring(mask).sum() == context_ring(mask, width_px=RING_WIDTH_PX).sum()


def test_quantile_rejects_an_empty_series() -> None:
    with pytest.raises(PlacementDiagnosisError, match="empty series"):
        quantile([], 0.5)


def test_findings_refuse_to_generalise_when_one_object_is_fully_in_band() -> None:
    objects = {
        "pcb1": {
            "area": {
                "synthetic_in_real_band_fraction": 0.427,
                "synthetic_min_exceeds_real_median": True,
            }
        },
        "capsules": {
            "area": {
                "synthetic_in_real_band_fraction": 1.0,
                "synthetic_min_exceeds_real_median": False,
            }
        },
    }
    findings = derive_findings(objects)
    assert findings["objects_with_majority_out_of_band_area"] == ["pcb1"]
    assert findings["objects_whose_smallest_placement_exceeds_the_real_median"] == ["pcb1"]
    assert findings["objects_with_area_fully_in_band"] == ["capsules"]
    assert findings["area_explains_every_object"] is False


def test_findings_allow_a_shared_explanation_only_when_no_object_is_in_band() -> None:
    objects = {
        name: {
            "area": {
                "synthetic_in_real_band_fraction": 0.3,
                "synthetic_min_exceeds_real_median": True,
            }
        }
        for name in ("pcb1", "capsules")
    }
    findings = derive_findings(objects)
    assert findings["objects_with_area_fully_in_band"] == []
    assert findings["area_explains_every_object"] is True


def test_a_source_in_the_blocklist_fails_closed(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "obj").mkdir(parents=True)
    mask = raw / "obj" / "mask.png"
    mask.write_bytes(b"mask-bytes")
    image = raw / "obj" / "image.JPG"
    image.write_bytes(b"image-bytes")
    import hashlib

    digest = hashlib.sha256(b"mask-bytes").hexdigest()
    records = [{"source_mask": "obj/mask.png", "background_image": "obj/image.JPG"}]
    with pytest.raises(PlacementDiagnosisError, match="test blocklist"):
        assert_sources_are_not_test(records, raw_root=raw, blocklist={digest})


def test_clean_sources_report_zero_hits(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "obj").mkdir(parents=True)
    (raw / "obj" / "mask.png").write_bytes(b"mask-bytes")
    (raw / "obj" / "image.JPG").write_bytes(b"image-bytes")
    records = [{"source_mask": "obj/mask.png", "background_image": "obj/image.JPG"}]
    guard = assert_sources_are_not_test(records, raw_root=raw, blocklist=set())
    assert guard == {"checked_sources": 2, "blocklist_hits": 0}


def test_missing_placement_records_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(PlacementDiagnosisError, match="Missing placement records"):
        load_placements(tmp_path, "pcb1")
