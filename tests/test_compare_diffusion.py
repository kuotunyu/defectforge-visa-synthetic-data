import pytest

from scripts.compare_diffusion import select_records, union_crop
from src.synthetic.generate_diffusion import DiffusionGenerationError


def record(sample_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "defect_type": "type1" if "__type1__" in sample_id else "type0",
    }


def test_union_crop_uses_same_bounded_context_for_both_buckets() -> None:
    assert union_crop(
        [10, 20, 30, 40],
        [5, 25, 50, 60],
        width=48,
        height=72,
    ) == (5, 20, 48, 72)


def test_select_records_aligns_samples_in_stable_order() -> None:
    original = {
        "pcb1__type1__0001__00": record("pcb1__type1__0001__00"),
        "pcb1__type0__0000__00": record("pcb1__type0__0000__00"),
    }
    searched = {
        "pcb1__type0__0000__00": record("pcb1__type0__0000__00"),
        "pcb1__type1__0001__00": record("pcb1__type1__0001__00"),
    }

    pairs = select_records(original, searched, maximum=2)

    assert [pair[0]["sample_id"] for pair in pairs] == [
        "pcb1__type0__0000__00",
        "pcb1__type1__0001__00",
    ]


def test_select_records_rejects_bucket_inventory_mismatch() -> None:
    with pytest.raises(DiffusionGenerationError, match="Bucket sample IDs differ"):
        select_records(
            {"pcb1__type0__0000__00": record("pcb1__type0__0000__00")},
            {"pcb1__type0__0001__00": record("pcb1__type0__0001__00")},
            maximum=1,
        )
