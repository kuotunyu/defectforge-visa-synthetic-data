from __future__ import annotations

import numpy as np
from PIL import Image

from scripts.build_sample_grids import overlay_mask, select_records


def test_select_records_round_robins_pseudo_types() -> None:
    records = [
        {"sample_id": "b1", "defect_type": "type1"},
        {"sample_id": "a1", "defect_type": "type0"},
        {"sample_id": "a2", "defect_type": "type0"},
        {"sample_id": "b2", "defect_type": "type1"},
    ]
    selected = select_records(records, count=3)
    assert [row["sample_id"] for row in selected] == ["a1", "b1", "a2"]


def test_overlay_mask_changes_only_positive_pixels() -> None:
    image = Image.fromarray(np.full((3, 4, 3), 100, dtype=np.uint8))
    mask_array = np.zeros((3, 4), dtype=np.uint8)
    mask_array[1, 2] = 255
    mask = Image.fromarray(mask_array)
    rendered = overlay_mask(image, mask)
    assert np.array_equal(rendered[0, 0], [100, 100, 100])
    assert not np.array_equal(rendered[1, 2], [100, 100, 100])
    assert rendered[1, 2, 0] > rendered[1, 2, 1]
