from __future__ import annotations

import pytest

from scripts.build_phase2_figures import (
    Phase2FigureError,
    classification_value,
    equivalent_real_count,
    segmentation_value,
)


def test_equivalent_real_count_interpolates_monotone_fit() -> None:
    result = equivalent_real_count([10, 20, 60], [0.4, 0.6, 0.8], 0.7)
    assert result["relation"] == "interpolated"
    assert result["estimate"] == pytest.approx(40.0)


def test_equivalent_real_count_reports_outside_range() -> None:
    low = equivalent_real_count([10, 20, 60], [0.4, 0.6, 0.8], 0.2)
    high = equivalent_real_count([10, 20, 60], [0.4, 0.6, 0.8], 0.9)
    assert (low["relation"], low["estimate"]) == ("at_or_below", 10.0)
    assert (high["relation"], high["estimate"]) == ("at_or_above", 60.0)


def test_classification_value_requires_unique_seed42_row() -> None:
    rows = [
        {
            "object": "pcb1",
            "canonical_group": "real_only",
            "seed": "42",
            "macro_f1": "0.5",
        }
    ]
    assert classification_value(
        rows,
        object_name="pcb1",
        group_name="real_only",
        metric="macro_f1",
    ) == pytest.approx(0.5)
    with pytest.raises(Phase2FigureError, match="expected 1 row, found 2"):
        classification_value(
            [*rows, dict(rows[0])],
            object_name="pcb1",
            group_name="real_only",
            metric="macro_f1",
        )


def test_segmentation_value_reads_logical_alias_row() -> None:
    rows = [{"object": "capsules", "logical_group": "all_mixed", "dice": "0.75"}]
    assert segmentation_value(
        rows,
        object_name="capsules",
        group_name="all_mixed",
        metric="dice",
    ) == pytest.approx(0.75)
