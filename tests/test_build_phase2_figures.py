from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from scripts.build_phase2_figures import (
    MAIN_GROUPS,
    OBJECTS,
    Phase2FigureError,
    classification_value,
    equivalent_real_count,
    main,
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


def test_main_rejects_bad_segmentation_before_writing_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classification = tmp_path / "classification.csv"
    fields = (
        "requested_group",
        "canonical_group",
        "object",
        "seed",
        "macro_f1",
        "auroc",
    )
    groups = (*MAIN_GROUPS, "real_20", "syn_125", "syn_250")
    rows = [
        {
            "requested_group": group_name,
            "canonical_group": group_name,
            "object": object_name,
            "seed": str(seed),
            "macro_f1": "0.5",
            "auroc": "0.6",
        }
        for group_name in groups
        for object_name in OBJECTS
        for seed in (42,)
    ]
    while len(rows) < 38:
        index = len(rows)
        rows.append(
            {
                "requested_group": f"extra_{index}",
                "canonical_group": f"extra_{index}",
                "object": OBJECTS[index % 2],
                "seed": "42",
                "macro_f1": "0.5",
                "auroc": "0.6",
            }
        )
    with classification.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    segmentation = tmp_path / "bad_segmentation.csv"
    segmentation.write_text("wrong,value\n1,2\n", encoding="utf-8")
    output_dir = tmp_path / "figures"
    validation = tmp_path / "validation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_phase2_figures.py",
            "--classification",
            str(classification),
            "--segmentation",
            str(segmentation),
            "--output-dir",
            str(output_dir),
            "--validation-out",
            str(validation),
        ],
    )
    with pytest.raises(Phase2FigureError, match="columns"):
        main()
    assert not output_dir.exists()
    assert not validation.exists()
