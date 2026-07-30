from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts.aggregate_segmentation import FORMAL_GROUPS, SEEDS
from scripts.build_phase2_figures import OBJECTS
from scripts.verify_seed42_reproduction import (
    ReproductionCheckError,
    build_report,
    compare,
)

FIELDS = (
    "logical_group",
    "canonical_group",
    "physical_run",
    "object",
    "seed",
    "run_name",
    "run_signature",
    "dice",
    "miou",
    "pixel_auroc",
    "aupro",
    "model_sha256",
)


def _row(group: str, object_name: str, seed: int, *, dice: float, digest: str) -> dict[str, str]:
    physical = group != "all_mixed"
    canonical = "filtered_syn" if group == "all_mixed" else group
    return {
        "logical_group": group,
        "canonical_group": canonical,
        "physical_run": str(physical).lower(),
        "object": object_name,
        "seed": str(seed),
        "run_name": f"m18_{canonical}_{object_name}_seed{seed}",
        "run_signature": f"signature-{canonical}-{object_name}",
        "dice": f"{dice:.6f}",
        "miou": "0.400000",
        "pixel_auroc": "0.800000",
        "aupro": "0.450000",
        "model_sha256": digest,
    }


def _write(path: Path, rows: Sequence[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _baseline(tmp_path: Path) -> Path:
    rows = [
        _row(group, object_name, 42, dice=0.5, digest="a" * 64)
        for group in (*FORMAL_GROUPS, "all_mixed")
        for object_name in OBJECTS
    ]
    return _write(tmp_path / "baseline.csv", rows)


def _current(tmp_path: Path, **overrides: object) -> Path:
    rows = []
    for group in (*FORMAL_GROUPS, "all_mixed"):
        for object_name in OBJECTS:
            for seed in SEEDS:
                dice = 0.5 if seed == 42 else 0.9
                digest = "a" * 64
                if seed == 42 and group == overrides.get("changed_group"):
                    dice = float(overrides.get("changed_dice", 0.5))
                    digest = str(overrides.get("changed_digest", "a" * 64))
                rows.append(_row(group, object_name, seed, dice=dice, digest=digest))
    return _write(tmp_path / "segmentation.csv", rows)


def test_identical_rerun_is_reported_as_bit_identical(tmp_path: Path) -> None:
    payload = compare(_baseline(tmp_path), _current(tmp_path))
    assert payload["bit_identical"] is True
    # all_mixed is an alias and must not be counted as a rerun.
    assert payload["compared_physical_runs"] == len(FORMAL_GROUPS) * len(OBJECTS)
    assert payload["model_sha256_matches"] == payload["compared_physical_runs"]
    assert all(value == 0.0 for value in payload["max_abs_metric_delta"].values())
    assert "逐 bit 相同" in build_report(payload)


def test_other_seeds_never_enter_the_comparison(tmp_path: Path) -> None:
    # Seeds 43/44 carry a very different Dice; the anchor comparison must ignore them.
    payload = compare(_baseline(tmp_path), _current(tmp_path))
    assert payload["anchor_seed"] == 42
    assert payload["max_abs_metric_delta"]["dice"] == 0.0


def test_a_changed_metric_is_reported_rather_than_hidden(tmp_path: Path) -> None:
    current = _current(
        tmp_path,
        changed_group="real_only",
        changed_dice=0.625,
        changed_digest="b" * 64,
    )
    payload = compare(_baseline(tmp_path), current)
    assert payload["bit_identical"] is False
    assert payload["model_sha256_matches"] == payload["compared_physical_runs"] - len(OBJECTS)
    assert payload["max_abs_metric_delta"]["dice"] == pytest.approx(0.125)
    assert "不完全相同" in build_report(payload)


def test_comparison_refuses_tables_covering_different_groups(tmp_path: Path) -> None:
    rows = [
        _row(group, object_name, 42, dice=0.5, digest="a" * 64)
        for group in FORMAL_GROUPS[:-1]
        for object_name in OBJECTS
    ]
    baseline = _write(tmp_path / "baseline.csv", rows)
    with pytest.raises(ReproductionCheckError, match="different groups"):
        compare(baseline, _current(tmp_path))


def test_comparison_refuses_a_table_without_the_anchor_seed(tmp_path: Path) -> None:
    rows = [
        _row(group, object_name, 43, dice=0.5, digest="a" * 64)
        for group in (*FORMAL_GROUPS, "all_mixed")
        for object_name in OBJECTS
    ]
    baseline = _write(tmp_path / "baseline.csv", rows)
    with pytest.raises(ReproductionCheckError, match="No seed-42 rows"):
        compare(baseline, _current(tmp_path))
