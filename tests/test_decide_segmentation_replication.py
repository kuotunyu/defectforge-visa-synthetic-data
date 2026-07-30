from __future__ import annotations

from pathlib import Path

import pytest

from scripts.aggregate_segmentation import LOGICAL_GROUPS, SEEDS
from scripts.build_phase2_figures import OBJECTS
from scripts.decide_segmentation_replication import (
    ReplicationDecisionError,
    build_report,
    decide,
    read_rows,
)


def _rows(
    *,
    conflict_seeds: dict[str, tuple[int, ...]] | None = None,
    std_aug_zero_seeds: tuple[int, ...] = (),
) -> list[dict[str, str]]:
    """Build a full replication table, then bend only the two decisive quantities."""
    conflict_seeds = conflict_seeds or {}
    rows: list[dict[str, str]] = []
    for group_name in LOGICAL_GROUPS:
        for object_name in OBJECTS:
            for seed in SEEDS:
                dice, aupro = 0.40, 0.40
                if group_name == "real_only":
                    dice, aupro = 0.50, 0.50
                elif group_name in {"filtered_syn", "all_mixed"}:
                    # Dice always drops; AUPRO only rises on the requested seeds.
                    dice = 0.10
                    aupro = (
                        0.90 if seed in conflict_seeds.get(object_name, ()) else 0.20
                    )
                elif group_name == "std_aug" and object_name == "capsules":
                    dice = 0.0 if seed in std_aug_zero_seeds else 0.35
                rows.append(
                    {
                        "logical_group": group_name,
                        "canonical_group": (
                            "filtered_syn" if group_name == "all_mixed" else group_name
                        ),
                        "object": object_name,
                        "seed": str(seed),
                        "dice": f"{dice:.4f}",
                        "miou": "0.40",
                        "pixel_auroc": "0.80",
                        "aupro": f"{aupro:.4f}",
                    }
                )
    return rows


def test_conflict_is_real_when_one_object_reaches_two_seeds() -> None:
    payload = decide(_rows(conflict_seeds={"pcb1": (SEEDS[0], SEEDS[2])}))
    conflict = payload["direction_conflict"]
    assert conflict["verdict"] == "real_phenomenon"
    assert conflict["triggering_objects"] == ["pcb1"]
    assert conflict["objects"]["pcb1"]["conflicting_seeds"] == [SEEDS[0], SEEDS[2]]
    assert conflict["objects"]["capsules"]["meets_rule"] is False


def test_conflict_is_an_artefact_when_no_object_reaches_two_seeds() -> None:
    payload = decide(
        _rows(conflict_seeds={"pcb1": (SEEDS[0],), "capsules": (SEEDS[1],)})
    )
    conflict = payload["direction_conflict"]
    assert conflict["verdict"] == "single_seed_artefact"
    assert conflict["triggering_objects"] == []


def test_collapse_is_systematic_at_two_zero_seeds_and_keeps_adr_031() -> None:
    payload = decide(_rows(std_aug_zero_seeds=(SEEDS[0], SEEDS[2])))
    collapse = payload["augmentation_collapse"]
    assert collapse["verdict"] == "systematic"
    assert collapse["zero_dice_seeds"] == [SEEDS[0], SEEDS[2]]
    assert collapse["adr_031_withdrawn"] is False


def test_collapse_is_noise_at_one_zero_seed_and_withdraws_adr_031() -> None:
    payload = decide(_rows(std_aug_zero_seeds=(SEEDS[0],)))
    collapse = payload["augmentation_collapse"]
    assert collapse["verdict"] == "seed_noise"
    assert collapse["adr_031_withdrawn"] is True
    assert "撤回" in build_report(payload)


def test_a_zero_delta_never_counts_as_a_direction_conflict() -> None:
    rows = _rows()
    for row in rows:
        if row["logical_group"] in {"filtered_syn", "all_mixed"}:
            # Identical AUPRO to real_only: a tie, not an opposing direction.
            row["aupro"] = "0.5000"
    conflict = decide(rows)["direction_conflict"]
    assert conflict["verdict"] == "single_seed_artefact"
    for item in conflict["objects"].values():
        assert item["conflicting_seeds"] == []


def test_decide_refuses_a_table_that_is_not_the_replication_set() -> None:
    rows = [row for row in _rows() if int(row["seed"]) == SEEDS[0]]
    with pytest.raises(ReplicationDecisionError, match="expected"):
        decide(rows)


def test_decide_refuses_a_duplicated_group(tmp_path: Path) -> None:
    rows = _rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ReplicationDecisionError, match="Expected one row"):
        decide(rows)


def test_read_rows_rejects_a_missing_table(tmp_path: Path) -> None:
    with pytest.raises(ReplicationDecisionError, match="Missing segmentation CSV"):
        read_rows(tmp_path / "absent.csv")
