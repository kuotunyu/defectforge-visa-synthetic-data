from __future__ import annotations

from pathlib import Path

import pytest

from scripts.decide_v3_source_attribution import (
    AttributionError,
    attribute,
    build_report,
    decide,
    load_pilot,
)

OBJECTS = ("pcb1", "capsules")


def _payload(scores: dict[str, dict[str, float]], *, authorized: bool = False) -> dict:
    """Mirror the runner's real schema: a flat run list keyed by encoded run names."""
    runs = []
    for name, per_object in scores.items():
        for object_name, value in per_object.items():
            runs.append(
                {
                    "run_name": f"m26_{name}_{object_name}_seed_42_dev",
                    "object": object_name,
                    "metrics": {"macro_f1": value, "auroc": 0.5},
                }
            )
    return {
        "seed": 42,
        "runs": runs,
        "gate": {
            "status": "passed" if authorized else "stopped",
            "confirmatory_run_authorized_by_gate": authorized,
        },
    }


def _scores(
    *,
    pcb1: tuple[float, float, float],
    capsules: tuple[float, float, float],
) -> dict[str, dict[str, float]]:
    """Each tuple is (real_only, db_copypaste, db_diffusion) for that object."""
    return {
        "real_only": {"pcb1": pcb1[0], "capsules": capsules[0]},
        "db_copypaste": {"pcb1": pcb1[1], "capsules": capsules[1]},
        "db_diffusion": {"pcb1": pcb1[2], "capsules": capsules[2]},
    }


def test_placement_dominates_only_when_it_wins_on_every_object() -> None:
    # P = 0.30, A = 0.05 on both objects.
    payload = _payload(_scores(pcb1=(0.70, 0.40, 0.35), capsules=(0.80, 0.50, 0.45)))
    result = attribute(payload)
    assert result["verdict"] == "placement_dominates"
    assert result["objects"]["pcb1"]["placement_penalty"] == pytest.approx(0.30)
    assert result["objects"]["pcb1"]["appearance_penalty"] == pytest.approx(0.05)


def test_appearance_dominates_only_when_it_wins_on_every_object() -> None:
    # P = 0.05, A = 0.30 on both objects.
    payload = _payload(_scores(pcb1=(0.70, 0.65, 0.35), capsules=(0.80, 0.75, 0.45)))
    assert attribute(payload)["verdict"] == "appearance_dominates"


def test_a_split_decision_is_reported_as_object_dependent() -> None:
    # Placement wins on pcb1, appearance wins on capsules.
    payload = _payload(_scores(pcb1=(0.70, 0.40, 0.35), capsules=(0.80, 0.75, 0.45)))
    result = attribute(payload)
    assert result["verdict"] == "object_dependent"
    assert result["objects"]["pcb1"]["dominant"] == "placement"
    assert result["objects"]["capsules"]["dominant"] == "appearance"


def test_an_exact_tie_never_supports_either_side() -> None:
    # P == A on capsules, so no single cause may be claimed.
    payload = _payload(_scores(pcb1=(0.70, 0.40, 0.35), capsules=(0.80, 0.60, 0.40)))
    result = attribute(payload)
    assert result["objects"]["capsules"]["dominant"] == "tie"
    assert result["verdict"] == "object_dependent"


def test_report_always_states_the_bundling_caveat() -> None:
    payload = _payload(_scores(pcb1=(0.70, 0.40, 0.35), capsules=(0.80, 0.50, 0.45)))
    report = build_report(decide(payload))
    assert "縫合" in report
    assert "不是三個乾淨因子的分解" in report


def test_attribution_never_implies_the_gate_passed() -> None:
    payload = _payload(
        _scores(pcb1=(0.70, 0.40, 0.35), capsules=(0.80, 0.50, 0.45)),
        authorized=False,
    )
    decided = decide(payload)
    assert decided["attribution"]["verdict"] == "placement_dominates"
    assert decided["confirmatory_run_authorized_by_gate"] is False
    report = build_report(decided)
    assert "不得讀 frozen test" in report


def test_gate_authorisation_is_carried_through_unchanged() -> None:
    payload = _payload(
        _scores(pcb1=(0.70, 0.40, 0.35), capsules=(0.80, 0.50, 0.45)),
        authorized=True,
    )
    assert decide(payload)["confirmatory_run_authorized_by_gate"] is True


def test_missing_candidate_fails_closed() -> None:
    scores = _scores(pcb1=(0.70, 0.40, 0.35), capsules=(0.80, 0.50, 0.45))
    del scores["db_copypaste"]
    with pytest.raises(AttributionError, match="missing candidate"):
        attribute(_payload(scores))


def test_load_pilot_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AttributionError, match="Missing v3 pilot result"):
        load_pilot(tmp_path / "absent.json")


def test_a_metric_that_cannot_separate_candidates_is_flagged() -> None:
    # pcb1 gives every candidate the same Macro-F1, so its "tie" carries no information.
    payload = _payload(_scores(pcb1=(0.6944, 0.6944, 0.6944), capsules=(0.80, 0.50, 0.45)))
    result = attribute(payload)
    assert result["objects"]["pcb1"]["metric_discriminates"] is False
    assert result["objects"]["capsules"]["metric_discriminates"] is True
    assert result["non_discriminating_objects"] == ["pcb1"]
    report = build_report(decide(payload))
    assert "無法分辨" in report
    assert "沒有提供資訊" in report


def test_a_discriminating_object_is_not_flagged() -> None:
    payload = _payload(_scores(pcb1=(0.70, 0.40, 0.35), capsules=(0.80, 0.50, 0.45)))
    result = attribute(payload)
    assert result["non_discriminating_objects"] == []
    assert "無法分辨" not in build_report(decide(payload))
