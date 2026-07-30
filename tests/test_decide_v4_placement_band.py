from __future__ import annotations

from pathlib import Path

import pytest

from scripts.decide_v4_placement_band import (
    PlacementBandDecisionError,
    build_report,
    decide,
    load_pilot,
)


def _payload(
    *,
    pcb1: tuple[float, float, float],
    capsules: tuple[float, float, float],
    authorized: bool = False,
) -> dict:
    """Each tuple is (real_only, db_current, db_inband) for that object."""
    scores = {
        "real_only": {"pcb1": pcb1[0], "capsules": capsules[0]},
        "db_current": {"pcb1": pcb1[1], "capsules": capsules[1]},
        "db_inband": {"pcb1": pcb1[2], "capsules": capsules[2]},
    }
    runs = [
        {
            "run_name": f"m26_{name}_{object_name}_seed_42_dev",
            "object": object_name,
            "metrics": {"macro_f1": value, "auroc": 0.5},
        }
        for name, per_object in scores.items()
        for object_name, value in per_object.items()
    ]
    return {
        "seed": 42,
        "runs": runs,
        "gate": {
            "status": "passed" if authorized else "stopped",
            "confirmatory_run_authorized_by_gate": authorized,
        },
    }


def test_a_blind_primary_object_is_not_reported_as_a_null_result() -> None:
    # The v4 outcome: every pcb1 candidate scores the same, so the pilot never tested itself.
    payload = decide(_payload(pcb1=(0.6944, 0.6944, 0.6944), capsules=(0.81, 0.68, 0.69)))
    assert payload["verdict"] == "uninformative_primary_object"
    assert payload["objects"]["pcb1"]["metric_discriminates"] is False
    report = build_report(payload)
    assert "沒有真正檢驗到自己的假說" in report
    assert "無效果" not in report.split("**判定：")[1].split("**")[0]


def test_in_band_helps_only_above_the_threshold() -> None:
    payload = decide(_payload(pcb1=(0.70, 0.60, 0.62), capsules=(0.81, 0.68, 0.69)))
    assert payload["verdict"] == "in_band_helps"


def test_in_band_harms_below_the_negative_threshold() -> None:
    payload = decide(_payload(pcb1=(0.70, 0.62, 0.60), capsules=(0.81, 0.68, 0.69)))
    assert payload["verdict"] == "in_band_harms"


def test_a_small_primary_delta_is_no_effect_when_the_metric_discriminates() -> None:
    payload = decide(_payload(pcb1=(0.70, 0.60, 0.605), capsules=(0.81, 0.68, 0.69)))
    assert payload["objects"]["pcb1"]["metric_discriminates"] is True
    assert payload["verdict"] == "no_effect"


def test_a_quiet_control_is_reported_as_normal() -> None:
    payload = decide(_payload(pcb1=(0.70, 0.60, 0.62), capsules=(0.81, 0.6815, 0.6889)))
    assert payload["control_anomalous"] is False
    assert "判定正常" in build_report(payload)


def test_a_moving_control_downgrades_confidence() -> None:
    payload = decide(_payload(pcb1=(0.70, 0.60, 0.62), capsules=(0.81, 0.60, 0.70)))
    assert payload["control_anomalous"] is True
    report = build_report(payload)
    assert "判定**異常**" in report
    assert "可信度下降" in report


def test_the_verdict_never_implies_the_gate_passed() -> None:
    payload = decide(_payload(pcb1=(0.70, 0.60, 0.62), capsules=(0.81, 0.68, 0.69)))
    assert payload["confirmatory_run_authorized_by_gate"] is False
    assert "不得讀 frozen test" in build_report(payload)


def test_gate_authorisation_is_carried_through_unchanged() -> None:
    payload = decide(
        _payload(pcb1=(0.70, 0.60, 0.62), capsules=(0.81, 0.68, 0.69), authorized=True)
    )
    assert payload["confirmatory_run_authorized_by_gate"] is True


def test_a_missing_arm_fails_closed() -> None:
    payload = _payload(pcb1=(0.70, 0.60, 0.62), capsules=(0.81, 0.68, 0.69))
    payload["runs"] = [
        run for run in payload["runs"] if "db_inband_pcb1" not in run["run_name"]
    ]
    with pytest.raises(PlacementBandDecisionError, match="Missing candidate"):
        decide(payload)


def test_load_pilot_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PlacementBandDecisionError, match="Missing v4 pilot result"):
        load_pilot(tmp_path / "absent.json")
