from __future__ import annotations

from pathlib import Path

import pytest

from scripts.decide_v5_seed_replication import (
    DISCLOSED_SEED,
    VERDICT_SEEDS,
    SeedReplicationError,
    build_report,
    decide,
    load_seed,
)


def _indexed(seed: int, per_object: dict[str, tuple[float, float, float]]) -> dict:
    """Each tuple is (real_only, db_current, db_inband) AUROC for that object."""
    runs = {}
    for object_name, (baseline, current, inband) in per_object.items():
        for name, value in (
            ("real_only", baseline),
            ("db_current", current),
            ("db_inband", inband),
        ):
            runs[(name, object_name)] = {
                "run_name": f"m26_{name}_{object_name}_seed_{seed}_dev",
                "object": object_name,
                "metrics": {"auroc": value, "macro_f1": 0.5},
            }
    return runs


def _seeds(
    *,
    seed42: dict[str, tuple[float, float, float]],
    seed43: dict[str, tuple[float, float, float]],
    seed44: dict[str, tuple[float, float, float]],
) -> dict[int, dict]:
    return {
        42: _indexed(42, seed42),
        43: _indexed(43, seed43),
        44: _indexed(44, seed44),
    }


def _gate(authorized: bool = False) -> dict[int, dict]:
    return {
        seed: {
            "status": "passed" if authorized else "stopped",
            "confirmatory_run_authorized_by_gate": authorized,
        }
        for seed in (42, 43, 44)
    }


def _flat(primary: tuple[float, float, float]) -> dict[str, tuple[float, float, float]]:
    return {"pcb1": primary, "capsules": (0.90, 0.86, 0.862)}


def test_the_disclosed_seed_never_enters_the_verdict() -> None:
    # Seed 42 is strongly negative; the fresh seeds are flat. The verdict must follow the
    # fresh seeds, otherwise the one seed whose direction was already known would decide.
    payload = decide(
        _seeds(
            seed42=_flat((0.92, 0.88, 0.80)),
            seed43=_flat((0.96, 0.96, 0.9605)),
            seed44=_flat((0.97, 0.93, 0.9305)),
        ),
        _gate(),
    )
    assert payload["disclosed_seed"] == DISCLOSED_SEED
    assert payload["verdict"] == "no_effect"
    # 0.9605-0.96 and 0.9305-0.93 both give +0.0005; seed 42's -0.08 must not pull it down.
    assert payload["primary_mean_delta"] == pytest.approx(0.0005, abs=1e-6)
    report = build_report(payload)
    assert "不進入任何判定式" in report


def test_help_is_declared_only_above_the_threshold() -> None:
    payload = decide(
        _seeds(
            seed42=_flat((0.92, 0.88, 0.80)),
            seed43=_flat((0.96, 0.90, 0.93)),
            seed44=_flat((0.97, 0.90, 0.93)),
        ),
        _gate(),
    )
    assert payload["verdict"] == "in_band_helps"


def test_harm_is_declared_only_below_the_negative_threshold() -> None:
    payload = decide(
        _seeds(
            seed42=_flat((0.92, 0.88, 0.95)),
            seed43=_flat((0.96, 0.93, 0.90)),
            seed44=_flat((0.97, 0.93, 0.90)),
        ),
        _gate(),
    )
    assert payload["verdict"] == "in_band_harms"


def test_all_verdict_seeds_blind_is_uninformative_not_null() -> None:
    payload = decide(
        _seeds(
            seed42=_flat((0.92, 0.88, 0.80)),
            seed43=_flat((0.90, 0.90, 0.90)),
            seed44=_flat((0.91, 0.91, 0.91)),
        ),
        _gate(),
    )
    assert payload["usable_verdict_seeds"] == []
    assert payload["verdict"] == "uninformative"
    assert payload["primary_mean_delta"] is None
    assert "未能檢驗" in build_report(payload)


def test_a_blind_seed_drops_out_but_the_other_still_decides() -> None:
    payload = decide(
        _seeds(
            seed42=_flat((0.92, 0.88, 0.80)),
            seed43=_flat((0.90, 0.90, 0.90)),
            seed44=_flat((0.97, 0.90, 0.93)),
        ),
        _gate(),
    )
    assert payload["usable_verdict_seeds"] == [44]
    assert payload["primary_mean_delta"] == pytest.approx(0.03)
    assert payload["verdict"] == "in_band_helps"


def test_a_moving_control_is_flagged() -> None:
    seeds = _seeds(
        seed42=_flat((0.92, 0.88, 0.80)),
        seed43=_flat((0.96, 0.96, 0.9605)),
        seed44=_flat((0.97, 0.93, 0.9305)),
    )
    for seed in VERDICT_SEEDS:
        seeds[seed][("db_inband", "capsules")]["metrics"]["auroc"] = 0.95
    payload = decide(seeds, _gate())
    assert payload["control_anomalous"] is True
    assert "可信度下降" in build_report(payload)


def test_a_quiet_control_is_not_flagged() -> None:
    payload = decide(
        _seeds(
            seed42=_flat((0.92, 0.88, 0.80)),
            seed43=_flat((0.96, 0.96, 0.9605)),
            seed44=_flat((0.97, 0.93, 0.9305)),
        ),
        _gate(),
    )
    assert payload["control_anomalous"] is False
    assert "判定正常" in build_report(payload)


def test_the_verdict_never_implies_the_gate_passed() -> None:
    payload = decide(
        _seeds(
            seed42=_flat((0.92, 0.88, 0.80)),
            seed43=_flat((0.96, 0.90, 0.93)),
            seed44=_flat((0.97, 0.90, 0.93)),
        ),
        _gate(),
    )
    assert payload["verdict"] == "in_band_helps"
    assert payload["confirmatory_run_authorized_by_gate"] is False
    assert "不得讀 frozen test" in build_report(payload)


def test_a_missing_seed_fails_closed() -> None:
    seeds = _seeds(
        seed42=_flat((0.92, 0.88, 0.80)),
        seed43=_flat((0.96, 0.96, 0.96)),
        seed44=_flat((0.97, 0.93, 0.93)),
    )
    del seeds[44]
    with pytest.raises(SeedReplicationError, match="Missing seed 44"):
        decide(seeds, _gate())


def test_a_pilot_that_loaded_test_data_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pilot.json"
    path.write_text(
        '{"seed": 43, "test_data_loaded": true, "runs": []}',
        encoding="utf-8",
    )
    with pytest.raises(SeedReplicationError, match="loaded test data"):
        load_seed(path, 43)


def test_a_seed_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pilot.json"
    path.write_text(
        '{"seed": 42, "test_data_loaded": false, "runs": []}',
        encoding="utf-8",
    )
    with pytest.raises(SeedReplicationError, match="is not seed 43"):
        load_seed(path, 43)
