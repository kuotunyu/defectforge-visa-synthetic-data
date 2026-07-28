from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_v2_classifier_pilot import evaluate_gate, load_pilot_config


def _report(macro_f1: float, auroc: float) -> dict[str, object]:
    return {"metrics": {"macro_f1": macro_f1, "auroc": auroc}}


def test_v2_pilot_config_is_preregistered() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_pilot_config(root / "configs/classifier_v2_pilot.yaml")
    assert config["status"] == "preregistered"
    assert config["objects"] == ["pcb1", "capsules"]
    assert {item["name"] for item in config["candidates"]} >= {
        "real_only",
        "v1_class_balanced",
        "domain_balanced_50",
        "domain_balanced_75",
    }


def test_v2_gate_authorizes_noninferior_mean_improvement() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_pilot_config(root / "configs/classifier_v2_pilot.yaml")
    reports = {
        "real_only": {
            "pcb1": _report(0.60, 0.80),
            "capsules": _report(0.50, 0.70),
        },
        "v1_class_balanced": {
            "pcb1": _report(0.40, 0.60),
            "capsules": _report(0.30, 0.50),
        },
        "domain_balanced_50": {
            "pcb1": _report(0.62, 0.81),
            "capsules": _report(0.52, 0.70),
        },
        "domain_balanced_75": {
            "pcb1": _report(0.61, 0.80),
            "capsules": _report(0.51, 0.71),
        },
    }
    gate = evaluate_gate(reports, config)
    assert gate["status"] == "passed"
    assert gate["winner"] == "domain_balanced_50"
    assert gate["confirmatory_run_authorized_by_gate"] is True


def test_v2_gate_stops_object_regression() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_pilot_config(root / "configs/classifier_v2_pilot.yaml")
    reports = {
        "real_only": {
            "pcb1": _report(0.60, 0.80),
            "capsules": _report(0.50, 0.70),
        },
        "v1_class_balanced": {
            "pcb1": _report(0.40, 0.60),
            "capsules": _report(0.30, 0.50),
        },
        "domain_balanced_50": {
            "pcb1": _report(0.70, 0.85),
            "capsules": _report(0.45, 0.65),
        },
        "domain_balanced_75": {
            "pcb1": _report(0.69, 0.84),
            "capsules": _report(0.44, 0.66),
        },
    }
    gate = evaluate_gate(reports, config)
    assert gate["status"] == "stopped"
    assert gate["confirmatory_run_authorized_by_gate"] is False
    assert gate["object_checks"]["capsules"]["macro_f1_noninferior"] is False
    assert gate["mean_macro_f1_gain_vs_real_only"] == pytest.approx(0.025)


def test_v2_verifier_exposes_help() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/verify_v2_classifier_pilot.py", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
