"""Apply the ADR-038 decision rules that were fixed before the v4 pilot ran.

Primary contrast, on validation Macro-F1, per object:

    D = db_inband - db_current

`pcb1` is the primary object because it is the one whose placement areas fall outside the
real band. `capsules` is a pre-declared control: 99.8% of its filtered samples were already
in band, so it must not move much; if it does, the difference came from selection randomness
rather than area.

ADR-038 also carried forward the ADR-034/ADR-036 lesson: if every candidate scores the same
on an object, that object is non-discriminating and its result may not be used as evidence.
When that happens on the primary object the pilot did not test its hypothesis, and this
script says so instead of reporting a null result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import write_text_lf  # isort: skip

PRIMARY_OBJECT = "pcb1"
CONTROL_OBJECT = "capsules"
BASELINE = "real_only"
CURRENT_ARM = "db_current"
INBAND_ARM = "db_inband"
PRIMARY_METRIC = "macro_f1"
EFFECT_THRESHOLD = 0.01
CONTROL_ALERT = 0.05


class PlacementBandDecisionError(RuntimeError):
    """Raised when the v4 result cannot support the pre-registered rules."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlacementBandDecisionError(message)


def load_pilot(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing v4 pilot result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"Invalid v4 pilot result: {path}")
    return payload


def index_runs(payload: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    seed = int(payload["seed"])
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for run in payload["runs"]:
        name = str(run["run_name"])
        suffix = f"_{run['object']}_seed_{seed}_dev"
        require(name.startswith("m26_") and name.endswith(suffix), f"Bad run name: {name}")
        key = (name[len("m26_") : -len(suffix)], str(run["object"]))
        require(key not in indexed, f"Duplicate run: {key}")
        indexed[key] = run
    return indexed


def _metric(
    indexed: Mapping[tuple[str, str], Mapping[str, Any]],
    candidate: str,
    object_name: str,
    metric: str = PRIMARY_METRIC,
) -> float:
    key = (candidate, object_name)
    require(key in indexed, f"Missing candidate: {candidate}/{object_name}")
    value = float(indexed[key]["metrics"][metric])
    require(math.isfinite(value), f"Non-finite {metric}: {key}")
    return value


def decide(payload: Mapping[str, Any]) -> dict[str, Any]:
    indexed = index_runs(payload)
    objects: dict[str, Any] = {}
    for object_name in sorted({name for _, name in indexed}):
        scores = {
            arm: _metric(indexed, arm, object_name)
            for arm in (BASELINE, CURRENT_ARM, INBAND_ARM)
        }
        delta = scores[INBAND_ARM] - scores[CURRENT_ARM]
        objects[object_name] = {
            "scores": scores,
            "delta": delta,
            "metric_discriminates": len(set(scores.values())) > 1,
        }

    require(PRIMARY_OBJECT in objects, f"Missing primary object: {PRIMARY_OBJECT}")
    primary = objects[PRIMARY_OBJECT]
    if not primary["metric_discriminates"]:
        # Pre-registered in ADR-038: a blind object cannot serve as evidence. The pilot
        # therefore failed to test its own hypothesis rather than returning a null.
        verdict = "uninformative_primary_object"
    elif primary["delta"] >= EFFECT_THRESHOLD:
        verdict = "in_band_helps"
    elif primary["delta"] <= -EFFECT_THRESHOLD:
        verdict = "in_band_harms"
    else:
        verdict = "no_effect"

    control = objects.get(CONTROL_OBJECT)
    control_anomalous = (
        control is not None and abs(control["delta"]) >= CONTROL_ALERT
    )
    gate = payload.get("gate", {})
    return {
        "status": "passed",
        "schema_version": 1,
        "source": "results/v4/pilot_classification.json",
        "preregistered_in": "ADR-038",
        "primary_object": PRIMARY_OBJECT,
        "control_object": CONTROL_OBJECT,
        "metric": PRIMARY_METRIC,
        "objects": objects,
        "verdict": verdict,
        "control_anomalous": control_anomalous,
        "confirmatory_run_authorized_by_gate": bool(
            gate.get("confirmatory_run_authorized_by_gate", False)
        ),
        "gate_status": str(gate.get("status", "unknown")),
    }


def build_report(payload: Mapping[str, Any]) -> str:
    labels = {
        "in_band_helps": "面積限回真實分布**有幫助**",
        "in_band_harms": "面積限回真實分布**有害**",
        "no_effect": "**無效果**",
        "uninformative_primary_object": "**本次未能檢驗假說**（主要物件無鑑別力）",
    }
    objects = payload["objects"]
    lines = [
        "# v4 放置面積 pilot 的預註冊判定",
        "",
        (
            "本報告只執行 [ADR-038](../docs/decisions.md#adr-038) 在**執行前**寫死的規則。"
            "數字由 `scripts/decide_v4_placement_band.py` 從 "
            "`results/v4/pilot_classification.json` 產生。"
        ),
        "",
        (
            f"主要物件：`{payload['primary_object']}`　"
            f"對照物件：`{payload['control_object']}`　"
            f"指標：`{payload['metric']}`"
        ),
        "",
        "| 物件 | real_only | db_current | db_inband | D = inband − current | 指標可分辨 |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for object_name, item in objects.items():
        scores = item["scores"]
        lines.append(
            f"| {object_name} | {scores[BASELINE]:.4f} | {scores[CURRENT_ARM]:.4f} "
            f"| {scores[INBAND_ARM]:.4f} | {item['delta']:+.4f} "
            f"| {'是' if item['metric_discriminates'] else '**否**'} |"
        )
    lines += ["", f"**判定：{labels[payload['verdict']]}**", ""]

    if payload["verdict"] == "uninformative_primary_object":
        lines += [
            (
                f"主要物件 `{payload['primary_object']}` 上三個 candidate 的 "
                f"`{payload['metric']}` **完全相同**，該指標在此物件無法分辨任何差異。"
                "ADR-038 已預先規定無鑑別力的物件不得當作證據，因此本次**不是**「沒有效果」，"
                "而是**這個實驗沒有真正檢驗到自己的假說**。"
            ),
            "",
            (
                "這是預註冊設計的缺陷：[ADR-036](../docs/decisions.md#adr-036) 已經記錄過"
                f"`{payload['primary_object']}` 的 `{payload['metric']}` 對所有 candidate 相同，"
                "ADR-038 卻仍以此組合作為主要判準。"
            ),
            "",
        ]

    control = objects.get(payload["control_object"])
    if control is not None:
        state = "**異常**" if payload["control_anomalous"] else "正常"
        lines += [
            (
                f"對照組 `{payload['control_object']}`：`D = {control['delta']:+.4f}`，"
                f"門檻 `{CONTROL_ALERT}`，判定{state}。"
            ),
            "",
        ]
        if payload["control_anomalous"]:
            lines += [
                (
                    "對照組本應幾乎不動（它有 99.8% 的樣本原本就在區間內）。"
                    "既然它明顯變動，代表差異來自選樣隨機性而非面積，本次結論可信度下降。"
                ),
                "",
            ]

    lines += [
        "## Confirmatory gate",
        "",
        f"- gate 狀態：`{payload['gate_status']}`",
        (
            "- 是否授權 confirmatory test："
            f"**{'是' if payload['confirmatory_run_authorized_by_gate'] else '否'}**"
        ),
        "",
        (
            "判定結論成立**不等於** gate 通過。gate 未過時一律不得讀 frozen test、"
            "不得跑 3 seeds（ADR-038）。"
        ),
        "",
    ]
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_text_lf(temporary, text)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot",
        type=Path,
        default=Path("results/v4/pilot_classification.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v4_placement_band.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/v4_placement_band.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = decide(load_pilot(args.pilot))
    atomic_write(args.report, build_report(payload))
    atomic_write(
        args.output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(payload["verdict"], ensure_ascii=False))
    print(f"control_anomalous={payload['control_anomalous']}")
    print(f"confirmatory_authorized={payload['confirmatory_run_authorized_by_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
