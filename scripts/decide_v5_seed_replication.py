"""Apply the ADR-040 decision rules that were fixed before the v5 seeds ran.

Primary metric is AUROC, because ADR-039 established that Macro-F1 cannot separate the
candidates on `pcb1`. The verdict averages `D = db_inband - db_current` over the seeds whose
results had never been observed when ADR-040 was written.

Seed 42 is reported and then excluded from every decision expression. Its AUROC values were
quoted in ADR-039, so it is the one seed this analysis is not blind to. Excluding it removes
the only seed that could have been chosen with knowledge of its direction.

ADR-040 also carried forward the ADR-039 requirement: a seed on which every candidate scores
the same is non-discriminating and drops out, and if no verdict seed discriminates the result
is `uninformative` rather than a null.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
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
METRIC = "auroc"
VERDICT_SEEDS = (43, 44)
DISCLOSED_SEED = 42
# Reused verbatim from ADR-026's per-object AUROC tolerance; not a new threshold.
EFFECT_THRESHOLD = 0.02
CONTROL_ALERT = 0.05


class SeedReplicationError(RuntimeError):
    """Raised when the v5 replication cannot support the pre-registered rules."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SeedReplicationError(message)


def load_seed(path: Path, seed: int) -> dict[tuple[str, str], Mapping[str, Any]]:
    require(path.is_file(), f"Missing pilot result for seed {seed}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(int(payload["seed"]) == seed, f"Pilot at {path} is not seed {seed}")
    require(
        payload.get("test_data_loaded") is False,
        f"Pilot at {path} reports having loaded test data",
    )
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for run in payload["runs"]:
        name = str(run["run_name"])
        suffix = f"_{run['object']}_seed_{seed}_dev"
        require(name.startswith("m26_") and name.endswith(suffix), f"Bad run name: {name}")
        key = (name[len("m26_") : -len(suffix)], str(run["object"]))
        require(key not in indexed, f"Duplicate run at seed {seed}: {key}")
        indexed[key] = run
    return indexed


def _value(
    indexed: Mapping[tuple[str, str], Mapping[str, Any]],
    candidate: str,
    object_name: str,
) -> float:
    key = (candidate, object_name)
    require(key in indexed, f"Missing candidate: {candidate}/{object_name}")
    value = float(indexed[key]["metrics"][METRIC])
    require(math.isfinite(value), f"Non-finite {METRIC}: {key}")
    return value


def _per_seed(
    seeds: Mapping[int, Mapping[tuple[str, str], Mapping[str, Any]]],
    object_name: str,
) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for seed, indexed in seeds.items():
        scores = {
            arm: _value(indexed, arm, object_name)
            for arm in (BASELINE, CURRENT_ARM, INBAND_ARM)
        }
        out[seed] = {
            "scores": scores,
            "delta": scores[INBAND_ARM] - scores[CURRENT_ARM],
            "metric_discriminates": len(set(scores.values())) > 1,
        }
    return out


def _mean_delta(per_seed: Mapping[int, Mapping[str, Any]]) -> tuple[float | None, list[int]]:
    usable = [seed for seed in VERDICT_SEEDS if per_seed[seed]["metric_discriminates"]]
    if not usable:
        return None, usable
    return statistics.fmean(per_seed[seed]["delta"] for seed in usable), usable


def decide(
    seeds: Mapping[int, Mapping[tuple[str, str], Mapping[str, Any]]],
    gate_status: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    for seed in (*VERDICT_SEEDS, DISCLOSED_SEED):
        require(seed in seeds, f"Missing seed {seed}")
    primary = _per_seed(seeds, PRIMARY_OBJECT)
    control = _per_seed(seeds, CONTROL_OBJECT)
    primary_mean, usable = _mean_delta(primary)
    control_mean, _ = _mean_delta(control)

    if primary_mean is None:
        verdict = "uninformative"
    elif primary_mean >= EFFECT_THRESHOLD:
        verdict = "in_band_helps"
    elif primary_mean <= -EFFECT_THRESHOLD:
        verdict = "in_band_harms"
    else:
        verdict = "no_effect"

    authorized = any(
        bool(item.get("confirmatory_run_authorized_by_gate", False))
        for seed, item in gate_status.items()
        if seed in VERDICT_SEEDS
    )
    return {
        "status": "passed",
        "schema_version": 1,
        "preregistered_in": "ADR-040",
        "metric": METRIC,
        "primary_object": PRIMARY_OBJECT,
        "control_object": CONTROL_OBJECT,
        "verdict_seeds": list(VERDICT_SEEDS),
        "disclosed_seed": DISCLOSED_SEED,
        "usable_verdict_seeds": usable,
        "primary": primary,
        "control": control,
        "primary_mean_delta": primary_mean,
        "control_mean_delta": control_mean,
        "verdict": verdict,
        "control_anomalous": control_mean is not None
        and abs(control_mean) >= CONTROL_ALERT,
        "confirmatory_run_authorized_by_gate": authorized,
        "gate_status": {
            str(seed): str(item.get("status", "unknown"))
            for seed, item in gate_status.items()
        },
    }


def build_report(payload: Mapping[str, Any]) -> str:
    labels = {
        "in_band_helps": "面積限回真實分布**有幫助**",
        "in_band_harms": "面積限回真實分布**有害**",
        "no_effect": "**無效果**",
        "uninformative": "**未能檢驗**（判定用的 seed 都無鑑別力）",
    }
    lines = [
        "# v5 三 seed 複跑的預註冊判定",
        "",
        (
            "本報告只執行 [ADR-040](../docs/decisions.md#adr-040) 在**執行前**寫死的規則。"
            "數字由 `scripts/decide_v5_seed_replication.py` 產生。"
        ),
        "",
        (
            f"主指標 `{payload['metric']}`　主要物件 `{payload['primary_object']}`　"
            f"對照物件 `{payload['control_object']}`"
        ),
        "",
        (
            f"**判定只計入 seed {', '.join(str(s) for s in payload['verdict_seeds'])}**。"
            f"seed {payload['disclosed_seed']} 的數值在 ADR-039 已被引用過，"
            "因此照常報告但不進入任何判定式——排除的是唯一對本分析不盲的 seed。"
        ),
        "",
    ]
    for label, key in (("主要物件", "primary"), ("對照物件", "control")):
        name = payload["primary_object"] if key == "primary" else payload["control_object"]
        lines += [
            f"## {label}：`{name}`",
            "",
            "| Seed | real_only | db_current | db_inband | D | 計入判定 | 指標可分辨 |",
            "|---:|---:|---:|---:|---:|:---:|:---:|",
        ]
        for seed in sorted(payload[key], key=int):
            item = payload[key][seed]
            scores = item["scores"]
            counted = int(seed) in payload["verdict_seeds"]
            lines.append(
                f"| {seed} | {scores[BASELINE]:.4f} | {scores[CURRENT_ARM]:.4f} "
                f"| {scores[INBAND_ARM]:.4f} | {item['delta']:+.4f} "
                f"| {'是' if counted else '**否**'} "
                f"| {'是' if item['metric_discriminates'] else '**否**'} |"
            )
        mean = payload["primary_mean_delta" if key == "primary" else "control_mean_delta"]
        lines += [
            "",
            (
                f"判定用平均 `D̄ = {mean:+.4f}`"
                if mean is not None
                else "判定用平均：**無可用 seed**"
            ),
            "",
        ]
    lines += [
        f"**判定：{labels[payload['verdict']]}**"
        + (
            f"（門檻 `±{EFFECT_THRESHOLD}`，沿用 ADR-026 的 per-object AUROC 容差原值）"
            if payload["verdict"] != "uninformative"
            else ""
        ),
        "",
    ]
    control_mean = payload["control_mean_delta"]
    if control_mean is not None:
        state = "**異常**" if payload["control_anomalous"] else "正常"
        lines += [
            (
                f"對照組 `{payload['control_object']}`：`D̄ = {control_mean:+.4f}`，"
                f"門檻 `{CONTROL_ALERT}`，判定{state}。"
            ),
            "",
        ]
        if payload["control_anomalous"]:
            lines += [
                "對照組本應幾乎不動，既然明顯變動，本次結論可信度下降。",
                "",
            ]
    lines += [
        "## Confirmatory gate",
        "",
        *(
            f"- seed {seed}：`{status}`"
            for seed, status in sorted(payload["gate_status"].items())
        ),
        (
            "- 是否授權 confirmatory test："
            f"**{'是' if payload['confirmatory_run_authorized_by_gate'] else '否'}**"
        ),
        "",
        (
            "判定結論成立**不等於** gate 通過。gate 未過時一律不得讀 frozen test、"
            "不得跑 confirmatory run（ADR-040）。"
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
        "--seed42",
        type=Path,
        default=Path("results/v4/pilot_classification.json"),
    )
    parser.add_argument(
        "--seed43",
        type=Path,
        default=Path("results/v5/pilot_classification_seed43.json"),
    )
    parser.add_argument(
        "--seed44",
        type=Path,
        default=Path("results/v5/pilot_classification_seed44.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v5_seed_replication.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/v5_seed_replication.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = {42: args.seed42, 43: args.seed43, 44: args.seed44}
    seeds = {seed: load_seed(path, seed) for seed, path in sources.items()}
    gate_status = {
        seed: json.loads(path.read_text(encoding="utf-8")).get("gate", {})
        for seed, path in sources.items()
    }
    payload = decide(seeds, gate_status)
    atomic_write(args.report, build_report(payload))
    atomic_write(
        args.output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(payload["verdict"], ensure_ascii=False))
    print(f"primary_mean_delta={payload['primary_mean_delta']}")
    print(f"confirmatory_authorized={payload['confirmatory_run_authorized_by_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
