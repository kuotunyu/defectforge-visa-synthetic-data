"""Apply the ADR-032 decision rules that were fixed before the replication ran.

Both rules below are transcribed from ADR-032 and must not be edited to fit an observed
result. The script reads only `results/segmentation.csv`, which M20 rebuilds from raw
`training_report.json` files, and emits a verdict payload plus a Markdown report.

Rule 1 (Dice/AUPRO direction conflict). For each object, take the per-seed
`filtered_syn - real_only` delta in Dice and in AUPRO. If, on at least one object, the two
deltas carry opposite signs in at least two of the three seeds, the conflict counts as a
real phenomenon; otherwise it was a single-seed artefact.

Rule 2 (`capsules/std_aug` collapse). If its Dice is exactly zero in at least two of the
three seeds the collapse counts as systematic; if only one seed is zero it counts as seed
noise, and ADR-031's claim about standard augmentation is withdrawn.
"""

from __future__ import annotations

import argparse
import csv
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

from scripts.aggregate_segmentation import SEEDS  # isort: skip
from scripts.build_phase2_figures import OBJECTS  # isort: skip
from src.common.integrity import write_text_lf  # isort: skip

CONFLICT_BASELINE = "real_only"
CONFLICT_TREATMENT = "filtered_syn"
COLLAPSE_OBJECT = "capsules"
COLLAPSE_GROUP = "std_aug"
# ADR-032 fixed both thresholds before the replication ran.
MIN_CONFLICTING_SEEDS = 2
MIN_ZERO_DICE_SEEDS = 2


class ReplicationDecisionError(RuntimeError):
    """Raised when the replication table cannot support the pre-registered rules."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplicationDecisionError(message)


def read_rows(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing segmentation CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"Segmentation CSV is empty: {path}")
    return rows


def _value(
    rows: Sequence[Mapping[str, str]],
    *,
    object_name: str,
    group_name: str,
    seed: int,
    metric: str,
) -> float:
    matches = [
        row
        for row in rows
        if row["object"] == object_name
        and row["logical_group"] == group_name
        and int(row["seed"]) == seed
    ]
    require(
        len(matches) == 1,
        f"Expected one row for {object_name}/{group_name}/seed{seed}, got {len(matches)}",
    )
    value = float(matches[0][metric])
    require(math.isfinite(value), f"Non-finite {metric}: {object_name}/{group_name}/{seed}")
    return value


def _opposite_signs(first: float, second: float) -> bool:
    """Zero is treated as agreeing with either sign, so a tie never creates a conflict."""
    return (first > 0.0 > second) or (second > 0.0 > first)


def direction_conflict(
    rows: Sequence[Mapping[str, str]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    per_object: dict[str, Any] = {}
    for object_name in OBJECTS:
        per_seed = []
        for seed in seeds:
            dice_delta = _value(
                rows,
                object_name=object_name,
                group_name=CONFLICT_TREATMENT,
                seed=seed,
                metric="dice",
            ) - _value(
                rows,
                object_name=object_name,
                group_name=CONFLICT_BASELINE,
                seed=seed,
                metric="dice",
            )
            aupro_delta = _value(
                rows,
                object_name=object_name,
                group_name=CONFLICT_TREATMENT,
                seed=seed,
                metric="aupro",
            ) - _value(
                rows,
                object_name=object_name,
                group_name=CONFLICT_BASELINE,
                seed=seed,
                metric="aupro",
            )
            per_seed.append(
                {
                    "seed": int(seed),
                    "dice_delta": dice_delta,
                    "aupro_delta": aupro_delta,
                    "opposite_signs": _opposite_signs(dice_delta, aupro_delta),
                }
            )
        conflicting = [item["seed"] for item in per_seed if item["opposite_signs"]]
        per_object[object_name] = {
            "per_seed": per_seed,
            "conflicting_seeds": conflicting,
            "meets_rule": len(conflicting) >= MIN_CONFLICTING_SEEDS,
            "dice_delta_mean": statistics.fmean(item["dice_delta"] for item in per_seed),
            "dice_delta_std": statistics.stdev([item["dice_delta"] for item in per_seed]),
            "aupro_delta_mean": statistics.fmean(item["aupro_delta"] for item in per_seed),
            "aupro_delta_std": statistics.stdev([item["aupro_delta"] for item in per_seed]),
        }
    triggering = sorted(name for name, item in per_object.items() if item["meets_rule"])
    return {
        "rule": (
            f"opposite signs in >= {MIN_CONFLICTING_SEEDS} of {len(seeds)} seeds "
            "on at least one object"
        ),
        "baseline": CONFLICT_BASELINE,
        "treatment": CONFLICT_TREATMENT,
        "objects": per_object,
        "triggering_objects": triggering,
        "verdict": "real_phenomenon" if triggering else "single_seed_artefact",
    }


def augmentation_collapse(
    rows: Sequence[Mapping[str, str]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    per_seed = [
        {
            "seed": int(seed),
            "dice": _value(
                rows,
                object_name=COLLAPSE_OBJECT,
                group_name=COLLAPSE_GROUP,
                seed=seed,
                metric="dice",
            ),
            "pixel_auroc": _value(
                rows,
                object_name=COLLAPSE_OBJECT,
                group_name=COLLAPSE_GROUP,
                seed=seed,
                metric="pixel_auroc",
            ),
        }
        for seed in seeds
    ]
    zero_seeds = [item["seed"] for item in per_seed if item["dice"] == 0.0]
    systematic = len(zero_seeds) >= MIN_ZERO_DICE_SEEDS
    return {
        "rule": f"Dice == 0 in >= {MIN_ZERO_DICE_SEEDS} of {len(seeds)} seeds",
        "object": COLLAPSE_OBJECT,
        "group": COLLAPSE_GROUP,
        "per_seed": per_seed,
        "zero_dice_seeds": zero_seeds,
        "verdict": "systematic" if systematic else "seed_noise",
        "adr_031_withdrawn": not systematic,
    }


def decide(rows: Sequence[Mapping[str, str]], seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in seeds)
    observed = sorted({int(row["seed"]) for row in rows})
    require(observed == sorted(seeds), f"Segmentation CSV seeds are {observed}, expected {list(seeds)}")
    return {
        "status": "passed",
        "schema_version": 1,
        "source": "results/segmentation.csv",
        "preregistered_in": "ADR-032",
        "seeds": list(seeds),
        "direction_conflict": direction_conflict(rows, seeds),
        "augmentation_collapse": augmentation_collapse(rows, seeds),
    }


def _verdict_label(payload: Mapping[str, Any]) -> str:
    return {
        "real_phenomenon": "真實現象",
        "single_seed_artefact": "單 seed 假象",
        "systematic": "系統性",
        "seed_noise": "seed 雜訊",
    }[str(payload["verdict"])]


def build_report(payload: Mapping[str, Any]) -> str:
    conflict = payload["direction_conflict"]
    collapse = payload["augmentation_collapse"]
    seeds = payload["seeds"]
    lines = [
        "# 分割 3-seed 複跑的預註冊判定",
        "",
        (
            "本報告只執行 [ADR-032](../docs/decisions.md#adr-032) 在**複跑開始前**就寫死的兩條"
            "規則。數字全部由 `scripts/decide_segmentation_replication.py` 從 "
            "`results/segmentation.csv` 產生，該表由 M20 聚合器從 raw `training_report.json` 重建。"
        ),
        "",
        f"Seeds：{', '.join(str(seed) for seed in seeds)}",
        "",
        "## 規則 1 — Dice／AUPRO 方向矛盾",
        "",
        (
            f"判準：`{conflict['treatment']} − {conflict['baseline']}` 的 Dice 差與 AUPRO 差"
            f"符號相反，出現在 {len(seeds)} 個 seed 中的 ≥ {MIN_CONFLICTING_SEEDS} 個，"
            "且至少發生在一個物件上。"
        ),
        "",
        "| 物件 | Seed | Dice Δ | AUPRO Δ | 符號相反 |",
        "|---|---:|---:|---:|:---:|",
    ]
    for object_name, item in conflict["objects"].items():
        for entry in item["per_seed"]:
            mark = "是" if entry["opposite_signs"] else "否"
            lines.append(
                f"| {object_name} | {entry['seed']} | {entry['dice_delta']:+.4f} | "
                f"{entry['aupro_delta']:+.4f} | {mark} |"
            )
    lines.extend(["", "| 物件 | 符號相反的 seed | 達標 | Dice Δ（mean ± std） | AUPRO Δ（mean ± std） |", "|---|---|:---:|---:|---:|"])
    for object_name, item in conflict["objects"].items():
        seeds_text = ", ".join(str(seed) for seed in item["conflicting_seeds"]) or "—"
        lines.append(
            f"| {object_name} | {seeds_text} | {'是' if item['meets_rule'] else '否'} | "
            f"{item['dice_delta_mean']:+.4f} ± {item['dice_delta_std']:.4f} | "
            f"{item['aupro_delta_mean']:+.4f} ± {item['aupro_delta_std']:.4f} |"
        )
    triggering = ", ".join(conflict["triggering_objects"]) or "無"
    lines.extend(
        [
            "",
            f"**判定：{_verdict_label(conflict)}**（達標物件：{triggering}）",
            "",
            "## 規則 2 — `capsules/std_aug` 崩潰",
            "",
            f"判準：Dice 在 {len(seeds)} 個 seed 中的 ≥ {MIN_ZERO_DICE_SEEDS} 個為 `0.0000`。",
            "",
            "| Seed | Dice | Pixel AUROC |",
            "|---:|---:|---:|",
        ]
    )
    for entry in collapse["per_seed"]:
        lines.append(f"| {entry['seed']} | {entry['dice']:.4f} | {entry['pixel_auroc']:.4f} |")
    zero_text = ", ".join(str(seed) for seed in collapse["zero_dice_seeds"]) or "無"
    lines.extend(
        [
            "",
            f"Dice 為零的 seed：{zero_text}",
            "",
            f"**判定：{_verdict_label(collapse)}**",
            "",
            (
                "ADR-031 的主張**撤回**。"
                if collapse["adr_031_withdrawn"]
                else "ADR-031 的主張維持不變。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_text_lf(temporary, text)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", type=Path, default=Path("results/segmentation.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/segmentation_replication.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/segmentation_replication.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = decide(read_rows(args.segmentation))
    atomic_write(args.report, build_report(payload))
    atomic_write(
        args.output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(payload["direction_conflict"]["verdict"], ensure_ascii=False))
    print(json.dumps(payload["augmentation_collapse"]["verdict"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
