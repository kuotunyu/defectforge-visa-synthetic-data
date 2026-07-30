"""Check the replicated seed-42 runs against the results published before ADR-032.

The ADR-032 replication re-executed seed 42 on a different Colab machine at a different
time, because Drive held no prior `runs/` tree for the loop to skip. That accident is free
evidence, so it is verified rather than discarded: every seed-42 row in the current
`results/segmentation.csv` is compared against `reports/segmentation_seed42_baseline.csv`,
the exact table published before the replication.

The script reports what it measures. It never rewrites either side, and a mismatch is a
finding to be published, not an error to be silenced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from scripts.aggregate_segmentation import ANCHOR_SEED  # isort: skip
from src.common.integrity import write_text_lf  # isort: skip

COMPARED_METRICS = ("dice", "miou", "pixel_auroc", "aupro")
KEY_COLUMNS = ("logical_group", "object")


class ReproductionCheckError(RuntimeError):
    """Raised when the two tables cannot be compared at all."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReproductionCheckError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _anchor_rows(path: Path) -> dict[tuple[str, str], Mapping[str, str]]:
    require(path.is_file(), f"Missing segmentation table: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"Segmentation table is empty: {path}")
    selected: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        if int(row["seed"]) != ANCHOR_SEED:
            continue
        key = (row["logical_group"], row["object"])
        require(key not in selected, f"Duplicate anchor row in {path.name}: {key}")
        selected[key] = row
    require(bool(selected), f"No seed-{ANCHOR_SEED} rows in {path}")
    return selected


def compare(baseline_path: Path, current_path: Path) -> dict[str, Any]:
    baseline = _anchor_rows(baseline_path)
    current = _anchor_rows(current_path)
    require(
        set(baseline) == set(current),
        "Baseline and current tables cover different groups at the anchor seed",
    )
    physical: list[dict[str, Any]] = []
    for key in sorted(baseline):
        before, after = baseline[key], current[key]
        if before["physical_run"] != "true":
            continue
        deltas = {
            metric: float(after[metric]) - float(before[metric])
            for metric in COMPARED_METRICS
        }
        for metric, value in deltas.items():
            require(math.isfinite(value), f"Non-finite delta: {key}/{metric}")
        physical.append(
            {
                "logical_group": key[0],
                "object": key[1],
                "run_name": after["run_name"],
                "model_sha256_matches": before["model_sha256"] == after["model_sha256"],
                "run_signature_matches": before["run_signature"] == after["run_signature"],
                "deltas": deltas,
            }
        )
    matching_models = [item for item in physical if item["model_sha256_matches"]]
    max_abs_delta = {
        metric: max((abs(item["deltas"][metric]) for item in physical), default=0.0)
        for metric in COMPARED_METRICS
    }
    identical = len(matching_models) == len(physical) and all(
        value == 0.0 for value in max_abs_delta.values()
    )
    return {
        "status": "passed",
        "schema_version": 1,
        "anchor_seed": ANCHOR_SEED,
        "baseline_csv": baseline_path.as_posix(),
        "baseline_sha256": sha256_file(baseline_path),
        "current_csv": current_path.as_posix(),
        "current_sha256": sha256_file(current_path),
        "compared_physical_runs": len(physical),
        "model_sha256_matches": len(matching_models),
        "bit_identical": identical,
        "max_abs_metric_delta": max_abs_delta,
        "runs": physical,
    }


def build_report(payload: Mapping[str, Any]) -> str:
    compared = int(payload["compared_physical_runs"])
    matches = int(payload["model_sha256_matches"])
    verdict = (
        "**逐 bit 相同**"
        if payload["bit_identical"]
        else "**不完全相同——差異如下表，已如實保留**"
    )
    lines = [
        "# Seed 42 跨機器重現檢查",
        "",
        (
            "ADR-032 的複跑在另一台 Colab 機器、另一個時間重新執行了 seed 42，因為 Drive 上"
            "沒有先前的 `runs/` 樹可供跳過。這個意外是免費的證據，因此加以驗證而非丟棄。"
        ),
        "",
        f"- 比較的實跑 run 數：**{compared}**",
        f"- `model.safetensors` SHA256 相同的 run 數：**{matches} / {compared}**",
        f"- 判定：{verdict}",
        "",
        "| 指標 | 最大絕對差 |",
        "|---|---:|",
    ]
    for metric, value in payload["max_abs_metric_delta"].items():
        lines.append(f"| {metric} | `{float(value):.8f}` |")
    lines.extend(
        [
            "",
            "| 物件 | 組別 | model SHA256 相同 | run signature 相同 |",
            "|---|---|:---:|:---:|",
        ]
    )
    for item in payload["runs"]:
        lines.append(
            f"| {item['object']} | {item['logical_group']} | "
            f"{'是' if item['model_sha256_matches'] else '否'} | "
            f"{'是' if item['run_signature_matches'] else '否'} |"
        )
    lines.extend(
        [
            "",
            f"基準表：`{payload['baseline_csv']}`（SHA256 `{payload['baseline_sha256']}`）",
            "",
            f"目前表：`{payload['current_csv']}`（SHA256 `{payload['current_sha256']}`）",
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
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("reports/segmentation_seed42_baseline.csv"),
    )
    parser.add_argument("--segmentation", type=Path, default=Path("results/segmentation.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/seed42_reproduction.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/seed42_reproduction.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = compare(args.baseline, args.segmentation)
    atomic_write(args.report, build_report(payload))
    atomic_write(
        args.output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"bit_identical={payload['bit_identical']} "
        f"models={payload['model_sha256_matches']}/{payload['compared_physical_runs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
