"""Rebuild and verify every numeric README result block from raw result CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aggregate_segmentation import ANCHOR_SEED, LOGICAL_GROUPS, SEEDS  # isort: skip
from scripts.build_phase2_figures import MAIN_GROUPS, OBJECTS  # isort: skip
from scripts.decide_segmentation_replication import decide  # isort: skip
from scripts.run_classifier_matrix import matrix_plan  # isort: skip
from src.common.integrity import write_text_lf  # isort: skip

BLOCK_NAMES = (
    "CLASSIFICATION_MAIN",
    "CLASSIFICATION_SEED_VARIANCE",
    "CLASSIFICATION_LEAKAGE_SURFACE",
    "SEGMENTATION_MAIN",
    "SEGMENTATION_SEED_VARIANCE",
    "SEGMENTATION_REPRODUCTION",
    "SEGMENTATION_THRESHOLD",
    "SEGMENTATION_REPLICATION",
    "RESULT_OUTCOME",
)
# docs/experiment_protocol.md requires Real-only and the best Filtered group to be
# replicated across three seeds and reported as mean +/- std.
MIN_REPLICATED_SEEDS = 3
# A run whose thresholded Dice is exactly zero predicted background everywhere; a high
# pixel AUROC on such a run means the probability map was informative and the fixed
# threshold, not the model, produced the empty mask.
INFORMATIVE_PIXEL_AUROC = 0.80
CLASSIFICATION_METRICS = (
    "macro_f1",
    "anomaly_f1",
    "auroc",
    "normal_false_positive_rate",
)
SEGMENTATION_METRICS = ("dice", "miou", "pixel_auroc", "aupro")
GROUP_LABELS = {
    "real_only": "Real-only（10 張）",
    "std_aug": "+ Standard Augmentation",
    "unfiltered_syn": "+ 未篩選 Synthetic Data",
    "filtered_syn": "+ 已篩選 Synthetic Data",
    "full_real": "Full-real（60 張）",
    "procedural_only": "僅 Procedural",
    "copypaste_only": "僅 Copy-paste",
    "diffusion_only": "僅 Diffusion",
    "all_mixed": "All-mixed（與已篩選 Synthetic Data 共用）",
}


class ReadmeVerificationError(RuntimeError):
    """Raised when README results cannot be reproduced exactly."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadmeVerificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    require(path.is_file(), f"Missing result CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    require(bool(rows), f"Result CSV is empty: {path}")
    return fieldnames, rows


def _finite_unit_interval(row: Mapping[str, str], metric: str, *, label: str) -> float:
    try:
        value = float(row[metric])
    except (KeyError, TypeError, ValueError) as error:
        raise ReadmeVerificationError(f"Invalid {label}/{metric}") from error
    require(math.isfinite(value), f"Non-finite {label}/{metric}")
    require(0.0 <= value <= 1.0, f"Out-of-range {label}/{metric}: {value}")
    return value


def validate_classification_rows(
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    required = {
        "requested_group",
        "canonical_group",
        "object",
        "seed",
        *CLASSIFICATION_METRICS,
    }
    require(required <= set(fieldnames), "classification.csv schema is incomplete")
    expected = {
        (spec.group, spec.object_name, spec.seed)
        for spec in matrix_plan()
    }
    observed: set[tuple[str, str, int]] = set()
    for row in rows:
        key = (row["requested_group"], row["object"], int(row["seed"]))
        require(key not in observed, f"Duplicate classification result: {key}")
        observed.add(key)
        for metric in CLASSIFICATION_METRICS:
            _finite_unit_interval(row, metric, label=f"classification/{key}")
    require(observed == expected, "classification.csv does not match the frozen 38-run matrix")


def validate_segmentation_rows(
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    required = {"logical_group", "canonical_group", "object", "seed", *SEGMENTATION_METRICS}
    require(required <= set(fieldnames), "segmentation.csv schema is incomplete")
    # ADR-032 replicated all eight formal groups across three seeds, so every logical group
    # must now appear once per object per seed.
    expected = {
        (group_name, object_name, seed)
        for group_name in LOGICAL_GROUPS
        for object_name in OBJECTS
        for seed in SEEDS
    }
    observed: set[tuple[str, str, int]] = set()
    for row in rows:
        key = (row["logical_group"], row["object"], int(row["seed"]))
        require(key not in observed, f"Duplicate segmentation result: {key}")
        observed.add(key)
        if row["logical_group"] == "all_mixed":
            require(
                row["canonical_group"] == "filtered_syn",
                "all_mixed must cite filtered_syn",
            )
        for metric in SEGMENTATION_METRICS:
            _finite_unit_interval(row, metric, label=f"segmentation/{key}")
    require(
        observed == expected,
        f"segmentation.csv does not match the {len(expected)} logical M20 rows",
    )


def _one(
    rows: Sequence[Mapping[str, str]],
    *,
    key: str,
    value: str,
    object_name: str,
    seed: int | None = None,
) -> Mapping[str, str]:
    matches = [
        row
        for row in rows
        if row[key] == value
        and row["object"] == object_name
        and (seed is None or int(row["seed"]) == seed)
    ]
    require(len(matches) == 1, f"Expected one {key}={value}/{object_name}, got {len(matches)}")
    return matches[0]


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def classification_block(rows: Sequence[Mapping[str, str]]) -> str:
    output: list[list[str]] = []
    for object_name in OBJECTS:
        for group_name in MAIN_GROUPS:
            row = _one(
                rows,
                key="canonical_group",
                value=group_name,
                object_name=object_name,
                seed=42,
            )
            output.append(
                [
                    object_name,
                    GROUP_LABELS[group_name],
                    f"{float(row['macro_f1']):.4f}",
                    f"{float(row['anomaly_f1']):.4f}",
                    f"{float(row['auroc']):.4f}",
                    f"{float(row['normal_false_positive_rate']):.4f}",
                ]
            )
    return _markdown_table(
        ("物件", "訓練組別", "Macro-F1", "瑕疵 F1", "AUROC", "正常樣本 FPR"),
        output,
    )


def _group_order(group_name: str) -> tuple[int, str]:
    if group_name in MAIN_GROUPS:
        return (MAIN_GROUPS.index(group_name), group_name)
    return (len(MAIN_GROUPS), group_name)


def seed_variance_block(rows: Sequence[Mapping[str, str]]) -> str:
    """Report mean +/- std for every group that reached the replicated-seed policy."""
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["object"], row["canonical_group"])].append(row)
    replicated = sorted(
        (key for key, items in grouped.items() if len(items) >= MIN_REPLICATED_SEEDS),
        key=lambda key: (OBJECTS.index(key[0]), _group_order(key[1])),
    )
    require(
        bool(replicated),
        f"No classification group reaches {MIN_REPLICATED_SEEDS} seeds",
    )
    output: list[list[str]] = []
    for object_name, group_name in replicated:
        items = grouped[(object_name, group_name)]
        seeds = [int(item["seed"]) for item in items]
        require(
            len(set(seeds)) == len(seeds),
            f"Duplicate seeds for {object_name}/{group_name}",
        )
        macro = [float(item["macro_f1"]) for item in items]
        auroc = [float(item["auroc"]) for item in items]
        output.append(
            [
                object_name,
                GROUP_LABELS.get(group_name, group_name),
                str(len(items)),
                f"{statistics.fmean(macro):.4f} ± {statistics.stdev(macro):.4f}",
                f"{statistics.fmean(auroc):.4f} ± {statistics.stdev(auroc):.4f}",
            ]
        )
    return _markdown_table(
        ("物件", "訓練組別", "Seeds", "Macro-F1（mean ± std）", "AUROC（mean ± std）"),
        output,
    )


def leakage_surface_block(rows: Sequence[Mapping[str, str]]) -> str:
    """Contrast procedural synthesis with and without the disclosed real-mask statistics.

    ADR-011 disclosed that procedural masks are constrained to the area and aspect-ratio
    percentiles of the ten few-shot training masks, and promised the `--no-real-stats`
    control be reported beside it. This renders that promise for classification.
    """
    output: list[list[str]] = []
    for object_name in OBJECTS:
        with_stats = _one(
            rows,
            key="canonical_group",
            value="src_procedural",
            object_name=object_name,
            seed=42,
        )
        without_stats = _one(
            rows,
            key="canonical_group",
            value="procedural_norealstats",
            object_name=object_name,
            seed=42,
        )
        output.append(
            [
                object_name,
                f"{float(with_stats['macro_f1']):.4f}",
                f"{float(without_stats['macro_f1']):.4f}",
                f"{float(with_stats['macro_f1']) - float(without_stats['macro_f1']):+.4f}",
                f"{float(with_stats['auroc']):.4f}",
                f"{float(without_stats['auroc']):.4f}",
                f"{float(with_stats['auroc']) - float(without_stats['auroc']):+.4f}",
            ]
        )
    return _markdown_table(
        (
            "物件",
            "Macro-F1（用統計量）",
            "Macro-F1（不用）",
            "Δ",
            "AUROC（用統計量）",
            "AUROC（不用）",
            "Δ",
        ),
        output,
    )


def threshold_block(rows: Sequence[Mapping[str, str]]) -> str:
    """Contrast the thresholded Dice with the threshold-free AUPRO for the main groups."""
    output: list[list[str]] = []
    for object_name in OBJECTS:
        baseline = _one(
            rows,
            key="logical_group",
            value="real_only",
            object_name=object_name,
            seed=ANCHOR_SEED,
        )
        base_dice = float(baseline["dice"])
        base_aupro = float(baseline["aupro"])
        for group_name in MAIN_GROUPS:
            row = _one(
                rows,
                key="logical_group",
                value=group_name,
                object_name=object_name,
                seed=ANCHOR_SEED,
            )
            dice = float(row["dice"])
            aupro = float(row["aupro"])
            if group_name == "real_only":
                dice_delta = aupro_delta = "—"
            else:
                dice_delta = f"{dice - base_dice:+.4f}"
                aupro_delta = f"{aupro - base_aupro:+.4f}"
            output.append(
                [
                    object_name,
                    GROUP_LABELS[group_name],
                    f"{dice:.4f}",
                    f"{aupro:.4f}",
                    dice_delta,
                    aupro_delta,
                ]
            )
    return _markdown_table(
        (
            "物件",
            "訓練組別",
            "Dice（threshold 0.5）",
            "AUPRO（不依賴 threshold）",
            "Dice Δ vs Real-only",
            "AUPRO Δ vs Real-only",
        ),
        output,
    )


def segmentation_block(rows: Sequence[Mapping[str, str]]) -> str:
    """Keep the pre-registered anchor seed as the main table, as ADR-032 requires."""
    output: list[list[str]] = []
    for object_name in OBJECTS:
        for group_name in LOGICAL_GROUPS:
            row = _one(
                rows,
                key="logical_group",
                value=group_name,
                object_name=object_name,
                seed=ANCHOR_SEED,
            )
            output.append(
                [
                    object_name,
                    GROUP_LABELS[group_name],
                    f"{float(row['dice']):.4f}",
                    f"{float(row['miou']):.4f}",
                    f"{float(row['pixel_auroc']):.4f}",
                    f"{float(row['aupro']):.4f}",
                ]
            )
    return _markdown_table(
        ("物件", "訓練組別", "Dice", "mIoU", "Pixel AUROC", "AUPRO"),
        output,
    )


def segmentation_seed_variance_block(rows: Sequence[Mapping[str, str]]) -> str:
    """Report mean +/- std over the ADR-032 replication for every logical group."""
    output: list[list[str]] = []
    for object_name in OBJECTS:
        for group_name in LOGICAL_GROUPS:
            items = [
                row
                for row in rows
                if row["logical_group"] == group_name and row["object"] == object_name
            ]
            seeds = [int(item["seed"]) for item in items]
            require(
                sorted(seeds) == sorted(SEEDS),
                f"Segmentation seeds are incomplete: {object_name}/{group_name}",
            )
            require(
                len(items) >= MIN_REPLICATED_SEEDS,
                f"Segmentation group below the replication policy: {object_name}/{group_name}",
            )
            cells = [
                (
                    f"{statistics.fmean(float(item[metric]) for item in items):.4f} ± "
                    f"{statistics.stdev([float(item[metric]) for item in items]):.4f}"
                )
                for metric in ("dice", "aupro")
            ]
            output.append([object_name, GROUP_LABELS[group_name], str(len(items)), *cells])
    return _markdown_table(
        ("物件", "訓練組別", "Seeds", "Dice（mean ± std）", "AUPRO（mean ± std）"),
        output,
    )


def segmentation_reproduction_block(payload: Mapping[str, Any]) -> str:
    """Render the cross-machine seed-42 reproduction from its own verified report."""
    compared = int(payload["compared_physical_runs"])
    matches = int(payload["model_sha256_matches"])
    anchor_seed = int(payload["anchor_seed"])
    deltas = payload["max_abs_metric_delta"]
    require(
        set(deltas) >= set(SEGMENTATION_METRICS),
        "Reproduction report is missing a compared metric",
    )
    verdict = (
        "**逐 bit 相同**"
        if payload["bit_identical"]
        else "**不完全相同——差異已保留在報告中**"
    )
    metric_cells = "、".join(
        f"{metric} `{float(deltas[metric]):.8f}`" for metric in SEGMENTATION_METRICS
    )
    return "\n".join(
        [
            (
                f"- 重新執行 seed {anchor_seed} 的實跑 run：**{compared}** 個"
                f"（{len(OBJECTS)} 個物件 × {compared // len(OBJECTS)} 組）。"
            ),
            (
                f"- `model.safetensors` SHA256 與已發佈值相同者：**{matches} / {compared}**。"
                f"判定：{verdict}。"
            ),
            f"- 四項指標的最大絕對差：{metric_cells}。",
            (
                "- 基準是複跑前已發佈的表格 `reports/segmentation_seed42_baseline.csv`；"
                "比對由 `scripts/verify_seed42_reproduction.py` 執行，"
                "逐 run 結果見[重現檢查報告](reports/seed42_reproduction.md)。"
            ),
        ]
    )


def segmentation_replication_block(rows: Sequence[Mapping[str, str]]) -> str:
    """Render the ADR-032 verdicts from the same implementation that decides them."""
    payload = decide(rows)
    conflict = payload["direction_conflict"]
    collapse = payload["augmentation_collapse"]
    conflict_verdict = (
        "**真實現象**"
        if conflict["verdict"] == "real_phenomenon"
        else "**單 seed 假象**"
    )
    collapse_verdict = (
        "**系統性**" if collapse["verdict"] == "systematic" else "**seed 雜訊**"
    )
    triggering = "、".join(conflict["triggering_objects"]) or "無"
    zero_seeds = "、".join(str(seed) for seed in collapse["zero_dice_seeds"]) or "無"
    conflict_rows = [
        [
            object_name,
            ", ".join(str(seed) for seed in item["conflicting_seeds"]) or "—",
            f"{item['dice_delta_mean']:+.4f} ± {item['dice_delta_std']:.4f}",
            f"{item['aupro_delta_mean']:+.4f} ± {item['aupro_delta_std']:.4f}",
            "是" if item["meets_rule"] else "否",
        ]
        for object_name, item in conflict["objects"].items()
    ]
    table = _markdown_table(
        (
            "物件",
            "Dice／AUPRO 符號相反的 seed",
            "Dice Δ（mean ± std）",
            "AUPRO Δ（mean ± std）",
            "達預註冊門檻",
        ),
        conflict_rows,
    )
    return "\n".join(
        [
            table,
            "",
            (
                f"- 規則 1（方向矛盾）判定：{conflict_verdict}。"
                f"門檻是「至少一個物件上、{len(SEEDS)} 個 seed 中 ≥2 個符號相反」，"
                f"達標物件：{triggering}。"
            ),
            (
                f"- 規則 2（`capsules/std_aug` 崩潰）判定：{collapse_verdict}。"
                f"Dice 為零的 seed：{zero_seeds}。"
                + (
                    "因此 ADR-031 的主張撤回。"
                    if collapse["adr_031_withdrawn"]
                    else "因此 ADR-031 的主張維持不變。"
                )
            ),
            (
                "- 兩條規則都在 [ADR-032](docs/decisions.md#adr-032) 於**複跑執行前**寫死，"
                "看到結果後未作任何修改。"
            ),
        ]
    )


def _mean_metric(
    rows: Sequence[Mapping[str, str]],
    *,
    key: str,
    group_name: str,
    metric: str,
    seed: int | None = None,
) -> float:
    values = [
        float(
            _one(
                rows,
                key=key,
                value=group_name,
                object_name=object_name,
                seed=seed,
            )[metric]
        )
        for object_name in OBJECTS
    ]
    return sum(values) / len(values)


def _macro_delta(rows: Sequence[Mapping[str, str]], metric: str, seed: int) -> float:
    """Two-object macro mean of `filtered_syn - real_only` at one seed."""
    return _mean_metric(
        rows,
        key="logical_group",
        group_name="filtered_syn",
        metric=metric,
        seed=seed,
    ) - _mean_metric(
        rows,
        key="logical_group",
        group_name="real_only",
        metric=metric,
        seed=seed,
    )


def outcome_payload(
    classification_rows: Sequence[Mapping[str, str]],
    segmentation_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    classification_delta = _mean_metric(
        classification_rows,
        key="canonical_group",
        group_name="filtered_syn",
        metric="macro_f1",
        seed=42,
    ) - _mean_metric(
        classification_rows,
        key="canonical_group",
        group_name="real_only",
        metric="macro_f1",
        seed=42,
    )
    segmentation_delta = _macro_delta(segmentation_rows, "dice", ANCHOR_SEED)
    aupro_delta = _macro_delta(segmentation_rows, "aupro", ANCHOR_SEED)
    dice_by_seed = [_macro_delta(segmentation_rows, "dice", seed) for seed in SEEDS]
    aupro_by_seed = [_macro_delta(segmentation_rows, "aupro", seed) for seed in SEEDS]
    dice_seed_mean = statistics.fmean(dice_by_seed)
    aupro_seed_mean = statistics.fmean(aupro_by_seed)
    # all_mixed is an alias of filtered_syn, so it must not be counted as a second run.
    physical = [row for row in segmentation_rows if row["logical_group"] != "all_mixed"]
    zero_dice = [row for row in physical if float(row["dice"]) == 0.0]
    informative = [
        row for row in zero_dice if float(row["pixel_auroc"]) >= INFORMATIVE_PIXEL_AUROC
    ]
    replication = decide(segmentation_rows)
    return {
        "classification_macro_f1_delta": classification_delta,
        "segmentation_dice_delta": segmentation_delta,
        "classification_negative": classification_delta <= 0.0,
        "segmentation_negative": segmentation_delta <= 0.0,
        "segmentation_aupro_delta": aupro_delta,
        "segmentation_aupro_negative": aupro_delta <= 0.0,
        "segmentation_metric_directions_agree": (segmentation_delta <= 0.0)
        == (aupro_delta <= 0.0),
        "segmentation_anchor_seed": ANCHOR_SEED,
        "segmentation_seeds": list(SEEDS),
        "segmentation_dice_delta_seed_mean": dice_seed_mean,
        "segmentation_dice_delta_seed_std": statistics.stdev(dice_by_seed),
        "segmentation_aupro_delta_seed_mean": aupro_seed_mean,
        "segmentation_aupro_delta_seed_std": statistics.stdev(aupro_by_seed),
        "segmentation_seed_mean_directions_agree": (dice_seed_mean <= 0.0)
        == (aupro_seed_mean <= 0.0),
        "segmentation_direction_conflict_verdict": replication["direction_conflict"]["verdict"],
        "segmentation_direction_conflict_objects": replication["direction_conflict"][
            "triggering_objects"
        ],
        "segmentation_std_aug_collapse_verdict": replication["augmentation_collapse"][
            "verdict"
        ],
        "segmentation_physical_runs": len(physical),
        "segmentation_zero_dice_runs": len(zero_dice),
        "segmentation_zero_dice_informative_runs": len(informative),
        "segmentation_zero_dice_max_pixel_auroc": (
            max(float(row["pixel_auroc"]) for row in zero_dice) if zero_dice else 0.0
        ),
    }


def outcome_block(payload: Mapping[str, Any]) -> str:
    classification_delta = float(payload["classification_macro_f1_delta"])
    segmentation_delta = float(payload["segmentation_dice_delta"])
    classification_statement = (
        "是——已篩選 Synthetic Data 未提升平均 Macro-F1。"
        if payload["classification_negative"]
        else "否——已篩選 Synthetic Data 提升了平均 Macro-F1。"
    )
    segmentation_statement = (
        "是——已篩選 Synthetic Data 未提升平均 Dice。"
        if payload["segmentation_negative"]
        else "否——已篩選 Synthetic Data 提升了平均 Dice。"
    )
    aupro_delta = float(payload["segmentation_aupro_delta"])
    physical_runs = int(payload["segmentation_physical_runs"])
    zero_dice_runs = int(payload["segmentation_zero_dice_runs"])
    informative_runs = int(payload["segmentation_zero_dice_informative_runs"])
    max_zero_dice_auroc = float(payload["segmentation_zero_dice_max_pixel_auroc"])
    anchor_seed = int(payload["segmentation_anchor_seed"])
    seeds = payload["segmentation_seeds"]
    dice_seed_mean = float(payload["segmentation_dice_delta_seed_mean"])
    dice_seed_std = float(payload["segmentation_dice_delta_seed_std"])
    aupro_seed_mean = float(payload["segmentation_aupro_delta_seed_mean"])
    aupro_seed_std = float(payload["segmentation_aupro_delta_seed_std"])
    directions = (
        "方向一致"
        if payload["segmentation_metric_directions_agree"]
        else "**方向相反**"
    )
    seed_mean_directions = (
        "方向一致"
        if payload["segmentation_seed_mean_directions_agree"]
        else "**方向相反**"
    )
    conflict_objects = "、".join(payload["segmentation_direction_conflict_objects"]) or "無"
    conflict_verdict = (
        "真實現象"
        if payload["segmentation_direction_conflict_verdict"] == "real_phenomenon"
        else "單 seed 假象"
    )
    collapse_verdict = (
        "系統性"
        if payload["segmentation_std_aug_collapse_verdict"] == "systematic"
        else "seed 雜訊"
    )
    return "\n".join(
        [
            (
                f"- Classification：已篩選 Synthetic Data 相對 Real-only 的平均 Macro-F1 "
                f"差異為 `{classification_delta:+.4f}`。"
            ),
            (
                f"- Segmentation：已篩選 Synthetic Data 相對 Real-only 的平均 Dice "
                f"差異為 `{segmentation_delta:+.4f}`（seed {anchor_seed} 錨點）。"
            ),
            f"- Classification 負面結果：**{classification_statement}**",
            f"- Segmentation 負面結果：**{segmentation_statement}**",
            (
                f"- Segmentation（threshold-free）：seed {anchor_seed} 的平均 AUPRO 差異為 "
                f"`{aupro_delta:+.4f}`，與 Dice {directions}。"
            ),
            (
                f"- **複跑後這個 AUPRO 提升沒有重現。**{len(seeds)} 個 seed"
                f"（{', '.join(str(seed) for seed in seeds)}）的兩物件平均："
                f"Dice `{dice_seed_mean:+.4f} ± {dice_seed_std:.4f}`、"
                f"AUPRO `{aupro_seed_mean:+.4f} ± {aupro_seed_std:.4f}`，兩者{seed_mean_directions}。"
                f"seed {anchor_seed} 單獨呈現的 AUPRO 正向差異是該 seed 的特例。"
            ),
            (
                f"- 依 ADR-032 **執行前寫死**的規則判定：Dice／AUPRO 方向矛盾為"
                f"**{conflict_verdict}**（達標物件：{conflict_objects}）；"
                f"`capsules/std_aug` 的 Dice 崩潰為**{collapse_verdict}**。"
            ),
            (
                f"- {physical_runs} 個實跑的 Segmentation run 中有 {zero_dice_runs} 個在固定 "
                f"threshold 0.5 下 Dice = 0（整張預測為背景），其中 {informative_runs} 個的 "
                f"pixel AUROC 仍達 {INFORMATIVE_PIXEL_AUROC:.2f} 以上"
                f"（最高 `{max_zero_dice_auroc:.4f}`）。"
            ),
            (
                "- 主結論仍以預註冊的 Macro-F1 與 Dice 為準；AUPRO 與 threshold 敏感度是"
                "**併列揭露**，不是事後換指標。"
            ),
        ]
    )


def render_blocks(
    classification_rows: Sequence[Mapping[str, str]],
    segmentation_rows: Sequence[Mapping[str, str]],
    reproduction: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    outcome = outcome_payload(classification_rows, segmentation_rows)
    return {
        "CLASSIFICATION_MAIN": classification_block(classification_rows),
        "CLASSIFICATION_SEED_VARIANCE": seed_variance_block(classification_rows),
        "CLASSIFICATION_LEAKAGE_SURFACE": leakage_surface_block(classification_rows),
        "SEGMENTATION_MAIN": segmentation_block(segmentation_rows),
        "SEGMENTATION_SEED_VARIANCE": segmentation_seed_variance_block(segmentation_rows),
        "SEGMENTATION_REPRODUCTION": segmentation_reproduction_block(reproduction),
        "SEGMENTATION_THRESHOLD": threshold_block(segmentation_rows),
        "SEGMENTATION_REPLICATION": segmentation_replication_block(segmentation_rows),
        "RESULT_OUTCOME": outcome_block(outcome),
    }, outcome


def _markers(name: str) -> tuple[str, str]:
    require(name in BLOCK_NAMES, f"Unknown README block: {name}")
    return f"<!-- BEGIN VERIFIED {name} -->", f"<!-- END VERIFIED {name} -->"


def replace_block(readme: str, name: str, value: str) -> str:
    start, end = _markers(name)
    require(readme.count(start) == 1, f"README needs exactly one {start}")
    require(readme.count(end) == 1, f"README needs exactly one {end}")
    before, remainder = readme.split(start, maxsplit=1)
    _, after = remainder.split(end, maxsplit=1)
    return f"{before}{start}\n{value}\n{end}{after}"


def read_block(readme: str, name: str) -> str:
    start, end = _markers(name)
    require(readme.count(start) == 1, f"README needs exactly one {start}")
    require(readme.count(end) == 1, f"README needs exactly one {end}")
    return readme.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0].strip()


def verify_readme(readme: str, blocks: Mapping[str, str]) -> None:
    limitations_heading = "## 限制與誠實揭露"
    require(limitations_heading in readme, "README needs a limitations section")
    for name, expected in blocks.items():
        observed = read_block(readme, name)
        require(observed == expected, f"README verified block is stale: {name}")
    limitations = readme.split(limitations_heading, maxsplit=1)[1]
    require(
        "<!-- BEGIN VERIFIED RESULT_OUTCOME -->" in limitations,
        "RESULT_OUTCOME must remain under the limitations section",
    )
    require("TBD" not in readme, "README still contains TBD")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_text_lf(temporary, text)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument(
        "--classification",
        type=Path,
        default=Path("results/classification.csv"),
    )
    parser.add_argument(
        "--segmentation",
        type=Path,
        default=Path("results/segmentation.csv"),
    )
    parser.add_argument(
        "--reproduction",
        type=Path,
        default=Path("reports/seed42_reproduction.json"),
        help="Output of scripts/verify_seed42_reproduction.py",
    )
    parser.add_argument(
        "--validation-out",
        type=Path,
        default=Path("reports/readme_validation.json"),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Replace only the three marked README blocks before verifying",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    classification_fields, classification_rows = read_csv(args.classification)
    segmentation_fields, segmentation_rows = read_csv(args.segmentation)
    validate_classification_rows(classification_fields, classification_rows)
    validate_segmentation_rows(segmentation_fields, segmentation_rows)
    require(args.reproduction.is_file(), f"Missing reproduction report: {args.reproduction}")
    reproduction = json.loads(args.reproduction.read_text(encoding="utf-8"))
    require(
        reproduction.get("status") == "passed",
        "Seed-42 reproduction report did not pass",
    )
    require(
        reproduction.get("current_sha256") == sha256_file(args.segmentation),
        "Seed-42 reproduction report was built from a different segmentation.csv",
    )
    blocks, outcome = render_blocks(classification_rows, segmentation_rows, reproduction)

    require(args.readme.is_file(), f"Missing README: {args.readme}")
    readme = args.readme.read_text(encoding="utf-8")
    if args.write:
        for name in BLOCK_NAMES:
            readme = replace_block(readme, name, blocks[name])
    verify_readme(readme, blocks)
    if args.write:
        atomic_write_text(args.readme, readme)

    validation = {
        "status": "passed",
        "schema_version": 1,
        "readme_sha256": sha256_file(args.readme),
        "classification_sha256": sha256_file(args.classification),
        "segmentation_sha256": sha256_file(args.segmentation),
        "reproduction_sha256": sha256_file(args.reproduction),
        "block_sha256": {name: canonical_sha256(value) for name, value in blocks.items()},
        "outcome": outcome,
        "negative_results_preserved": True,
    }
    atomic_write_text(
        args.validation_out,
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(f"Verified README: {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
