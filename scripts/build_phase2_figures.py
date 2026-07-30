"""Build the preregistered M21 result figures from verified result CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.isotonic import IsotonicRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OBJECTS = ("pcb1", "capsules")
OBJECT_COLORS = {"pcb1": "#0b7285", "capsules": "#e8590c"}
# ADR-032 replicated segmentation across three seeds. The preregistered figures stay
# anchored to seed 42; the spread is reported in the README, not redrawn here. Kept as a
# literal so this module needs no project imports.
SEGMENTATION_SEEDS = (42, 43, 44)
ANCHOR_SEED = 42
MAIN_GROUPS = (
    "real_only",
    "std_aug",
    "unfiltered_syn",
    "filtered_syn",
    "full_real",
)
SEGMENTATION_GROUPS = (
    "real_only",
    "std_aug",
    "unfiltered_syn",
    "filtered_syn",
    "full_real",
    "procedural_only",
    "copypaste_only",
    "diffusion_only",
    "all_mixed",
)


class Phase2FigureError(RuntimeError):
    """Raised when a figure cannot be traced to a unique result row."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase2FigureError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _one(rows: Sequence[Mapping[str, str]], message: str) -> Mapping[str, str]:
    require(len(rows) == 1, f"{message}: expected 1 row, found {len(rows)}")
    return rows[0]


def classification_value(
    rows: Sequence[Mapping[str, str]],
    *,
    object_name: str,
    group_name: str,
    metric: str,
) -> float:
    row = _one(
        [
            item
            for item in rows
            if item["object"] == object_name
            and item["canonical_group"] == group_name
            and int(item["seed"]) == 42
        ],
        f"Missing M16 row {object_name}/{group_name}",
    )
    value = float(row[metric])
    require(math.isfinite(value), f"Nonfinite M16 value {object_name}/{group_name}/{metric}")
    return value


def segmentation_value(
    rows: Sequence[Mapping[str, str]],
    *,
    object_name: str,
    group_name: str,
    metric: str,
) -> float:
    row = _one(
        [
            item
            for item in rows
            if item["object"] == object_name
            and item["logical_group"] == group_name
            and int(item["seed"]) == ANCHOR_SEED
        ],
        f"Missing M20 row {object_name}/{group_name}",
    )
    value = float(row[metric])
    require(math.isfinite(value), f"Nonfinite M20 value {object_name}/{group_name}/{metric}")
    return value


def validate_figure_inputs(
    classification_rows: Sequence[Mapping[str, str]],
    segmentation_rows: Sequence[Mapping[str, str]],
) -> None:
    classification_columns = {
        "requested_group",
        "canonical_group",
        "object",
        "seed",
        "macro_f1",
        "auroc",
    }
    segmentation_columns = {
        "logical_group",
        "canonical_group",
        "object",
        "seed",
        "dice",
        "aupro",
    }
    require(bool(classification_rows), "Classification result CSV is empty")
    require(bool(segmentation_rows), "Segmentation result CSV is empty")
    require(
        classification_columns <= set(classification_rows[0]),
        "Classification result CSV columns are incomplete",
    )
    require(
        segmentation_columns <= set(segmentation_rows[0]),
        "Segmentation result CSV columns are incomplete",
    )
    classification_keys = [
        (row["requested_group"], row["object"], int(row["seed"]))
        for row in classification_rows
    ]
    require(
        len(classification_keys) == 38 and len(set(classification_keys)) == 38,
        "Classification result CSV must contain 38 unique formal rows",
    )
    expected_segmentation = {
        (group_name, object_name, seed)
        for group_name in SEGMENTATION_GROUPS
        for object_name in OBJECTS
        for seed in SEGMENTATION_SEEDS
    }
    segmentation_keys = [
        (row["logical_group"], row["object"], int(row["seed"])) for row in segmentation_rows
    ]
    require(
        len(segmentation_keys) == len(expected_segmentation)
        and len(set(segmentation_keys)) == len(expected_segmentation)
        and set(segmentation_keys) == expected_segmentation,
        f"Segmentation result CSV must contain {len(expected_segmentation)} exact logical rows",
    )
    required_macro_f1_groups = tuple(
        dict.fromkeys((*MAIN_GROUPS, "real_20", "syn_125", "syn_250"))
    )
    for object_name in OBJECTS:
        for group_name in required_macro_f1_groups:
            classification_value(
                classification_rows,
                object_name=object_name,
                group_name=group_name,
                metric="macro_f1",
            )
        for group_name in MAIN_GROUPS:
            classification_value(
                classification_rows,
                object_name=object_name,
                group_name=group_name,
                metric="auroc",
            )
        for group_name in SEGMENTATION_GROUPS:
            for metric in ("dice", "aupro"):
                segmentation_value(
                    segmentation_rows,
                    object_name=object_name,
                    group_name=group_name,
                    metric=metric,
                )


def equivalent_real_count(
    real_counts: Sequence[float],
    real_scores: Sequence[float],
    synthetic_score: float,
) -> dict[str, Any]:
    x = np.asarray(real_counts, dtype=np.float64)
    y = np.asarray(real_scores, dtype=np.float64)
    require(
        x.ndim == y.ndim == 1 and len(x) == len(y) >= 2,
        "Invalid real scaling inputs",
    )
    require(np.all(np.diff(x) > 0), "Real counts must increase")
    fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(x, y)
    if synthetic_score <= fitted[0]:
        estimate = float(x[0])
        relation = "at_or_below"
    elif synthetic_score >= fitted[-1]:
        estimate = float(x[-1])
        relation = "at_or_above"
    else:
        unique_scores: list[float] = []
        unique_counts: list[float] = []
        for count, score in zip(x, fitted, strict=True):
            if unique_scores and score == unique_scores[-1]:
                unique_counts[-1] = float(count)
            else:
                unique_scores.append(float(score))
                unique_counts.append(float(count))
        estimate = float(np.interp(synthetic_score, unique_scores, unique_counts))
        relation = "interpolated"
    return {
        "estimate": estimate,
        "relation": relation,
        "raw_scores": y.tolist(),
        "isotonic_scores": fitted.tolist(),
        "synthetic_score": float(synthetic_score),
    }


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_real_scaling(
    rows: Sequence[Mapping[str, str]],
    path: Path,
) -> dict[str, Any]:
    counts = (10.0, 20.0, 60.0)
    groups = ("real_only", "real_20", "full_real")
    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    equivalents: dict[str, Any] = {}
    for object_name in OBJECTS:
        scores = [
            classification_value(
                rows,
                object_name=object_name,
                group_name=group,
                metric="macro_f1",
            )
            for group in groups
        ]
        synthetic = classification_value(
            rows,
            object_name=object_name,
            group_name="filtered_syn",
            metric="macro_f1",
        )
        equivalent = equivalent_real_count(counts, scores, synthetic)
        equivalents[object_name] = equivalent
        color = OBJECT_COLORS[object_name]
        axis.plot(counts, scores, marker="o", linewidth=2.0, color=color, label=object_name)
        axis.scatter(
            equivalent["estimate"],
            synthetic,
            marker="*",
            s=190,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        relation = equivalent["relation"]
        prefix = "≈"
        if relation == "at_or_below":
            prefix = "≤"
        elif relation == "at_or_above":
            prefix = "≥"
        axis.annotate(
            f"+500 filtered syn: {prefix}{equivalent['estimate']:.1f} real",
            (equivalent["estimate"], synthetic),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
    axis.set_xticks(counts)
    axis.set_xlabel("Number of real defect training images")
    axis.set_ylabel("Macro-F1")
    axis.set_title("Real-data scaling and filtered-synthetic equivalent")
    axis.grid(True, color="#dee2e6", linewidth=0.7)
    axis.legend(frameon=False)
    _save(figure, path)
    return equivalents


def plot_synthetic_volume(
    rows: Sequence[Mapping[str, str]],
    path: Path,
) -> None:
    counts = (125, 250, 500)
    groups = ("syn_125", "syn_250", "filtered_syn")
    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for object_name in OBJECTS:
        color = OBJECT_COLORS[object_name]
        scores = [
            classification_value(
                rows,
                object_name=object_name,
                group_name=group,
                metric="macro_f1",
            )
            for group in groups
        ]
        baseline = classification_value(
            rows,
            object_name=object_name,
            group_name="real_only",
            metric="macro_f1",
        )
        axis.plot(counts, scores, marker="o", linewidth=2.0, color=color, label=object_name)
        axis.axhline(baseline, color=color, linewidth=1.0, linestyle="--", alpha=0.55)
    axis.set_xticks(counts)
    axis.set_xlabel("Filtered synthetic defect images added")
    axis.set_ylabel("Macro-F1")
    axis.set_title("Synthetic-volume sweep (dashed = Real-only)")
    axis.grid(True, color="#dee2e6", linewidth=0.7)
    axis.legend(frameon=False)
    _save(figure, path)


def _table_figure(
    *,
    column_labels: Sequence[str],
    row_labels: Sequence[str],
    cells: Sequence[Sequence[str]],
    title: str,
    path: Path,
    width: float,
) -> None:
    height = 1.6 + 0.48 * len(row_labels)
    figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        rowLabels=row_labels,
        colLabels=column_labels,
        cellLoc="center",
        rowLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e7f5ff")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8f9fa")
    axis.set_title(title, fontsize=13, pad=12)
    _save(figure, path)


def plot_main_comparison(
    rows: Sequence[Mapping[str, str]],
    path: Path,
) -> None:
    cells = []
    for group_name in MAIN_GROUPS:
        object_values = {
            object_name: {
                metric: classification_value(
                    rows,
                    object_name=object_name,
                    group_name=group_name,
                    metric=metric,
                )
                for metric in ("macro_f1", "auroc")
            }
            for object_name in OBJECTS
        }
        cells.append(
            [
                f"{object_values['pcb1']['macro_f1']:.4f}",
                f"{object_values['pcb1']['auroc']:.4f}",
                f"{object_values['capsules']['macro_f1']:.4f}",
                f"{object_values['capsules']['auroc']:.4f}",
                f"{np.mean([object_values[name]['macro_f1'] for name in OBJECTS]):.4f}",
                f"{np.mean([object_values[name]['auroc'] for name in OBJECTS]):.4f}",
            ]
        )
    _table_figure(
        column_labels=(
            "PCB F1",
            "PCB AUROC",
            "Capsules F1",
            "Capsules AUROC",
            "Mean F1",
            "Mean AUROC",
        ),
        row_labels=MAIN_GROUPS,
        cells=cells,
        title="Five-group classification comparison",
        path=path,
        width=10.5,
    )


def plot_segmentation_table(
    rows: Sequence[Mapping[str, str]],
    path: Path,
) -> None:
    cells = []
    for group_name in SEGMENTATION_GROUPS:
        values = {
            object_name: {
                metric: segmentation_value(
                    rows,
                    object_name=object_name,
                    group_name=group_name,
                    metric=metric,
                )
                for metric in ("dice", "aupro")
            }
            for object_name in OBJECTS
        }
        cells.append(
            [
                f"{values['pcb1']['dice']:.4f}",
                f"{values['pcb1']['aupro']:.4f}",
                f"{values['capsules']['dice']:.4f}",
                f"{values['capsules']['aupro']:.4f}",
                f"{np.mean([values[name]['dice'] for name in OBJECTS]):.4f}",
                f"{np.mean([values[name]['aupro'] for name in OBJECTS]):.4f}",
            ]
        )
    _table_figure(
        column_labels=(
            "PCB Dice",
            "PCB AUPRO",
            "Capsules Dice",
            "Capsules AUPRO",
            "Mean Dice",
            "Mean AUPRO",
        ),
        row_labels=SEGMENTATION_GROUPS,
        cells=cells,
        title=(
            f"Nine logical segmentation groups, seed {ANCHOR_SEED} anchor "
            "(all_mixed cites filtered_syn)"
        ),
        path=path,
        width=11.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument(
        "--validation-out",
        type=Path,
        default=Path("reports/phase2_figures_validation.json"),
    )
    args = parser.parse_args()
    require(args.classification.is_file(), f"Missing result CSV: {args.classification}")
    require(args.segmentation.is_file(), f"Missing result CSV: {args.segmentation}")
    classification_path = args.classification.resolve(strict=True)
    segmentation_path = args.segmentation.resolve(strict=True)
    classification_rows = read_csv(classification_path)
    segmentation_rows = read_csv(segmentation_path)
    validate_figure_inputs(classification_rows, segmentation_rows)
    output_dir = args.output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    figure_names = {
        "real_scaling_curve": "real_scaling_curve.png",
        "synthetic_volume_curve": "synthetic_volume_curve.png",
        "main_comparison_table": "main_comparison_table.png",
        "segmentation_table": "segmentation_table.png",
    }
    with tempfile.TemporaryDirectory(
        prefix=".phase2_figures_",
        dir=output_dir.parent,
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        temporary_figures = {
            name: temporary_dir / filename for name, filename in figure_names.items()
        }
        equivalents = plot_real_scaling(
            classification_rows,
            temporary_figures["real_scaling_curve"],
        )
        plot_synthetic_volume(
            classification_rows,
            temporary_figures["synthetic_volume_curve"],
        )
        plot_main_comparison(
            classification_rows,
            temporary_figures["main_comparison_table"],
        )
        plot_segmentation_table(
            segmentation_rows,
            temporary_figures["segmentation_table"],
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        figures = {
            name: output_dir / filename for name, filename in figure_names.items()
        }
        for name, final_path in figures.items():
            os.replace(temporary_figures[name], final_path)
    payload = {
        "status": "passed",
        "schema_version": 1,
        "classification_sha256": sha256_file(classification_path),
        "segmentation_sha256": sha256_file(segmentation_path),
        "filtered_synthetic_equivalent_real_count": equivalents,
        "figures": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in figures.items()
        },
        "visual_inspection_required": True,
    }
    args.validation_out.parent.mkdir(parents=True, exist_ok=True)
    validation_temporary = args.validation_out.with_suffix(args.validation_out.suffix + ".tmp")
    validation_temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(validation_temporary, args.validation_out)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
