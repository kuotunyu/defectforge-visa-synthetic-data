"""Join M14 quality rows to M16 source ablations and plot quality vs improvement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE_GROUPS = {
    "src_copypaste": "stageA_copypaste",
    "src_procedural": "stageA_procedural",
    "src_diffusion": "stageB_sd2/searched",
}
OBJECT_COLORS = {"pcb1": "#0b7285", "capsules": "#e8590c"}
SOURCE_MARKERS = {
    "src_copypaste": "o",
    "src_procedural": "s",
    "src_diffusion": "^",
}


class QualityDownstreamError(RuntimeError):
    """Raised when M17 cannot make an exact preregistered join."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualityDownstreamError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _one(rows: Sequence[Mapping[str, str]], message: str) -> Mapping[str, str]:
    require(len(rows) == 1, f"{message}: expected 1 row, found {len(rows)}")
    return rows[0]


def build_points(
    classification_rows: Sequence[Mapping[str, str]],
    quality_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    objects = sorted({row["object"] for row in classification_rows})
    require(objects == ["capsules", "pcb1"], f"Unexpected classification objects: {objects}")
    points: list[dict[str, Any]] = []
    for object_name in objects:
        baseline = _one(
            [
                row
                for row in classification_rows
                if row["object"] == object_name
                and row["canonical_group"] == "real_only"
                and int(row["seed"]) == 42
            ],
            f"Missing real_only baseline for {object_name}",
        )
        baseline_f1 = float(baseline["macro_f1"])
        for group_name, input_name in SOURCE_GROUPS.items():
            downstream = _one(
                [
                    row
                    for row in classification_rows
                    if row["object"] == object_name
                    and row["canonical_group"] == group_name
                    and int(row["seed"]) == 42
                ],
                f"Missing {group_name} classification row for {object_name}",
            )
            quality = _one(
                [
                    row
                    for row in quality_rows
                    if row["object"] == object_name
                    and row["view"] == "unfiltered"
                    and row["input_name"] == input_name
                    and row["defect_type"] == "__all__"
                    and row["status"] == "ok"
                ],
                f"Missing {input_name} quality row for {object_name}",
            )
            kid = float(quality["kid"])
            nn_mean = float(quality["nn_mean"])
            macro_f1 = float(downstream["macro_f1"])
            require(all(math.isfinite(value) for value in (kid, nn_mean, macro_f1)), "Nonfinite M17")
            points.append(
                {
                    "object": object_name,
                    "source_group": group_name,
                    "quality_input_name": input_name,
                    "quality_view": "unfiltered",
                    "quality_defect_type": "__all__",
                    "kid": kid,
                    "nn_mean": nn_mean,
                    "baseline_run_name": baseline["run_name"],
                    "baseline_macro_f1": baseline_f1,
                    "downstream_run_name": downstream["run_name"],
                    "downstream_macro_f1": macro_f1,
                    "macro_f1_delta": macro_f1 - baseline_f1,
                }
            )
    require(len(points) == 6, "M17 must contain exactly six preregistered source points")
    return points


def correlation(points: Sequence[Mapping[str, Any]]) -> float | None:
    x = np.asarray([float(point["kid"]) for point in points], dtype=np.float64)
    y = np.asarray([float(point["macro_f1_delta"]) for point in points], dtype=np.float64)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def plot(points: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for point in points:
        object_name = str(point["object"])
        source_group = str(point["source_group"])
        axis.scatter(
            float(point["kid"]),
            float(point["macro_f1_delta"]),
            color=OBJECT_COLORS[object_name],
            marker=SOURCE_MARKERS[source_group],
            s=85,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        short_source = source_group.removeprefix("src_")
        axis.annotate(
            f"{object_name}/{short_source}",
            (float(point["kid"]), float(point["macro_f1_delta"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axhline(0.0, color="#495057", linewidth=1.0, linestyle="--")
    axis.grid(True, color="#dee2e6", linewidth=0.7, alpha=0.8)
    axis.set_xlabel("Generation KID (lower is better)")
    axis.set_ylabel("Macro-F1 change vs Real-only")
    axis.set_title("Generation quality vs downstream utility")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_report(points: Sequence[Mapping[str, Any]], pearson: float | None) -> str:
    lines = [
        "# M17 generation quality vs downstream improvement",
        "",
        (
            "Preregistered join: the three unfiltered source-only generation rows are matched to "
            "their seed-42 M16 source-ablation classifiers. Improvement is measured against the "
            "same object's seed-42 Real-only Macro-F1."
        ),
        "",
        "| Object | Source | KID ↓ | NN mean ↑ | Real-only F1 | Source F1 | Δ Macro-F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        lines.append(
            f"| {point['object']} | {point['source_group']} | {point['kid']:.6f} | "
            f"{point['nn_mean']:.6f} | {point['baseline_macro_f1']:.6f} | "
            f"{point['downstream_macro_f1']:.6f} | {point['macro_f1_delta']:+.6f} |"
        )
    if pearson is None:
        interpretation = "Pearson correlation is undefined because one axis is constant."
    else:
        interpretation = (
            f"Across the six preregistered points, Pearson r(KID, ΔMacro-F1) = {pearson:.4f}. "
            "This descriptive six-point statistic is not treated as a significance test."
        )
    lines.extend(
        [
            "",
            interpretation,
            "",
            (
                "The sign and strength are reported as observed; no source, metric, or object is "
                "removed after reading downstream results."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classification",
        type=Path,
        default=Path("results/classification.csv"),
    )
    parser.add_argument(
        "--quality",
        type=Path,
        default=Path("results/generation_quality.csv"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("reports/figures/quality_vs_downstream.png"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/quality_vs_downstream.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/quality_vs_downstream.json"),
    )
    args = parser.parse_args()
    classification_path = args.classification.resolve(strict=True)
    quality_path = args.quality.resolve(strict=True)
    points = build_points(read_csv(classification_path), read_csv(quality_path))
    pearson = correlation(points)
    plot(points, args.figure)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(points, pearson), encoding="utf-8")
    payload = {
        "status": "passed",
        "schema_version": 1,
        "points": points,
        "pearson_kid_vs_macro_f1_delta": pearson,
        "classification_sha256": sha256_file(classification_path),
        "generation_quality_sha256": sha256_file(quality_path),
        "figure_sha256": sha256_file(args.figure),
        "report_sha256": sha256_file(args.report),
        "selection_policy": "three_unfiltered_source_only_groups_seed42_preregistered",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
