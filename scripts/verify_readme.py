"""Rebuild and verify every numeric README result block from raw result CSV files."""

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

from scripts.aggregate_segmentation import LOGICAL_GROUPS  # isort: skip
from scripts.build_phase2_figures import MAIN_GROUPS, OBJECTS  # isort: skip
from scripts.run_classifier_matrix import matrix_plan  # isort: skip

BLOCK_NAMES = (
    "CLASSIFICATION_MAIN",
    "SEGMENTATION_MAIN",
    "RESULT_OUTCOME",
)
CLASSIFICATION_METRICS = (
    "macro_f1",
    "anomaly_f1",
    "auroc",
    "normal_false_positive_rate",
)
SEGMENTATION_METRICS = ("dice", "miou", "pixel_auroc", "aupro")
GROUP_LABELS = {
    "real_only": "Real-only (10)",
    "std_aug": "+ Standard Augmentation",
    "unfiltered_syn": "+ Unfiltered Synthetic",
    "filtered_syn": "+ Filtered Synthetic",
    "full_real": "Full-real (60)",
    "procedural_only": "Procedural-only",
    "copypaste_only": "Copy-paste-only",
    "diffusion_only": "Diffusion-only",
    "all_mixed": "All-mixed (alias of Filtered Synthetic)",
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
    expected = {
        (group_name, object_name)
        for group_name in LOGICAL_GROUPS
        for object_name in OBJECTS
    }
    observed: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["logical_group"], row["object"])
        require(key not in observed, f"Duplicate segmentation result: {key}")
        observed.add(key)
        require(int(row["seed"]) == 42, f"Unexpected segmentation seed: {key}")
        if row["logical_group"] == "all_mixed":
            require(
                row["canonical_group"] == "filtered_syn",
                "all_mixed must cite filtered_syn",
            )
        for metric in SEGMENTATION_METRICS:
            _finite_unit_interval(row, metric, label=f"segmentation/{key}")
    require(observed == expected, "segmentation.csv does not match the 18 logical M20 rows")


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
        ("Object", "Training group", "Macro-F1", "Defect F1", "AUROC", "Normal FPR"),
        output,
    )


def segmentation_block(rows: Sequence[Mapping[str, str]]) -> str:
    output: list[list[str]] = []
    for object_name in OBJECTS:
        for group_name in LOGICAL_GROUPS:
            row = _one(
                rows,
                key="logical_group",
                value=group_name,
                object_name=object_name,
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
        ("Object", "Training group", "Dice", "mIoU", "Pixel AUROC", "AUPRO"),
        output,
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
    segmentation_delta = _mean_metric(
        segmentation_rows,
        key="logical_group",
        group_name="filtered_syn",
        metric="dice",
    ) - _mean_metric(
        segmentation_rows,
        key="logical_group",
        group_name="real_only",
        metric="dice",
    )
    return {
        "classification_macro_f1_delta": classification_delta,
        "segmentation_dice_delta": segmentation_delta,
        "classification_negative": classification_delta <= 0.0,
        "segmentation_negative": segmentation_delta <= 0.0,
    }


def outcome_block(payload: Mapping[str, Any]) -> str:
    classification_delta = float(payload["classification_macro_f1_delta"])
    segmentation_delta = float(payload["segmentation_dice_delta"])
    classification_statement = (
        "yes — Filtered Synthetic did not improve mean Macro-F1."
        if payload["classification_negative"]
        else "no — Filtered Synthetic improved mean Macro-F1."
    )
    segmentation_statement = (
        "yes — Filtered Synthetic did not improve mean Dice."
        if payload["segmentation_negative"]
        else "no — Filtered Synthetic improved mean Dice."
    )
    return "\n".join(
        [
            (
                f"- Classification, Filtered Synthetic vs Real-only: "
                f"`{classification_delta:+.4f}` mean Macro-F1."
            ),
            (
                f"- Segmentation, Filtered Synthetic vs Real-only: "
                f"`{segmentation_delta:+.4f}` mean Dice."
            ),
            f"- Classification negative result: **{classification_statement}**",
            f"- Segmentation negative result: **{segmentation_statement}**",
        ]
    )


def render_blocks(
    classification_rows: Sequence[Mapping[str, str]],
    segmentation_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], dict[str, Any]]:
    outcome = outcome_payload(classification_rows, segmentation_rows)
    return {
        "CLASSIFICATION_MAIN": classification_block(classification_rows),
        "SEGMENTATION_MAIN": segmentation_block(segmentation_rows),
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
    require("## Limitations" in readme, "README needs a Limitations section")
    for name, expected in blocks.items():
        observed = read_block(readme, name)
        require(observed == expected, f"README verified block is stale: {name}")
    limitations = readme.split("## Limitations", maxsplit=1)[1]
    require(
        "<!-- BEGIN VERIFIED RESULT_OUTCOME -->" in limitations,
        "RESULT_OUTCOME must remain under Limitations",
    )
    require("TBD" not in readme, "README still contains TBD")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
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
    blocks, outcome = render_blocks(classification_rows, segmentation_rows)

    require(args.readme.is_file(), f"Missing README: {args.readme}")
    readme = args.readme.read_text(encoding="utf-8")
    if args.write:
        for name in BLOCK_NAMES:
            readme = replace_block(readme, name, blocks[name])
        atomic_write_text(args.readme, readme)
    verify_readme(readme, blocks)

    validation = {
        "status": "passed",
        "schema_version": 1,
        "readme_sha256": sha256_file(args.readme),
        "classification_sha256": sha256_file(args.classification),
        "segmentation_sha256": sha256_file(args.segmentation),
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
