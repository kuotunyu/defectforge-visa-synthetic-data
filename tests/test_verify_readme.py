from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

import pytest

from scripts.aggregate_segmentation import LOGICAL_GROUPS
from scripts.build_phase2_figures import OBJECTS
from scripts.run_classifier_matrix import matrix_plan
from scripts.verify_readme import (
    BLOCK_NAMES,
    ReadmeVerificationError,
    render_blocks,
    replace_block,
    validate_classification_rows,
    validate_segmentation_rows,
    verify_readme,
)

CLASSIFICATION_FIELDS = (
    "requested_group",
    "canonical_group",
    "object",
    "seed",
    "macro_f1",
    "anomaly_f1",
    "auroc",
    "normal_false_positive_rate",
)
SEGMENTATION_FIELDS = (
    "logical_group",
    "canonical_group",
    "object",
    "seed",
    "dice",
    "miou",
    "pixel_auroc",
    "aupro",
)


def _classification_rows() -> list[dict[str, str]]:
    rows = []
    for spec in matrix_plan():
        score = 0.60
        if spec.group == "filtered_syn":
            score = 0.55
        rows.append(
            {
                "requested_group": spec.group,
                "canonical_group": spec.group,
                "object": spec.object_name,
                "seed": str(spec.seed),
                "macro_f1": str(score),
                "anomaly_f1": "0.61",
                "auroc": "0.70",
                "normal_false_positive_rate": "0.10",
            }
        )
    return rows


def _segmentation_rows() -> list[dict[str, str]]:
    rows = []
    for group_name in LOGICAL_GROUPS:
        for object_name in OBJECTS:
            score = 0.50 if group_name == "real_only" else 0.51
            rows.append(
                {
                    "logical_group": group_name,
                    "canonical_group": (
                        "filtered_syn" if group_name == "all_mixed" else group_name
                    ),
                    "object": object_name,
                    "seed": "42",
                    "dice": str(score),
                    "miou": "0.40",
                    "pixel_auroc": "0.80",
                    "aupro": "0.45",
                }
            )
    return rows


def _readme_with_blocks(blocks: Mapping[str, str]) -> str:
    text = "# Project\n\n## Results\n\n"
    for name in ("CLASSIFICATION_MAIN", "SEGMENTATION_MAIN"):
        text += f"<!-- BEGIN VERIFIED {name} -->\nstale\n<!-- END VERIFIED {name} -->\n\n"
    text += "## Limitations\n\n"
    text += (
        "<!-- BEGIN VERIFIED RESULT_OUTCOME -->\n"
        "stale\n"
        "<!-- END VERIFIED RESULT_OUTCOME -->\n"
    )
    for name in BLOCK_NAMES:
        text = replace_block(text, name, blocks[name])
    return text


def test_rendered_readme_preserves_negative_result() -> None:
    classification_rows = _classification_rows()
    segmentation_rows = _segmentation_rows()
    validate_classification_rows(CLASSIFICATION_FIELDS, classification_rows)
    validate_segmentation_rows(SEGMENTATION_FIELDS, segmentation_rows)
    blocks, outcome = render_blocks(classification_rows, segmentation_rows)
    readme = _readme_with_blocks(blocks)
    verify_readme(readme, blocks)
    assert outcome["classification_negative"] is True
    assert "Classification negative result: **yes" in blocks["RESULT_OUTCOME"]
    assert outcome["segmentation_negative"] is False


def test_verify_readme_rejects_stale_numeric_block() -> None:
    blocks, _ = render_blocks(_classification_rows(), _segmentation_rows())
    readme = _readme_with_blocks(blocks).replace("0.6000", "0.9999", 1)
    with pytest.raises(ReadmeVerificationError, match="stale"):
        verify_readme(readme, blocks)


def test_classification_validation_requires_frozen_matrix() -> None:
    rows = _classification_rows()[:-1]
    with pytest.raises(ReadmeVerificationError, match="38-run"):
        validate_classification_rows(CLASSIFICATION_FIELDS, rows)


def test_segmentation_validation_requires_all_mixed_alias() -> None:
    rows = _segmentation_rows()
    all_mixed = next(row for row in rows if row["logical_group"] == "all_mixed")
    all_mixed["canonical_group"] = "all_mixed"
    with pytest.raises(ReadmeVerificationError, match="cite filtered_syn"):
        validate_segmentation_rows(SEGMENTATION_FIELDS, rows)


def test_csv_fixture_columns_remain_serializable(tmp_path: Path) -> None:
    path = tmp_path / "classification.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLASSIFICATION_FIELDS)
        writer.writeheader()
        writer.writerows(_classification_rows())
    assert path.read_text(encoding="utf-8").startswith("requested_group,")
