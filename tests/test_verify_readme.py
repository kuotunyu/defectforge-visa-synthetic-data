from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import pytest

from scripts.aggregate_segmentation import LOGICAL_GROUPS
from scripts.build_phase2_figures import OBJECTS
from scripts.run_classifier_matrix import matrix_plan
from scripts.verify_readme import (
    BLOCK_NAMES,
    MIN_REPLICATED_SEEDS,
    ReadmeVerificationError,
    main,
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
    text = "# 專案\n\n## 實驗結果\n\n"
    for name in BLOCK_NAMES:
        if name == "RESULT_OUTCOME":
            continue
        text += f"<!-- BEGIN VERIFIED {name} -->\nstale\n<!-- END VERIFIED {name} -->\n\n"
    text += "## 限制與誠實揭露\n\n"
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
    assert "Classification 負面結果：**是" in blocks["RESULT_OUTCOME"]
    assert outcome["segmentation_negative"] is False


def test_seed_variance_block_only_lists_replicated_groups() -> None:
    counts: Counter[tuple[str, str]] = Counter(
        (spec.object_name, spec.group) for spec in matrix_plan()
    )
    replicated = {key for key, count in counts.items() if count >= MIN_REPLICATED_SEEDS}
    assert replicated, "the protocol requires at least one replicated classification group"

    blocks, _ = render_blocks(_classification_rows(), _segmentation_rows())
    table = blocks["CLASSIFICATION_SEED_VARIANCE"]
    body = [
        line
        for line in table.splitlines()
        if line.startswith("| ") and "---" not in line and "Seeds" not in line
    ]
    assert len(body) == len(replicated)
    assert "±" in table
    assert all(f"| {MIN_REPLICATED_SEEDS} |" in line for line in body)


def test_outcome_discloses_threshold_sensitivity_without_replacing_dice() -> None:
    segmentation_rows = _segmentation_rows()
    for row in segmentation_rows:
        group = row["logical_group"]
        if group == "real_only":
            row["dice"], row["aupro"] = "0.50", "0.40"
        elif group in {"filtered_syn", "all_mixed"}:
            row["dice"], row["aupro"] = "0.10", "0.90"
        elif group == "copypaste_only":
            row["dice"], row["pixel_auroc"] = "0.0", "0.95"
        elif group == "diffusion_only":
            row["dice"], row["pixel_auroc"] = "0.0", "0.55"

    blocks, outcome = render_blocks(_classification_rows(), segmentation_rows)

    # The pre-registered Dice conclusion must survive untouched.
    assert outcome["segmentation_negative"] is True
    assert outcome["segmentation_dice_delta"] < 0.0
    # AUPRO points the other way and is disclosed alongside, never substituted.
    assert outcome["segmentation_aupro_delta"] > 0.0
    assert outcome["segmentation_metric_directions_agree"] is False
    assert "併列揭露" in blocks["RESULT_OUTCOME"]
    # all_mixed is an alias of filtered_syn and must not inflate the run count.
    assert outcome["segmentation_physical_runs"] == len(segmentation_rows) - len(OBJECTS)
    assert outcome["segmentation_zero_dice_runs"] == 2 * len(OBJECTS)
    assert outcome["segmentation_zero_dice_informative_runs"] == len(OBJECTS)
    assert outcome["segmentation_zero_dice_max_pixel_auroc"] == pytest.approx(0.95)


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


def test_write_mode_validates_candidate_before_changing_readme(tmp_path: Path) -> None:
    classification = tmp_path / "classification.csv"
    with classification.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLASSIFICATION_FIELDS)
        writer.writeheader()
        writer.writerows(_classification_rows())
    segmentation = tmp_path / "segmentation.csv"
    with segmentation.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEGMENTATION_FIELDS)
        writer.writeheader()
        writer.writerows(_segmentation_rows())
    readme = tmp_path / "README.md"
    original = "# 專案\n\n## 實驗結果\n\n"
    for name in BLOCK_NAMES:
        original += (
            f"<!-- BEGIN VERIFIED {name} -->\n"
            "stale\n"
            f"<!-- END VERIFIED {name} -->\n\n"
        )
    readme.write_text(original, encoding="utf-8")
    validation = tmp_path / "readme_validation.json"

    with pytest.raises(ReadmeVerificationError, match="limitations"):
        main(
            [
                "--readme",
                str(readme),
                "--classification",
                str(classification),
                "--segmentation",
                str(segmentation),
                "--validation-out",
                str(validation),
                "--write",
            ]
        )

    assert readme.read_text(encoding="utf-8") == original
    assert not validation.exists()
