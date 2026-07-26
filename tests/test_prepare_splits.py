from pathlib import Path

import pytest

from scripts.prepare_splits import ValidationError, validate_source_rows


def _rows(object_name: str, split_type: str) -> list[dict[str, str]]:
    from scripts.prepare_splits import EXPECTED_COUNTS

    rows: list[dict[str, str]] = []
    index = 0
    for (set_name, label), count in EXPECTED_COUNTS[split_type][object_name].items():
        for _ in range(count):
            suffix = "normal" if label == "normal" else "anomaly"
            image = f"{object_name}/{suffix}/{index}.jpg"
            mask = "" if label == "normal" else f"{object_name}/masks/{index}.png"
            rows.append(
                {
                    "object": object_name,
                    "split": set_name,
                    "label": label,
                    "image": image,
                    "mask": mask,
                }
            )
            index += 1
    return rows


def test_validate_source_rows_rejects_missing_image(tmp_path: Path) -> None:
    rows = _rows("pcb1", "2cls_fewshot")
    with pytest.raises(ValidationError, match="CSV image is missing"):
        validate_source_rows(
            {"2cls_fewshot": rows},
            objects=("pcb1",),
            visa_raw=tmp_path,
        )
