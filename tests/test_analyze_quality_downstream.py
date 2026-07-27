from __future__ import annotations

import pytest

from scripts.analyze_quality_downstream import (
    SOURCE_GROUPS,
    QualityDownstreamError,
    build_points,
    correlation,
)


def _classification_rows() -> list[dict[str, str]]:
    rows = []
    for object_name in ("pcb1", "capsules"):
        rows.append(
            {
                "object": object_name,
                "canonical_group": "real_only",
                "seed": "42",
                "run_name": f"real-{object_name}",
                "macro_f1": "0.5",
            }
        )
        for index, group_name in enumerate(SOURCE_GROUPS, start=1):
            rows.append(
                {
                    "object": object_name,
                    "canonical_group": group_name,
                    "seed": "42",
                    "run_name": f"{group_name}-{object_name}",
                    "macro_f1": str(0.5 + index / 10),
                }
            )
    return rows


def _quality_rows() -> list[dict[str, str]]:
    rows = []
    for object_index, object_name in enumerate(("pcb1", "capsules")):
        for source_index, input_name in enumerate(SOURCE_GROUPS.values()):
            rows.append(
                {
                    "object": object_name,
                    "view": "unfiltered",
                    "input_name": input_name,
                    "defect_type": "__all__",
                    "status": "ok",
                    "kid": str(object_index + source_index / 10),
                    "nn_mean": "0.7",
                }
            )
    return rows


def test_build_points_exactly_joins_six_preregistered_rows() -> None:
    points = build_points(_classification_rows(), _quality_rows())
    assert len(points) == 6
    assert {point["source_group"] for point in points} == set(SOURCE_GROUPS)
    assert all(point["macro_f1_delta"] > 0 for point in points)
    assert correlation(points) is not None


def test_build_points_rejects_duplicate_quality_row() -> None:
    quality = _quality_rows()
    quality.append(dict(quality[0]))
    with pytest.raises(QualityDownstreamError, match="expected 1 row, found 2"):
        build_points(_classification_rows(), quality)
