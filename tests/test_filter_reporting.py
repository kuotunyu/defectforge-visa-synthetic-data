import pytest

from src.filtering.reporting import (
    FilterReportError,
    deterministic_selection,
    embedded_summary,
    render_markdown,
    summarize,
    summary_sha256,
)


def record(sample_id: str, reasons: list[str]) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "object": "pcb1",
        "generator": "stageA_copypaste",
        "defect_type": "type0",
        "filter": {
            "passed": not reasons,
            "reject_reasons": reasons,
            "input_name": "stageA_copypaste",
            "thresholds": {"tau_copy": 0.98},
        },
    }


def test_summary_roundtrips_through_markdown() -> None:
    records = [
        record("a", []),
        record("b", ["AREA_OUT_OF_RANGE", "NN_TOO_LOW"]),
    ]
    summary = summarize(records)
    markdown = render_markdown(summary)
    assert embedded_summary(markdown) == summary
    assert len(summary_sha256(summary)) == 64
    assert summary["accepted"] == 1
    assert summary["reason_counts"]["NN_TOO_LOW"] == 1


def test_summary_rejects_inconsistent_pass_flag() -> None:
    invalid = record("a", [])
    invalid["filter"]["passed"] = False
    with pytest.raises(FilterReportError, match="disagree"):
        summarize([invalid])


def test_deterministic_selection_spans_population() -> None:
    records = [record(str(index), []) for index in range(10)]
    selected = deterministic_selection(records, passed=True, count=3)
    assert [value["sample_id"] for value in selected] == ["0", "4", "9"]
