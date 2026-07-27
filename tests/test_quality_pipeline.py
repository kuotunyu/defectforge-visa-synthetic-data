from types import SimpleNamespace

from src.evaluation.quality_pipeline import generated_group_indices


def sample(input_name: str, object_name: str, defect_type: str, sample_id: str):
    return SimpleNamespace(
        input_name=input_name,
        record={
            "object": object_name,
            "defect_type": defect_type,
            "sample_id": sample_id,
        },
    )


def decision(passed: bool):
    return {"filter": {"passed": passed}}


def test_generated_group_indices_respects_view_and_type() -> None:
    samples = [
        sample("a", "pcb1", "type0", "0"),
        sample("a", "pcb1", "type1", "1"),
        sample("a", "pcb1", "type0", "2"),
        sample("b", "pcb1", "type0", "3"),
    ]
    decisions = [decision(True), decision(True), decision(False), decision(True)]
    assert generated_group_indices(
        samples,
        decisions,
        input_name="a",
        object_name="pcb1",
        defect_type="type0",
        passed_only=False,
    ) == [0, 2]
    assert generated_group_indices(
        samples,
        decisions,
        input_name="a",
        object_name="pcb1",
        defect_type="type0",
        passed_only=True,
    ) == [0]
