import pytest

from scripts.verify_phase1 import (
    Phase1AcceptanceError,
    checked_milestones,
    parse_identity_rows,
    validate_instruction_handoff,
)


def test_checked_milestones_ignores_unchecked_and_phase2_rows() -> None:
    plan = """- [x] **M0** rules
- [x] **M1** setup
- [ ] **M2** data
- [x] **M16** classification"""
    assert checked_milestones(plan) == {0, 1, 16}


def test_instruction_handoff_accepts_m11_and_m18_boundary() -> None:
    rows = """| 1. 上 Colab 方式 | complete |
| 2. Runtime 選型 | complete |
| 3. 需要的 Colab Secrets | complete |
| 4. 實測時數與 compute units | 已完成；CU unavailable |
| 5. 跑完要下載哪些檔案 / 放回哪個路徑 | complete |"""
    text = f"## Notebook 1\n{rows}\n## Notebook 2 — M18（尚未建立）\n由 M18 負責"
    assert validate_instruction_handoff(text)["m11_handoff_rows"] == 5


def test_instruction_handoff_accepts_completed_m18_handoff() -> None:
    notebook_1_rows = """| 1. 上 Colab 方式 | complete |
| 2. Runtime 選型 | complete |
| 3. 需要的 Colab Secrets | complete |
| 4. 實測時數與 compute units | 已完成；CU unavailable |
| 5. 跑完要下載哪些檔案 / 放回哪個路徑 | complete |"""
    notebook_2_rows = """| 1. 上 Colab 方式 | complete |
| 2. Runtime 選型 | complete |
| 3. Colab Secrets | none |
| 4. 實測時數與 compute units | recorded |
| 5. 跑完下載／回收 | exact ZIP names |"""
    text = (
        f"## Notebook 1\n{notebook_1_rows}\n"
        f"## Notebook 2 — 02_train_segformer.ipynb\n{notebook_2_rows}\n"
    )

    result = validate_instruction_handoff(text)

    assert result["m18_status"] == "handoff_complete"


def test_instruction_handoff_rejects_dependency_cycle() -> None:
    rows = """| 1. 上 Colab 方式 | complete |
| 2. Runtime 選型 | complete |
| 3. 需要的 Colab Secrets | complete |
| 4. 實測時數與 compute units | 已完成 |
| 5. 跑完要下載哪些檔案 / 放回哪個路徑 | complete |"""
    text = f"## Notebook 1\n{rows}\n## Notebook 2 — M18（尚未建立）\nM15 最終驗收"
    with pytest.raises(Phase1AcceptanceError, match="dependency cycle"):
        validate_instruction_handoff(text)


def test_parse_identity_rows_requires_complete_git_fields() -> None:
    row = (
        "abc\tkuotunyu\t61350295+kuotunyu@users.noreply.github.com\t"
        "kuotunyu\t61350295+kuotunyu@users.noreply.github.com\tfeat(M1): setup"
    )
    assert parse_identity_rows(row)[0]["subject"] == "feat(M1): setup"

    with pytest.raises(Phase1AcceptanceError, match="Unexpected git identity row"):
        parse_identity_rows("incomplete\trow")
