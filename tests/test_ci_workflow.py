from __future__ import annotations

from pathlib import Path


def test_verify_workflow_is_locked_and_fail_closed() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "verify.yml"
    ).read_text(encoding="utf-8")

    assert "permissions:\n  contents: read" in workflow
    assert "runs-on: windows-latest" in workflow
    assert "fetch-depth: 0" in workflow
    assert "persist-credentials: false" in workflow
    assert "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09" in workflow
    assert "astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990" in workflow
    assert 'version: "0.11.18"' in workflow
    assert "uv sync --frozen" in workflow
    assert "uv run --frozen ruff check ." in workflow
    assert "uv run --frozen pytest -q" in workflow
    assert "uv run --frozen python scripts/verify_publish.py" in workflow
