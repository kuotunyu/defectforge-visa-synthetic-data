from __future__ import annotations

from pathlib import Path

import pytest

from scripts.package_m18_colab import (
    REQUIRED_UNTRACKED,
    SOURCE_DIRECTORIES,
    SOURCE_ROOT_FILES,
    M18PackageError,
    _safe_pool_id,
    source_files,
    write_zip,
)


def test_safe_pool_id_is_windows_portable_and_view_specific() -> None:
    assert _safe_pool_id("stageB_sd2/searched", "pcb1::type0") == (
        "stageB_sd2__searched__pcb1_type0"
    )
    assert _safe_pool_id("filtered", "sample") != _safe_pool_id("unfiltered", "sample")


def test_write_zip_rejects_duplicate_member_names(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("data", encoding="utf-8")
    with pytest.raises(M18PackageError, match="duplicate members"):
        write_zip(
            tmp_path / "duplicate.zip",
            [(source, "same.txt"), (source, "same.txt")],
        )


def _write_minimal_source_tree(root: Path) -> None:
    for relative in SOURCE_ROOT_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source\n", encoding="utf-8")
    for relative in SOURCE_DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)
    for relative in REQUIRED_UNTRACKED:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source\n", encoding="utf-8")


def test_source_files_work_without_git_and_exclude_unrelated_outputs(tmp_path: Path) -> None:
    _write_minimal_source_tree(tmp_path)
    included = tmp_path / "src" / "module.py"
    included.write_text("pass\n", encoding="utf-8")
    excluded = tmp_path / "results" / "model.safetensors"
    excluded.parent.mkdir()
    excluded.write_bytes(b"weights")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "installed.py").write_text("pass\n", encoding="utf-8")

    files = source_files(tmp_path)

    assert included in files
    assert excluded not in files
    assert tmp_path / ".venv" / "installed.py" not in files
    assert all(".git" not in path.parts for path in files)


def test_source_files_reject_secret_named_files_in_allowlisted_tree(tmp_path: Path) -> None:
    _write_minimal_source_tree(tmp_path)
    (tmp_path / "configs" / "local.env").write_text("TOKEN=value\n", encoding="utf-8")

    with pytest.raises(M18PackageError, match="secret or Git internals"):
        source_files(tmp_path)
