from __future__ import annotations

from pathlib import Path

import pytest

from scripts.package_m18_colab import M18PackageError, _safe_pool_id, write_zip


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
