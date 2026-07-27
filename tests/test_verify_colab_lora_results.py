import zipfile
from pathlib import Path

import pytest

from scripts.verify_colab_lora_results import (
    ColabResultValidationError,
    validate_archive,
)


def test_validate_archive_reports_hash_size_and_member_count(tmp_path: Path) -> None:
    archive_path = tmp_path / "results.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("lora_sdxl/result.json", "{}")

    result = validate_archive(archive_path)

    assert result["file"] == archive_path.name
    assert result["bytes"] == archive_path.stat().st_size
    assert result["members"] == 1
    assert len(result["sha256"]) == 64


def test_validate_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.json", "{}")

    with pytest.raises(ColabResultValidationError, match="Unsafe archive member"):
        validate_archive(archive_path)
