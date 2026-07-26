import tarfile
from pathlib import Path

import pytest

from scripts.download_visa import ValidationError, safe_extract


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("unsafe", encoding="utf-8")
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(payload, arcname="../escaped.txt")

    with pytest.raises(ValidationError, match="Unsafe tar member"):
        safe_extract(archive_path, tmp_path / "out")

    assert not (tmp_path / "escaped.txt").exists()
