from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.verify_publish import (
    COAUTHOR_TRAILER,
    EXPECTED_EMAIL,
    EXPECTED_NAME,
    audit_identities,
    scan_text,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_scan_text_reports_locations_without_secret_values() -> None:
    fake_github = "gho_" + "a" * 24
    fake_hf = "hf_" + "b" * 24
    private_path = "C:" + "\\Users\\private\\repo"
    text = f"first={fake_github}\nsecond={fake_hf}\npath={private_path}\n"
    findings = scan_text(Path("fixture.txt"), text)
    assert [item["kind"] for item in findings] == [
        "github_token",
        "huggingface_token",
        "personal_windows_path",
    ]
    serialized = repr(findings)
    assert fake_github not in serialized
    assert fake_hf not in serialized


def test_coauthor_scan_requires_a_real_trailer_line() -> None:
    prose = "This document explains why Co-Authored-By trailers are forbidden."
    trailer = "Subject\n\nCo-Authored-By: Other <other@example.com>\n"
    assert COAUTHOR_TRAILER.search(prose) is None
    assert COAUTHOR_TRAILER.search(trailer) is not None


def test_scan_text_allows_portable_userprofile_placeholder() -> None:
    findings = scan_text(
        Path("docs/environment.md"),
        r"Cache: %USERPROFILE%\.cache\huggingface",
    )
    assert findings == []


def test_identity_audit_detects_other_contributor_and_coauthor(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", EXPECTED_NAME)
    _git(tmp_path, "config", "user.email", EXPECTED_EMAIL)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("first\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "expected identity")

    clean = audit_identities(tmp_path)
    assert clean["invalid_rows"] == []
    assert clean["coauthor_trailer_count"] == 0

    _git(tmp_path, "config", "user.name", "Other Person")
    _git(tmp_path, "config", "user.email", "other@example.com")
    tracked.write_text("second\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(
        tmp_path,
        "commit",
        "-m",
        "invalid identity",
        "-m",
        "Co-Authored-By: Another Person <another@example.com>",
    )

    rejected = audit_identities(tmp_path)
    assert rejected["invalid_rows"] == [
        ["Other Person", "other@example.com", "Other Person", "other@example.com"]
    ]
    assert rejected["coauthor_trailer_count"] == 1
