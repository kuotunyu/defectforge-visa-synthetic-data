from __future__ import annotations

from pathlib import Path

from scripts.verify_publish import COAUTHOR_TRAILER, scan_text


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
