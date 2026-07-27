from __future__ import annotations

import pytest

from scripts.verify_license_chain import END, START, LicenseChainError, extract


def test_extract_returns_marked_table() -> None:
    text = f"before\n{START}\n| A | B |\n{END}\nafter\n"
    assert extract(text, label="fixture") == "| A | B |"


def test_extract_rejects_duplicate_marker() -> None:
    text = f"{START}\n{START}\nvalue\n{END}\n"
    with pytest.raises(LicenseChainError, match="one license start"):
        extract(text, label="fixture")
