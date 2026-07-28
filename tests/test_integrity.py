from __future__ import annotations

from src.common.integrity import write_text_lf


def test_write_text_lf_is_platform_independent(tmp_path) -> None:
    target = tmp_path / "evidence.txt"
    write_text_lf(target, "first\r\nsecond\rthird\n")
    assert target.read_bytes() == b"first\nsecond\nthird\n"
