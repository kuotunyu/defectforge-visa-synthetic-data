from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scripts.record_phase2_visual_review import (
    CONFIRMATION,
    VisualReviewError,
    build_review,
)
from scripts.verify_publish import FINAL_FIGURES


def _write_media(root: Path) -> None:
    for relative in FINAL_FIGURES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 360), "white").save(path)
    gif = root / "assets" / "demo.gif"
    gif.parent.mkdir(parents=True)
    frames = [Image.new("RGB", (640, 360), color) for color in ("white", "black")]
    frames[0].save(
        gif,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )


def test_visual_review_requires_explicit_confirmation(tmp_path: Path) -> None:
    _write_media(tmp_path)
    with pytest.raises(VisualReviewError, match="Pass --confirm"):
        build_review(
            tmp_path,
            confirmation="not-reviewed",
            note="I opened every final artifact and checked the full frame.",
        )


def test_visual_review_binds_every_final_media_hash(tmp_path: Path) -> None:
    _write_media(tmp_path)

    review = build_review(
        tmp_path,
        confirmation=CONFIRMATION,
        note="All tables are legible and every overlay aligns with the input.",
    )

    assert review["status"] == "passed"
    assert set(review["reviewed_sha256"]) == {
        *FINAL_FIGURES,
        "assets/demo.gif",
    }
