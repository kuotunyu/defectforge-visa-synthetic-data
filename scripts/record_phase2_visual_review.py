"""Record hash-bound M21/M22 visual review after every final image was opened."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_publish import FINAL_FIGURES, audit_final_media, sha256_file  # isort: skip

CONFIRMATION = "REVIEWED-FINAL-MEDIA"


class VisualReviewError(RuntimeError):
    """Raised when a visual-review report would not be supported by current media."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VisualReviewError(message)


def build_review(
    repo: Path,
    *,
    confirmation: str,
    note: str,
) -> dict[str, object]:
    require(
        confirmation == CONFIRMATION,
        f"Pass --confirm {CONFIRMATION} only after opening every final image",
    )
    require(len(note.strip()) >= 20, "Visual review note must describe the observed result")
    media = audit_final_media(repo)
    require(media["all_figures_valid"], "One or more final figures are missing or invalid")
    require(media["demo_gif"]["valid"], "Final demo GIF is missing or invalid")
    targets = (*FINAL_FIGURES, "assets/demo.gif")
    return {
        "status": "passed",
        "schema_version": 1,
        "reviewed_at": datetime.now(UTC).isoformat(),
        "method": "manual full-frame visual inspection",
        "reviewed_sha256": {relative: sha256_file(repo / relative) for relative in targets},
        "checks": {
            "all_media_opened": True,
            "no_clipping_or_overlap": True,
            "labels_and_numbers_legible": True,
            "real_scaling_curve_answers_equivalent_real_count": True,
            "sample_masks_align_with_visible_defects": True,
            "demo_shows_input_probability_mask_heatmap_and_latency": True,
        },
        "note": note.strip(),
    }


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/phase2_visual_review.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo.resolve(strict=True)
    payload = build_review(
        repo,
        confirmation=args.confirm,
        note=args.note,
    )
    output = args.output if args.output.is_absolute() else repo / args.output
    atomic_write_json(output, payload)
    print(f"Recorded hash-bound visual review: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
