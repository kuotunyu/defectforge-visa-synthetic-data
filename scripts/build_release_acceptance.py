"""Write the one-page M24 acceptance report after every other local gate passes."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_publish import build_audit  # isort: skip
from src.common.integrity import sha256_file  # isort: skip

ACCEPTANCE_PATH = "reports/release_acceptance.md"
SELF_CHECKS = {"release_acceptance_complete", "required_paths"}


class ReleaseAcceptanceError(RuntimeError):
    """Raised when a one-page acceptance report would conceal an open gate."""


def release_blockers(audit: Mapping[str, Any]) -> list[str]:
    checks = audit.get("checks")
    required = audit.get("required")
    if not isinstance(checks, dict) or not isinstance(required, dict):
        return ["M24 audit schema is invalid"]
    blockers = [
        f"check failed: {name}"
        for name, passed in sorted(checks.items())
        if name not in SELF_CHECKS and passed is not True
    ]
    missing = required.get("missing")
    if not isinstance(missing, list):
        blockers.append("required.missing is invalid")
    else:
        blockers.extend(
            f"required path missing: {path}" for path in missing if path != ACCEPTANCE_PATH
        )
    return blockers


def render_acceptance(
    audit: Mapping[str, Any],
    *,
    classification_sha256: str,
    segmentation_sha256: str,
) -> str:
    checks = audit["checks"]
    passed = [
        name for name, value in sorted(checks.items()) if name not in SELF_CHECKS and value is True
    ]
    lines = [
        "# DefectForge Pre-publication Acceptance",
        "",
        "> Local acceptance only. This report does not publish, upload, create a repository,",
        "> or change visibility. External actions remain gated on explicit user approval.",
        "",
        "## Passed",
        "",
        *[f"- [x] `{name}`" for name in passed],
        f"- [x] Classification CSV SHA256 `{classification_sha256}`",
        f"- [x] Segmentation CSV SHA256 `{segmentation_sha256}`",
        "",
        "## Fixed",
        "",
        "- [x] Frozen evidence bytes survive Windows, Linux, clone, and GitHub Source ZIP.",
        "- [x] M18 source packaging works without `.git` and excludes outputs and secrets.",
        "- [x] Phase 1 handoff verification accepts completed M18 without weakening M15.",
        "- [x] M24 gate binds final reports, figures, README, demo, licenses, and HF dry-run.",
        "",
        "## Residual risks",
        "",
        "- [x] Scope is limited to VisA `pcb1` and `capsules` with ten defect seeds each.",
        "- [x] Pseudo-types are not official VisA per-image defect labels.",
        "- [x] Negative or null synthetic-data outcomes are retained in README Limitations.",
        "- [x] GitHub/Hugging Face writes and public visibility still require user approval.",
        "",
    ]
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(ACCEPTANCE_PATH),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo.resolve(strict=True)
    audit = build_audit(repo)
    blockers = release_blockers(audit)
    if blockers:
        raise ReleaseAcceptanceError(
            "Refusing to write acceptance report:\n- " + "\n- ".join(blockers)
        )
    classification = repo / "results" / "classification.csv"
    segmentation = repo / "results" / "segmentation.csv"
    text = render_acceptance(
        audit,
        classification_sha256=sha256_file(classification),
        segmentation_sha256=sha256_file(segmentation),
    )
    output = args.output if args.output.is_absolute() else repo / args.output
    atomic_write(output, text)
    print(f"Wrote local pre-publication acceptance: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
