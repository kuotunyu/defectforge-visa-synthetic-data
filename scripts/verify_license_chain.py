"""Verify the exact shared license table in README and both HF cards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file

START = "<!-- BEGIN VERIFIED LICENSE_CHAIN -->"
END = "<!-- END VERIFIED LICENSE_CHAIN -->"
DOCUMENTS = (
    Path("README.md"),
    Path("hf_cards/dataset/README.md"),
    Path("hf_cards/model/README.md"),
)


class LicenseChainError(RuntimeError):
    """Raised when the published license chain diverges."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LicenseChainError(message)


def extract(text: str, *, label: str) -> str:
    require(text.count(START) == 1, f"{label}: expected one license start marker")
    require(text.count(END) == 1, f"{label}: expected one license end marker")
    return text.split(START, maxsplit=1)[1].split(END, maxsplit=1)[0].strip()


def verify(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    source = project_root / "docs" / "license_chain.md"
    require(source.is_file(), "Missing docs/license_chain.md")
    expected = source.read_text(encoding="utf-8").strip()
    hashes: dict[str, str] = {}
    for relative in DOCUMENTS:
        path = project_root / relative
        require(path.is_file(), f"Missing license document: {relative}")
        require(
            extract(path.read_text(encoding="utf-8"), label=relative.as_posix()) == expected,
            f"License chain changed: {relative}",
        )
        hashes[relative.as_posix()] = sha256_file(path)
    verification = project_root / "reports" / "model_license_verification.json"
    require(verification.is_file(), "Missing current upstream model license verification")
    payload = json.loads(verification.read_text(encoding="utf-8"))
    require(payload.get("status") == "passed", "Upstream model license verification failed")
    return {
        "status": "passed",
        "schema_version": 1,
        "source_sha256": sha256_file(source),
        "document_sha256": hashes,
        "upstream_verification_sha256": sha256_file(verification),
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/license_chain_validation.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = verify(args.project_root)
    atomic_write_json(args.output, payload)
    print(f"Verified shared license chain: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
