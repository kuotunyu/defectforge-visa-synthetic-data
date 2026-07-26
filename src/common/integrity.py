"""Integrity helpers shared by dataset preparation and leakage guards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

BUFFER_BYTES = 8 * 1024 * 1024


class IntegrityError(RuntimeError):
    """A frozen artifact or test-data guard failed validation."""


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject non-object roots."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegrityError(f"Expected a JSON object in {path}")
    return value


def read_checksum_file(path: Path) -> tuple[str, str]:
    """Read one sha256sum-compatible ``<digest>  <name>`` line."""

    parts = path.read_text(encoding="utf-8").strip().split(maxsplit=1)
    if len(parts) != 2 or len(parts[0]) != 64:
        raise IntegrityError(f"Malformed checksum file: {path}")
    return parts[0].lower(), parts[1].strip()


def verify_frozen_manifest(splits_dir: Path) -> tuple[dict[str, Any], str]:
    """Load the manifest only after its frozen checksum has been verified."""

    manifest_path = splits_dir / "split_manifest.json"
    checksum_path = splits_dir / "MANIFEST.sha256"
    expected, filename = read_checksum_file(checksum_path)
    if filename != manifest_path.name:
        raise IntegrityError(
            f"{checksum_path} names {filename!r}, expected {manifest_path.name!r}"
        )
    actual = sha256_file(manifest_path)
    if actual != expected:
        raise IntegrityError(
            f"Frozen manifest checksum mismatch: expected {expected}, observed {actual}"
        )
    return load_json(manifest_path), actual


def assert_not_blocklisted(files: list[Path], blocklist_path: Path) -> None:
    """Raise if any supplied file's content is present in the test blocklist."""

    blocklist = load_json(blocklist_path)
    blocked = set(blocklist.get("sha256", []))
    for path in files:
        digest = sha256_file(path)
        if digest in blocked:
            raise IntegrityError(f"Test-data blocklist hit: {path} ({digest})")
