"""Download, verify, safely extract, and inventory the VisA dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import Paths, load_paths  # isort: skip


VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
EXPECTED_TAR_BYTES = 1_929_840_640
EXPECTED_COUNTS = {
    "pcb1": {"normal": 1_004, "anomaly": 100},
    "capsules": {"normal": 602, "anomaly": 100},
}
BUFFER_BYTES = 8 * 1024 * 1024

LOGGER = logging.getLogger("download_visa")


class ValidationError(RuntimeError):
    """An M2 assertion failed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Verify and extract an existing tar without accessing the network.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def download_with_resume(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "DefectForge/0.1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    LOGGER.info("Downloading %s to %s (resume offset=%d)", url, partial, existing)
    request = Request(url, headers=headers)
    try:
        response = urlopen(request, timeout=60)
    except HTTPError as error:
        if error.code == 416 and existing == EXPECTED_TAR_BYTES:
            partial.replace(destination)
            return
        raise

    status = getattr(response, "status", response.getcode())
    if existing and status != 206:
        LOGGER.warning("Server ignored Range; restarting partial download")
        existing = 0
        partial.unlink(missing_ok=True)

    mode = "ab" if existing else "wb"
    downloaded = existing
    last_report = time.monotonic()
    with response, partial.open(mode) as handle:
        while block := response.read(BUFFER_BYTES):
            handle.write(block)
            downloaded += len(block)
            now = time.monotonic()
            if now - last_report >= 15:
                LOGGER.info("Downloaded %.1f%%", downloaded / EXPECTED_TAR_BYTES * 100)
                last_report = now

    if downloaded != EXPECTED_TAR_BYTES:
        raise ValidationError(
            f"Downloaded file has {downloaded} bytes; expected {EXPECTED_TAR_BYTES}"
        )
    partial.replace(destination)


def assert_expected_size(path: Path) -> None:
    if not path.is_file():
        raise ValidationError(f"VisA tar does not exist: {path}")
    actual = path.stat().st_size
    if actual != EXPECTED_TAR_BYTES:
        raise ValidationError(
            f"VisA tar has {actual} bytes; expected exactly {EXPECTED_TAR_BYTES}"
        )


def safe_extract(tar_path: Path, raw_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    root = raw_root.resolve()
    with tarfile.open(tar_path, mode="r:") as archive:
        members = archive.getmembers()
        for member in members:
            resolved = (root / member.name).resolve(strict=False)
            if root != resolved and root not in resolved.parents:
                raise ValidationError(f"Unsafe tar member path: {member.name!r}")
        LOGGER.info("Extracting %d members into %s", len(members), raw_root)
        archive.extractall(raw_root, members=members, filter="data")


def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        raise ValidationError(f"Expected image directory does not exist: {folder}")
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def validate_inventory(visa_root: Path) -> dict[str, dict[str, int]]:
    observed: dict[str, dict[str, int]] = {}
    for object_name, expected in EXPECTED_COUNTS.items():
        object_root = visa_root / object_name / "Data"
        normal = _count_images(object_root / "Images" / "Normal")
        anomaly = _count_images(object_root / "Images" / "Anomaly")
        masks = _count_images(object_root / "Masks" / "Anomaly")
        observed[object_name] = {
            "normal": normal,
            "anomaly": anomaly,
            "masks": masks,
        }
        if normal != expected["normal"] or anomaly != expected["anomaly"]:
            raise ValidationError(
                f"{object_name} inventory mismatch: observed {observed[object_name]}, "
                f"expected normal={expected['normal']} anomaly={expected['anomaly']}"
            )
        if masks != anomaly:
            raise ValidationError(
                f"{object_name} has {anomaly} anomaly images but {masks} masks"
            )
    return observed


def write_checksum_manifest(paths: Paths, digest: str) -> Path:
    output = paths.splits / "source_checksums.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        paths.visa_tar.name: {
            "bytes": EXPECTED_TAR_BYTES,
            "sha256": digest,
            "downloaded_at": datetime.now(UTC).isoformat(),
            "url": VISA_URL,
        }
    }
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


def append_log(paths: Paths, *, started: float, inventory: dict[str, object]) -> None:
    log_path = paths.project_root / "logs" / "download_visa.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "tar_bytes": paths.visa_tar.stat().st_size,
        "inventory": inventory,
    }
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    started = time.monotonic()
    try:
        paths = load_paths(args.paths)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "url": VISA_URL,
                        "tar": str(paths.visa_tar),
                        "expected_bytes": EXPECTED_TAR_BYTES,
                        "extract_to": str(paths.visa_raw),
                        "checksum_manifest": str(paths.splits / "source_checksums.json"),
                    },
                    indent=2,
                )
            )
            return 0

        if not args.skip_download and not paths.visa_tar.is_file():
            download_with_resume(VISA_URL, paths.visa_tar)
        assert_expected_size(paths.visa_tar)

        digest = sha256_file(paths.visa_tar)
        LOGGER.info("SHA256 %s", digest)
        # The official tar contains object folders at its root, not a top-level
        # "VisA/" directory, so extraction targets the configured visa_raw folder.
        safe_extract(paths.visa_tar, paths.visa_raw)
        inventory = validate_inventory(paths.visa_raw)
        checksum_path = write_checksum_manifest(paths, digest)
        append_log(paths, started=started, inventory=inventory)
        LOGGER.info("M2 verified; checksum manifest: %s", checksum_path)
        return 0
    except ValidationError as error:
        LOGGER.error("%s", error)
        return 2
    except (OSError, HTTPError, tarfile.TarError, shutil.Error):
        LOGGER.exception("M2 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
