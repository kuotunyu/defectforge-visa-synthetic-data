"""Fail-closed Hugging Face Space publisher for a verified DefectForge package."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from huggingface_hub import HfApi, SpaceHardware
from huggingface_hub.errors import RepositoryNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file
from src.common.paths import load_paths

PACKAGE_MARKER = "package_manifest.json"


class SpacePublishError(RuntimeError):
    """Raised when a Space publish action is not explicitly or safely authorized."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SpacePublishError(message)


def validate_package(source: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    marker_path = source / PACKAGE_MARKER
    require(marker_path.is_file(), "Space package marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    require(marker.get("schema_version") == 1, "Unsupported Space package schema")
    require(marker.get("status") == "passed", "Space package did not pass")
    require(marker.get("package_kind") == "defectforge-hf-space", "Wrong package kind")
    require(marker.get("source_dirty") is False, "Public upload requires a clean source commit")
    files = marker.get("files")
    require(isinstance(files, list) and bool(files), "Space package inventory is empty")
    expected_paths = set()
    for record in files:
        require(isinstance(record, dict), "Invalid Space file record")
        relative = Path(str(record["path"]))
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe Space path")
        path = source / relative
        require(path.is_file(), f"Missing Space file: {relative.as_posix()}")
        require(path.stat().st_size == int(record["bytes"]), f"Size changed: {relative}")
        require(sha256_file(path) == record["sha256"], f"SHA256 changed: {relative}")
        expected_paths.add(relative.as_posix())
    observed_paths = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path.name != PACKAGE_MARKER
    }
    require(observed_paths == expected_paths, "Space package inventory changed")
    return marker


def _token(paths_file: Path) -> str:
    paths = load_paths(paths_file)
    token = dotenv_values(paths.dotenv).get("HF_TOKEN")
    require(isinstance(token, str) and bool(token), "HF_TOKEN is unavailable")
    return token


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo-id", default="steven0226/defectforge-visa-demo")
    parser.add_argument("--confirm-upload", action="store_true")
    parser.add_argument("--make-public", action="store_true")
    parser.add_argument("--sleep-time", type=int, default=3600)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require("/" in args.repo_id and not args.repo_id.startswith("/"), "Invalid Space repo ID")
    require(args.sleep_time >= 300, "Space sleep time must be at least 300 seconds")
    marker = validate_package(args.source)
    plan = {
        "status": "dry-run" if not args.confirm_upload else "ready",
        "repo_id": args.repo_id,
        "source": str(args.source.resolve(strict=True)),
        "source_commit": marker["source_commit"],
        "file_count": marker["file_count"],
        "total_bytes": marker["total_bytes"],
        "hardware": "cpu-basic",
        "sleep_time": args.sleep_time,
        "make_public": args.make_public,
    }
    if not args.confirm_upload:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    token = _token(args.paths)
    api = HfApi(token=token)
    identity = api.whoami()
    owner = args.repo_id.split("/", maxsplit=1)[0]
    require(identity.get("name") == owner, "HF identity does not own the target Space")
    try:
        api.repo_info(args.repo_id, repo_type="space", token=token)
        created = False
    except RepositoryNotFoundError:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            private=True,
            space_sdk="gradio",
            space_hardware=SpaceHardware.CPU_BASIC,
            space_sleep_time=args.sleep_time,
            token=token,
        )
        created = True
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=args.source,
        token=token,
        commit_message="Deploy verified DefectForge demo",
        ignore_patterns=[PACKAGE_MARKER],
    )
    runtime = api.request_space_hardware(
        repo_id=args.repo_id,
        hardware=SpaceHardware.CPU_BASIC,
        sleep_time=args.sleep_time,
        token=token,
    )
    if args.make_public:
        api.update_repo_settings(
            repo_id=args.repo_id,
            repo_type="space",
            private=False,
            token=token,
        )
    print(
        json.dumps(
            {
                **plan,
                "status": "uploaded",
                "created": created,
                "commit": commit.oid,
                "runtime_stage": str(runtime.stage),
                "requested_hardware": str(runtime.requested_hardware),
                "url": f"https://huggingface.co/spaces/{args.repo_id}",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
