"""Plan or explicitly upload prevalidated DefectForge bundles to private HF repos."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_publish import scan_text  # isort: skip
from src.common.integrity import sha256_file  # isort: skip
from src.common.paths import load_paths  # isort: skip

REPOSITORIES = {
    "dataset": "steven0226/defectforge-visa-synthetic",
    "model": "steven0226/defectforge-visa-lora",
}
EXPECTED_HF_ACCOUNT = "steven0226"
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}


class HuggingFaceUploadError(RuntimeError):
    """Raised when a planned HF upload violates the publication contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HuggingFaceUploadError(message)


def inventory(root: Path, *, repo_type: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    require(root.is_dir(), f"Upload root is not a directory: {root}")
    card = root / "README.md"
    require(card.is_file(), f"{repo_type} bundle has no README.md")
    card_text = card.read_text(encoding="utf-8")
    require("TBD" not in card_text and "TODO" not in card_text, f"{repo_type} card is unfinished")
    require(not scan_text(Path("README.md"), card_text), f"{repo_type} card failed secret scan")

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    image_count = 0
    mask_count = 0
    safetensor_count = 0
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"Upload bundle contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        lower_parts = {part.lower() for part in Path(relative).parts}
        if repo_type == "dataset":
            require(
                not ({"raw", "prepared", "visa_raw"} & lower_parts),
                f"Dataset bundle may contain a real-data directory: {relative}",
            )
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            require(
                not scan_text(Path(relative), text),
                f"Upload text failed secret/path scan: {relative}",
            )
        size = path.stat().st_size
        total_bytes += size
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            image_count += 1
            if "mask" in lower_parts or "masks" in lower_parts:
                mask_count += 1
        if path.suffix.lower() == ".safetensors":
            safetensor_count += 1
        entries.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    require(bool(entries), f"{repo_type} bundle is empty")
    if repo_type == "dataset":
        require(image_count > 0 and mask_count > 0, "Dataset bundle needs images and masks")
    else:
        require(safetensor_count > 0, "Model bundle needs SafeTensors weights")
    return {
        "repo_type": repo_type,
        "repo_id": REPOSITORIES[repo_type],
        "root": str(root),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "image_count": image_count,
        "mask_count": mask_count,
        "safetensor_count": safetensor_count,
        "files": entries,
    }


def default_roots(paths_config: Path) -> dict[str, Path]:
    paths = load_paths(paths_config)
    publish_root = paths.data_root / "publish"
    return {
        "dataset": publish_root / "hf_dataset",
        "model": publish_root / "hf_model",
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _selected_targets(value: str) -> tuple[str, ...]:
    return ("dataset", "model") if value == "both" else (value,)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--target", choices=("dataset", "model", "both"), default="both")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument(
        "--plan-out",
        type=Path,
        default=Path("reports/hf_upload_plan.json"),
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Create/update private repos and upload; omitted means local dry-run only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = default_roots(args.paths)
    if args.dataset_root is not None:
        roots["dataset"] = args.dataset_root
    if args.model_root is not None:
        roots["model"] = args.model_root
    targets = _selected_targets(args.target)
    bundles = {target: inventory(roots[target], repo_type=target) for target in targets}
    plan = {
        "schema_version": 1,
        "status": "passed",
        "mode": "confirmed_private_upload" if args.confirm else "dry_run",
        "changes_visibility": False,
        "repositories_created_private": True,
        "bundles": bundles,
    }
    atomic_write_json(args.plan_out, plan)
    if not args.confirm:
        print(f"HF dry-run passed; no network write performed: {args.plan_out}")
        return 0

    token = os.environ.get("HF_TOKEN")
    require(bool(token), "Set HF_TOKEN in the environment before --confirm")
    api = HfApi(token=token)
    identity = api.whoami(token=token)
    require(identity.get("name") == EXPECTED_HF_ACCOUNT, "HF_TOKEN belongs to the wrong account")
    for target in targets:
        bundle = bundles[target]
        repo_id = REPOSITORIES[target]
        api.create_repo(
            repo_id=repo_id,
            repo_type=target,
            private=True,
            exist_ok=True,
            token=token,
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type=target,
            folder_path=bundle["root"],
            commit_message=f"Publish verified DefectForge {target} bundle",
            token=token,
        )
        api.auth_check(repo_id, repo_type=target, token=token, write=True)
    print("Private HF upload complete. Repository visibility was not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
