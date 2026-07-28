"""Build a verified Hugging Face Space folder from formal M16/M18 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file
from src.common.paths import Paths, load_paths
from src.inference.demo_gradio import SelectedCheckpoint, select_object_checkpoints
from src.training.train_classifier import load_config as load_classifier_config
from src.training.train_segmenter import load_config as load_segmenter_config

PACKAGE_MARKER = "package_manifest.json"
TEXT_SUFFIXES = {".json", ".md", ".py", ".txt", ".yaml", ".yml"}


class SpacePackageError(RuntimeError):
    """Raised when Space staging is unsafe or incomplete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SpacePackageError(message)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_state(project_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(len(commit) == 40, "Git HEAD is not a full commit SHA")
    return commit, bool(status)


def _copy_file(source: Path, target: Path) -> None:
    require(source.is_file(), f"Missing formal artifact: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _portable(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return PurePosixPath(*relative.parts).as_posix()


def _base_record(
    selection: SelectedCheckpoint,
    *,
    weight_target: Path,
    package_root: Path,
) -> dict[str, Any]:
    weight = selection.checkpoint.weight
    return {
        "path": _portable(weight_target, package_root),
        "bytes": weight.stat().st_size,
        "sha256": selection.checkpoint.model_sha256,
        "run_name": selection.run_name,
        "group_name": selection.group_name,
        "primary_metric": selection.primary_metric,
        "primary_value": selection.primary_value,
        "secondary_metric": selection.secondary_metric,
        "secondary_value": selection.secondary_value,
        "training_report_sha256": sha256_file(selection.checkpoint.report),
    }


def _package_classifier(
    selection: SelectedCheckpoint,
    *,
    object_name: str,
    package_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    target_root = package_root / "models" / object_name / "classifier"
    weight_target = target_root / "model.safetensors"
    _copy_file(selection.checkpoint.weight, weight_target)
    _copy_file(selection.checkpoint.report, target_root / "training_report.json")
    source_root = selection.checkpoint.weight.parent
    for name in ("run_config.json", "data_manifest.json"):
        _copy_file(source_root / name, target_root / name)
    model = config["model"]
    return {
        **_base_record(selection, weight_target=weight_target, package_root=package_root),
        "architecture": str(model["name"]),
        "repository": str(model["repository"]),
        "revision": str(model["revision"]),
        "base_sha256": str(model["sha256"]),
        "base_license": "apache-2.0",
        "input_size": int(model["input_size"]),
        "num_classes": int(model["num_classes"]),
        "mean": [float(value) for value in model["mean"]],
        "std": [float(value) for value in model["std"]],
    }


def _package_segmenter(
    selection: SelectedCheckpoint,
    *,
    object_name: str,
    package_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    target_root = package_root / "models" / object_name / "segmenter"
    source_root = selection.checkpoint.weight.parent
    weight_target = target_root / "model.safetensors"
    _copy_file(selection.checkpoint.weight, weight_target)
    _copy_file(source_root / "config.json", target_root / "config.json")
    run_root = source_root.parent
    _copy_file(selection.checkpoint.report, target_root / "training_report.json")
    for name in ("run_config.json", "data_manifest.json"):
        _copy_file(run_root / name, target_root / name)
    model = config["model"]
    return {
        **_base_record(selection, weight_target=weight_target, package_root=package_root),
        "architecture": "SegformerForSemanticSegmentation",
        "repository": str(model["repository"]),
        "revision": str(model["revision"]),
        "base_sha256": str(model["sha256"]),
        "base_license": "NVIDIA Source Code License for SegFormer",
        "input_size": int(model["input_size"]),
        "num_labels": int(model["num_labels"]),
        "mean": [float(value) for value in model["mean"]],
        "std": [float(value) for value in model["std"]],
        "formal_threshold": float(config["metrics"]["threshold"]),
    }


def _scan_text(package_root: Path) -> list[str]:
    findings: list[str] = []
    forbidden = (
        "C:\\Users\\3Hml",
        "D:\\sdg-data",
        "D:/sdg-data",
        "HF_TOKEN=",
        "hf_",
    )
    for path in package_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                findings.append(f"{_portable(path, package_root)}:{needle}")
    return findings


def inventory(package_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in package_root.rglob("*") if item.is_file()):
        if path.name == PACKAGE_MARKER:
            continue
        relative = _portable(path, package_root)
        require(".." not in PurePosixPath(relative).parts, f"Unsafe package path: {relative}")
        require(not path.is_symlink(), f"Symlink is not allowed in Space package: {relative}")
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def build_package(
    *,
    paths: Paths,
    output: Path,
    allow_dirty: bool,
    replace: bool,
) -> dict[str, Any]:
    output = output.resolve(strict=False)
    require(output != paths.project_root, "Package output cannot replace the project root")
    require(not output.is_relative_to(paths.project_root), "Package output must stay outside Git")
    commit, dirty = git_state(paths.project_root)
    require(allow_dirty or not dirty, "Git worktree is dirty; commit first or pass --allow-dirty")
    template = paths.project_root / "deploy" / "hf_space"
    require((template / "app.py").is_file(), "Space template is incomplete")
    if output.exists():
        require(replace, f"Package output already exists: {output}")
        marker = output / PACKAGE_MARKER
        require(marker.is_file(), "Refusing to replace an unmarked directory")
        existing = json.loads(marker.read_text(encoding="utf-8"))
        require(existing.get("package_kind") == "defectforge-hf-space", "Invalid package marker")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    require(not staging.exists(), "Staging directory collision")
    try:
        shutil.copytree(
            template,
            staging,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "models"),
        )
        classifier_config = load_classifier_config(paths.configs / "classifier.yaml")
        segmenter_config = load_segmenter_config(paths.configs / "segmenter.yaml")
        objects: dict[str, Any] = {}
        for object_name in paths.objects:
            classifier, segmenter = select_object_checkpoints(
                paths=paths,
                object_name=object_name,
                classification_results=paths.project_root / "results/classification.csv",
                segmentation_results=paths.project_root / "results/segmentation.csv",
                segmentation_runs_root=paths.colab_results / "segmentation",
            )
            objects[object_name] = {
                "classifier": _package_classifier(
                    classifier,
                    object_name=object_name,
                    package_root=staging,
                    config=classifier_config,
                ),
                "segmenter": _package_segmenter(
                    segmenter,
                    object_name=object_name,
                    package_root=staging,
                    config=segmenter_config,
                ),
            }
        model_manifest = {
            "schema_version": 1,
            "status": "passed",
            "source_commit": commit,
            "source_dirty": dirty,
            "classification_results_sha256": sha256_file(
                paths.project_root / "results/classification.csv"
            ),
            "segmentation_results_sha256": sha256_file(
                paths.project_root / "results/segmentation.csv"
            ),
            "objects": objects,
        }
        atomic_write_json(staging / "models" / "manifest.json", model_manifest)
        findings = _scan_text(staging)
        require(not findings, f"Sensitive/local path found in package: {findings[:5]}")
        files = inventory(staging)
        require(files, "Space package inventory is empty")
        marker_payload = {
            "schema_version": 1,
            "status": "passed",
            "package_kind": "defectforge-hf-space",
            "source_commit": commit,
            "source_dirty": dirty,
            "file_count": len(files),
            "total_bytes": sum(record["bytes"] for record in files),
            "files": files,
        }
        atomic_write_json(staging / PACKAGE_MARKER, marker_payload)
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
        return marker_payload
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_paths(args.paths)
    output = args.output or paths.data_root / "publish" / "hf_space_demo"
    report = build_package(
        paths=paths,
        output=output,
        allow_dirty=args.allow_dirty,
        replace=args.replace,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output.resolve(strict=False)),
                "source_commit": report["source_commit"],
                "source_dirty": report["source_dirty"],
                "file_count": report["file_count"],
                "total_bytes": report["total_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
