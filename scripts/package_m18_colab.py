"""Build a minimal source archive and per-object M18 segmentation data archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file
from src.common.paths import Paths, load_paths
from src.training.segmenter_data import (
    SegmentationGroup,
    build_segmentation_group,
    group_payload_sha256,
    resolve_image_path,
    resolve_mask_path,
)

FORMAL_GROUPS = (
    "real_only",
    "std_aug",
    "unfiltered_syn",
    "filtered_syn",
    "full_real",
    "procedural_only",
    "copypaste_only",
    "diffusion_only",
)
SYNTHETIC_GROUPS = (
    "unfiltered_syn",
    "filtered_syn",
    "procedural_only",
    "copypaste_only",
    "diffusion_only",
)
REQUIRED_UNTRACKED = (
    Path("configs/segmenter.yaml"),
    Path("notebooks/02_train_segformer.ipynb"),
    Path("scripts/package_m18_colab.py"),
    Path("scripts/validate_segmenter_runs.py"),
    Path("scripts/validate_m18_colab_notebook.py"),
    Path("src/training/segmenter_data.py"),
    Path("src/training/train_segmenter.py"),
)
SOURCE_ROOT_FILES = (
    Path("LICENSE"),
    Path("README.md"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)
SOURCE_DIRECTORIES = (
    Path("configs"),
    Path("notebooks"),
    Path("scripts"),
    Path("splits"),
    Path("src"),
)
SOURCE_FILE_LIMIT_BYTES = 10 * 1024 * 1024


class M18PackageError(RuntimeError):
    """Raised when a Colab package cannot preserve the M18 data contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M18PackageError(message)


def _is_source_noise(path: Path) -> bool:
    return (
        path.name == ".gitkeep"
        or "__pycache__" in path.parts
        or path.suffix.lower() in {".pyc", ".pyo"}
    )


def source_files(project_root: Path = PROJECT_ROOT) -> list[Path]:
    """Return the deterministic, secret-safe source set required by M18 Colab.

    A fixed allowlist keeps the bundle reproducible in a Git checkout, a GitHub
    source archive, or any ordinary extracted directory. It also prevents
    unrelated reports, results, virtual environments, or model weights from
    entering the upload bundle.
    """

    relative_files = list(SOURCE_ROOT_FILES)
    for relative_directory in SOURCE_DIRECTORIES:
        directory = project_root / relative_directory
        require(directory.is_dir(), f"Source directory is missing: {relative_directory}")
        relative_files.extend(
            path.relative_to(project_root)
            for path in directory.rglob("*")
            if path.is_file() and not _is_source_noise(path.relative_to(project_root))
        )
    for relative in REQUIRED_UNTRACKED:
        if relative not in relative_files:
            relative_files.append(relative)

    files = sorted({project_root / relative for relative in relative_files})
    require(all(path.is_file() for path in files), "Source archive contains a missing file")
    require(not any(path.is_symlink() for path in files), "Source archive would contain a symlink")
    require(
        not any(
            ".env" in path.name.lower()
            or "secret" in path.name.lower()
            or "credential" in path.name.lower()
            or ".git" in path.parts
            for path in files
        ),
        "Source archive would contain a secret or Git internals",
    )
    require(
        all(path.stat().st_size <= SOURCE_FILE_LIMIT_BYTES for path in files),
        f"Source archive contains a file larger than {SOURCE_FILE_LIMIT_BYTES} bytes",
    )
    return files


def _metadata_index(root: Path) -> dict[str, Mapping[str, Any]]:
    metadata_path = root / "metadata.jsonl"
    require(metadata_path.is_file(), f"Synthetic metadata is missing: {metadata_path}")
    index: dict[str, Mapping[str, Any]] = {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            require(
                isinstance(record, dict), f"Invalid metadata line {metadata_path}:{line_number}"
            )
            image_path = str(record["image_path"]).replace("\\", "/")
            require(image_path not in index, f"Duplicate synthetic image path: {image_path}")
            index[image_path] = record
    return index


def _safe_pool_id(view: str, sample_id: str) -> str:
    raw = f"{view.replace('/', '__')}__{sample_id}"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    require(bool(safe) and safe not in {".", ".."}, f"Invalid pooled sample id: {raw}")
    return safe


def _build_groups(
    paths: Paths,
    config: Mapping[str, Any],
    object_name: str,
) -> dict[str, SegmentationGroup]:
    return {
        name: build_segmentation_group(
            paths,
            config,
            group_name=name,
            object_name=object_name,
            seed=paths.seed,
            mode="final",
        )
        for name in FORMAL_GROUPS
    }


def _real_members(
    paths: Paths,
    group: SegmentationGroup,
) -> list[tuple[Path, str]]:
    members: dict[str, Path] = {}
    for sample in (*group.train, *group.test):
        require(sample.kind == "real", "full_real unexpectedly contains synthetic data")
        image = resolve_image_path(paths, sample)
        image_name = (
            Path("01-defectforge/raw/VisA") / image.relative_to(paths.visa_raw)
        ).as_posix()
        members[image_name] = image
        mask = resolve_mask_path(paths, sample)
        if mask is not None:
            mask_name = (
                Path("01-defectforge/raw/VisA") / mask.relative_to(paths.visa_raw)
            ).as_posix()
            members[mask_name] = mask
    return [(source, name) for name, source in sorted(members.items())]


def _synthetic_pool(
    paths: Paths,
    groups: Mapping[str, SegmentationGroup],
) -> tuple[
    list[tuple[Path, str]],
    list[dict[str, Any]],
    dict[str, list[str]],
]:
    indexes: dict[str, dict[str, Mapping[str, Any]]] = {}
    members: dict[str, Path] = {}
    pooled_records: dict[str, dict[str, Any]] = {}
    selections: dict[str, list[str]] = {}

    for group_name in SYNTHETIC_GROUPS:
        selected_ids: list[str] = []
        synthetic_samples = [
            sample for sample in groups[group_name].train if sample.kind == "synthetic"
        ]
        require(len(synthetic_samples) == 500, f"{group_name} does not contain 500 synthetic masks")
        for sample in synthetic_samples:
            view = sample.root.removeprefix("synthetic/")
            if view not in indexes:
                indexes[view] = _metadata_index(paths.synthetic / view)
            source_record = indexes[view].get(sample.image_path)
            require(
                source_record is not None, f"Missing source metadata: {view}/{sample.image_path}"
            )
            original_id = str(source_record["sample_id"])
            pool_id = _safe_pool_id(view, original_id)
            selected_ids.append(pool_id)
            if pool_id in pooled_records:
                existing = pooled_records[pool_id]
                require(
                    existing["original_view"] == view
                    and existing["original_sample_id"] == original_id,
                    f"Pooled sample ID collision: {pool_id}",
                )
                continue

            image = resolve_image_path(paths, sample)
            mask = resolve_mask_path(paths, sample)
            require(mask is not None, f"Synthetic sample has no mask: {sample.sample_id}")
            image_relative = f"images/{pool_id}{image.suffix.lower()}"
            mask_relative = f"masks/{pool_id}{mask.suffix.lower()}"
            archive_root = Path("01-defectforge/synthetic/m18_colab_pool")
            image_name = (archive_root / image_relative).as_posix()
            mask_name = (archive_root / mask_relative).as_posix()
            members[image_name] = image
            members[mask_name] = mask

            rewritten = dict(source_record)
            rewritten["sample_id"] = pool_id
            rewritten["original_sample_id"] = original_id
            rewritten["original_view"] = view
            rewritten["image_path"] = image_relative
            rewritten["mask_path"] = mask_relative
            pooled_records[pool_id] = rewritten
        require(len(set(selected_ids)) == 500, f"{group_name} packaged duplicate samples")
        selections[group_name] = sorted(selected_ids)

    return (
        [(source, name) for name, source in sorted(members.items())],
        [pooled_records[key] for key in sorted(pooled_records)],
        selections,
    )


def write_zip(path: Path, members: Sequence[tuple[Path, str]]) -> None:
    require(not path.exists(), f"Refusing to overwrite archive: {path}")
    names = [name for _, name in members]
    require(len(names) == len(set(names)), f"Archive contains duplicate members: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for source, archive_name in members:
            archive.write(source, archive_name)


def _write_generated_files(
    root: Path,
    *,
    object_name: str,
    pooled_records: Sequence[Mapping[str, Any]],
    selections: Mapping[str, Sequence[str]],
    groups: Mapping[str, SegmentationGroup],
) -> tuple[Path, Path, dict[str, Any]]:
    metadata = root / "metadata.jsonl"
    metadata.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in pooled_records
        ),
        encoding="utf-8",
    )
    selection_payload = {
        "schema_version": 1,
        "pipeline_version": "0.1.0",
        "object": object_name,
        "groups": {name: list(values) for name, values in selections.items()},
        "group_payload_sha256": {
            name: group_payload_sha256(group) for name, group in groups.items()
        },
        "pooled_records": len(pooled_records),
        "all_mixed_alias_of": "filtered_syn",
    }
    selection = root / "m18_colab_selection.json"
    selection.write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata, selection, selection_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/segmenter.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--object", dest="objects", action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = load_paths(args.paths)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "Invalid segmenter config")
    output_dir = (args.output_dir or paths.data_root / "colab" / "m18").resolve(strict=False)
    objects = tuple(args.objects or paths.objects)
    require(set(objects) <= set(paths.objects), f"Unsupported objects: {objects}")

    source_zip = output_dir / "defectforge_m18_source.zip"
    source_members = [
        (path, (Path("defectforge") / path.relative_to(PROJECT_ROOT)).as_posix())
        for path in source_files()
    ]
    if not args.dry_run:
        write_zip(source_zip, source_members)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_archive": {
            "file": source_zip.name,
            "files": len(source_members),
            "uncompressed_bytes": sum(path.stat().st_size for path, _ in source_members),
        },
        "data_archives": {},
        "formal_groups_per_object": len(FORMAL_GROUPS),
        "all_mixed_alias_of": "filtered_syn",
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        manifest["source_archive"].update(
            {
                "bytes": source_zip.stat().st_size,
                "sha256": sha256_file(source_zip),
            }
        )

    for object_name in objects:
        groups = _build_groups(paths, config, object_name)
        real_members = _real_members(paths, groups["full_real"])
        synthetic_members, pooled_records, selections = _synthetic_pool(paths, groups)
        with tempfile.TemporaryDirectory(prefix=f"defectforge-m18-{object_name}-") as temporary:
            generated_root = Path(temporary)
            metadata, selection, selection_payload = _write_generated_files(
                generated_root,
                object_name=object_name,
                pooled_records=pooled_records,
                selections=selections,
                groups=groups,
            )
            generated_members = [
                (
                    metadata,
                    "01-defectforge/synthetic/m18_colab_pool/metadata.jsonl",
                ),
                (
                    selection,
                    "01-defectforge/m18_colab_selection.json",
                ),
            ]
            data_zip = output_dir / f"m18_seg_{object_name}.zip"
            all_members = [*real_members, *synthetic_members, *generated_members]
            unpacked_bytes = sum(path.stat().st_size for path, _ in all_members)
            if not args.dry_run:
                write_zip(data_zip, all_members)
        archive_manifest = {
            "file": data_zip.name,
            "files": len(real_members) + len(synthetic_members) + 2,
            "uncompressed_bytes": unpacked_bytes,
            "real_files": len(real_members),
            "pooled_synthetic_records": len(pooled_records),
            "selection_sha256": canonical_sha256(selection_payload),
            "training_blocklist_hits": 0,
            "contains_frozen_test_for_evaluation": True,
        }
        if not args.dry_run:
            archive_manifest.update(
                {
                    "bytes": data_zip.stat().st_size,
                    "sha256": sha256_file(data_zip),
                }
            )
        manifest["data_archives"][object_name] = archive_manifest

    if not args.dry_run:
        manifest_path = output_dir / "m18_colab_bundle.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["manifest"] = str(manifest_path)
    print(json.dumps(manifest, indent=2))
    return 0


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
