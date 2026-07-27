"""Build minimal, checksummed source and data archives for the M11 Colab notebook."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import assert_not_blocklisted, sha256_file
from src.common.paths import load_paths
from src.training.train_inpaint_lora import load_training_samples


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    relatives = [
        Path(raw.decode("utf-8"))
        for raw in result.stdout.split(b"\0")
        if raw
    ]
    required_untracked = (
        Path("configs/lora_sdxl.yaml"),
        Path("notebooks/01_train_inpaint_lora_sdxl.ipynb"),
        Path("scripts/package_m11_colab.py"),
        Path("scripts/validate_colab_notebook.py"),
    )
    for relative in required_untracked:
        if relative not in relatives:
            relatives.append(relative)
    files = sorted(PROJECT_ROOT / relative for relative in relatives)
    if any(not path.is_file() for path in files):
        raise RuntimeError("Source archive contains a missing tracked file")
    if any(".env" in path.name.lower() or ".git" in path.parts for path in files):
        raise RuntimeError("Source archive would include a secret or Git internals")
    return files


def heldout_files(paths: Any, object_name: str, trigger_tokens: set[str]) -> list[Path]:
    placement_root = paths.synthetic / "placements" / object_name
    metadata = placement_root / "placements.jsonl"
    selected: dict[str, dict[str, Any]] = {}
    with metadata.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            token = str(record["trigger_token"])
            if token in trigger_tokens and token not in selected:
                selected[token] = record
            if selected.keys() == trigger_tokens:
                break
    if selected.keys() != trigger_tokens:
        raise RuntimeError(f"Missing held-out placement types for {object_name}")
    result = [metadata]
    for record in selected.values():
        result.extend(
            [
                paths.visa_raw / str(record["background_image"]),
                placement_root / str(record["mask_path"]),
            ]
        )
    return result


def data_files(paths: Any) -> list[Path]:
    files: set[Path] = set()
    blocklist_inputs: set[Path] = set()
    for object_name in paths.objects:
        samples, *_ = load_training_samples(paths, object_name)
        trigger_tokens = {sample.trigger_token for sample in samples}
        for sample in samples:
            blocklist_inputs.update(
                {
                    paths.visa_raw / sample.image_path,
                    paths.visa_raw / sample.mask_path,
                }
            )
        files.update(blocklist_inputs)
        heldout = heldout_files(paths, object_name, trigger_tokens)
        files.update(heldout)
        blocklist_inputs.update(
            path for path in heldout if path.suffix.lower() in {".jpg", ".png"}
        )
    assert_not_blocklisted(
        sorted(blocklist_inputs),
        paths.splits / "test_blocklist.json",
    )
    return sorted(files)


def write_zip(path: Path, members: list[tuple[Path, str]]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite archive: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for source, archive_name in members:
            archive.write(source, archive_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    paths = load_paths(args.paths)
    output_dir = args.output_dir or paths.data_root / "colab" / "m11"
    source_zip = output_dir / "defectforge_source.zip"
    data_zip = output_dir / "m11_sdxl_inputs.zip"
    source_members = [
        (path, (Path("defectforge") / path.relative_to(PROJECT_ROOT)).as_posix())
        for path in tracked_files()
    ]
    selected_data = data_files(paths)
    data_members: list[tuple[Path, str]] = []
    for path in selected_data:
        if path.is_relative_to(paths.visa_raw):
            relative = Path("01-defectforge/raw/VisA") / path.relative_to(paths.visa_raw)
        elif path.is_relative_to(paths.synthetic):
            relative = Path("01-defectforge/synthetic") / path.relative_to(paths.synthetic)
        else:
            raise RuntimeError(f"Data file is outside permitted roots: {path}")
        data_members.append((path, relative.as_posix()))
    write_zip(source_zip, source_members)
    write_zip(data_zip, data_members)
    manifest = {
        "schema_version": 1,
        "source_archive": {
            "file": source_zip.name,
            "files": len(source_members),
            "bytes": source_zip.stat().st_size,
            "sha256": sha256_file(source_zip),
        },
        "data_archive": {
            "file": data_zip.name,
            "files": len(data_members),
            "bytes": data_zip.stat().st_size,
            "sha256": sha256_file(data_zip),
        },
        "test_blocklist_hits": 0,
    }
    manifest_path = output_dir / "m11_colab_bundle.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
