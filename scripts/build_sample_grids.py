"""Build deterministic real-versus-synthetic M21 grids with mask overlays."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file  # isort: skip
from src.common.paths import load_paths  # isort: skip

OBJECTS = ("pcb1", "capsules")
SOURCE_ROWS = (
    ("real", "Real few-shot defects"),
    ("stageA_copypaste", "Filtered copy-paste"),
    ("stageA_procedural", "Filtered procedural"),
    ("stageB_sd2", "Filtered SD2 searched"),
    ("stageB_sdxl", "SDXL searched"),
)


class SampleGridError(RuntimeError):
    """Raised when the M21 sample-grid evidence is incomplete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SampleGridError(message)


@dataclass(frozen=True, slots=True)
class GridSample:
    sample_id: str
    defect_type: str
    image: Path
    mask: Path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"Missing metadata: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"Invalid metadata row {path}:{line_number}")
        rows.append(value)
    require(bool(rows), f"Empty metadata: {path}")
    return rows


def select_records(
    records: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> list[Mapping[str, Any]]:
    """Round-robin frozen pseudo-types, then stable sample ID."""

    require(count > 0, "Grid sample count must be positive")
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[str(record.get("defect_type", "unknown"))].append(record)
    require(bool(by_type), "No records available for sample grid")
    for values in by_type.values():
        values.sort(key=lambda row: str(row["sample_id"]))
    selected: list[Mapping[str, Any]] = []
    index = 0
    ordered_types = sorted(by_type)
    while len(selected) < count:
        added = False
        for defect_type in ordered_types:
            values = by_type[defect_type]
            if index < len(values):
                selected.append(values[index])
                added = True
                if len(selected) == count:
                    break
        require(added, f"Only {len(selected)} records are available; need {count}")
        index += 1
    return selected


def overlay_mask(image: Image.Image, mask: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    binary = np.asarray(mask.convert("L")) > 0
    require(binary.shape == rgb.shape[:2], "Image/mask dimensions differ")
    overlay = rgb.copy()
    red = np.asarray([239.0, 68.0, 68.0], dtype=np.float32)
    overlay[binary] = 0.62 * rgb[binary] + 0.38 * red
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _real_samples(paths: Any, object_name: str, count: int) -> list[GridSample]:
    selection = json.loads(
        (paths.splits / "fewshot_selection.json").read_text(encoding="utf-8")
    )
    records = selection["objects"][object_name]["fewshot_seed"]
    selected = records[:count]
    require(len(selected) == count, f"Insufficient real few-shot rows: {object_name}")
    return [
        GridSample(
            sample_id=Path(row["image_path"]).stem,
            defect_type="real",
            image=paths.visa_raw / row["image_path"],
            mask=paths.visa_raw / row["mask_path"],
        )
        for row in selected
    ]


def _filtered_samples(
    paths: Any,
    object_name: str,
    *,
    generator: str,
    count: int,
) -> list[GridSample]:
    root = paths.synthetic / "filtered"
    rows = [
        row
        for row in read_jsonl(root / "metadata.jsonl")
        if row["object"] == object_name
        and row["generator"] == generator
        and (generator != "stageB_sd2" or row.get("bucket") == "searched")
    ]
    return [
        GridSample(
            sample_id=str(row["sample_id"]),
            defect_type=str(row["defect_type"]),
            image=root / str(row["image_path"]),
            mask=root / str(row["mask_path"]),
        )
        for row in select_records(rows, count=count)
    ]


def _sdxl_samples(paths: Any, object_name: str, count: int) -> list[GridSample]:
    root = paths.synthetic / "stageB_sdxl" / "searched"
    rows = [
        row
        for row in read_jsonl(root / "metadata.jsonl")
        if row["object"] == object_name and row.get("bucket") == "searched"
    ]
    return [
        GridSample(
            sample_id=str(row["sample_id"]),
            defect_type=str(row["defect_type"]),
            image=root / str(row["image_path"]),
            mask=root / str(row["mask_path"]),
        )
        for row in select_records(rows, count=count)
    ]


def collect(paths: Any, object_name: str, count: int) -> dict[str, list[GridSample]]:
    return {
        "real": _real_samples(paths, object_name, count),
        "stageA_copypaste": _filtered_samples(
            paths,
            object_name,
            generator="stageA_copypaste",
            count=count,
        ),
        "stageA_procedural": _filtered_samples(
            paths,
            object_name,
            generator="stageA_procedural",
            count=count,
        ),
        "stageB_sd2": _filtered_samples(
            paths,
            object_name,
            generator="stageB_sd2",
            count=count,
        ),
        "stageB_sdxl": _sdxl_samples(paths, object_name, count),
    }


def render(
    samples: Mapping[str, Sequence[GridSample]],
    *,
    object_name: str,
    output: Path,
) -> None:
    columns = len(next(iter(samples.values())))
    figure, axes = plt.subplots(
        len(SOURCE_ROWS),
        columns,
        figsize=(4.0 * columns, 3.2 * len(SOURCE_ROWS)),
        constrained_layout=True,
        squeeze=False,
    )
    figure.patch.set_facecolor("#f3f0e8")
    for row_index, (source, label) in enumerate(SOURCE_ROWS):
        row_samples = samples[source]
        require(len(row_samples) == columns, f"Grid row length changed: {source}")
        for column_index, sample in enumerate(row_samples):
            require(sample.image.is_file(), f"Missing grid image: {sample.image}")
            require(sample.mask.is_file(), f"Missing grid mask: {sample.mask}")
            with Image.open(sample.image) as image, Image.open(sample.mask) as mask:
                rendered = overlay_mask(image, mask)
            axis = axes[row_index, column_index]
            axis.imshow(rendered)
            axis.set_axis_off()
            title = f"{sample.defect_type} · {sample.sample_id[:28]}"
            if column_index == 0:
                title = f"{label}\n{title}"
            axis.set_title(
                title,
                fontsize=9 if column_index == 0 else 8,
                loc="left",
                fontweight="bold" if column_index == 0 else "normal",
            )
    figure.suptitle(
        f"DefectForge sample audit · {object_name}\nred overlay = generation/truth mask",
        fontsize=16,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


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
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument(
        "--validation-out",
        type=Path,
        default=Path("reports/sample_grids_validation.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_paths(args.paths)
    objects = tuple(args.objects or OBJECTS)
    require(set(objects) <= set(OBJECTS), f"Unknown object selection: {objects}")
    evidence: dict[str, Any] = {}
    for object_name in objects:
        samples = collect(paths, object_name, args.count)
        output = args.output_dir / f"sample_grid_{object_name}.png"
        render(samples, object_name=object_name, output=output)
        evidence[object_name] = {
            "figure": output.as_posix(),
            "figure_sha256": sha256_file(output),
            "samples": {
                source: [
                    {
                        "sample_id": sample.sample_id,
                        "defect_type": sample.defect_type,
                        "image_sha256": sha256_file(sample.image),
                        "mask_sha256": sha256_file(sample.mask),
                    }
                    for sample in source_samples
                ]
                for source, source_samples in samples.items()
            },
        }
    atomic_write_json(
        args.validation_out,
        {
            "status": "passed",
            "schema_version": 1,
            "visual_inspection_required": True,
            "objects": evidence,
        },
    )
    print(f"Built sample grids: {', '.join(objects)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
