"""Render fixed-context original-versus-searched Stage B contact sheets."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import load_json, sha256_file  # isort: skip
from src.common.paths import Paths, load_paths  # isort: skip
from src.synthetic.generate_diffusion import (  # isort: skip
    DiffusionGenerationError,
    write_png_atomic,
)

PANEL_SIZE = 256
CAPTION_HEIGHT = 44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml", type=Path)
    parser.add_argument("--out-name", default="stageB_sd2")
    parser.add_argument("--object", required=True)
    parser.add_argument("--maximum", default=12, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_metadata(path: Path, object_name: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise DiffusionGenerationError(f"Missing metadata: {path}")
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DiffusionGenerationError(
                    f"Metadata is not an object: {path}:{line_number}"
                )
            if value.get("object") != object_name:
                continue
            sample_id = str(value.get("sample_id", ""))
            if not sample_id or sample_id in records:
                raise DiffusionGenerationError(
                    f"Missing or duplicate sample_id: {path}:{line_number}"
                )
            records[sample_id] = value
    return records


def read_sidecar(root: Path, sample_id: str) -> dict[str, Any]:
    path = root / ".records" / f"{sample_id}.json"
    value = load_json(path)
    if value.get("record", {}).get("sample_id") != sample_id:
        raise DiffusionGenerationError(f"Malformed sidecar: {path}")
    return value


def union_crop(
    first: Sequence[int],
    second: Sequence[int],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if len(first) != 4 or len(second) != 4:
        raise DiffusionGenerationError("Expected [x, y, width, height] crop boxes")
    first_x, first_y, first_width, first_height = (int(value) for value in first)
    second_x, second_y, second_width, second_height = (int(value) for value in second)
    left = max(0, min(first_x, second_x))
    top = max(0, min(first_y, second_y))
    right = min(width, max(first_x + first_width, second_x + second_width))
    bottom = min(height, max(first_y + first_height, second_y + second_height))
    if right <= left or bottom <= top:
        raise DiffusionGenerationError("Invalid comparison crop")
    return left, top, right, bottom


def select_records(
    original: dict[str, dict[str, Any]],
    searched: dict[str, dict[str, Any]],
    maximum: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if maximum < 1:
        raise DiffusionGenerationError("--maximum must be positive")
    if set(original) != set(searched):
        missing_searched = sorted(set(original) - set(searched))
        missing_original = sorted(set(searched) - set(original))
        raise DiffusionGenerationError(
            "Bucket sample IDs differ: "
            f"missing searched={missing_searched[:3]}, "
            f"missing original={missing_original[:3]}"
        )
    if len(original) < maximum:
        raise DiffusionGenerationError(
            f"Requested {maximum} rows, but only {len(original)} aligned samples exist"
        )
    by_type: dict[str, list[str]] = {}
    for sample_id, record in original.items():
        by_type.setdefault(str(record.get("defect_type", "")), []).append(sample_id)
    if "" in by_type:
        raise DiffusionGenerationError("Comparison record is missing defect_type")
    for sample_ids in by_type.values():
        sample_ids.sort()
    selected_ids: list[str] = []
    while len(selected_ids) < maximum:
        added = False
        for defect_type in sorted(by_type):
            sample_ids = by_type[defect_type]
            if sample_ids:
                selected_ids.append(sample_ids.pop(0))
                added = True
                if len(selected_ids) == maximum:
                    break
        if not added:
            break
    return [(original[sample_id], searched[sample_id]) for sample_id in selected_ids]


def assert_pair_invariants(
    *,
    paths: Paths,
    original_root: Path,
    searched_root: Path,
    original: dict[str, Any],
    searched: dict[str, Any],
) -> None:
    sample_id = str(original["sample_id"])
    if searched["sample_id"] != sample_id:
        raise DiffusionGenerationError("Comparison pair sample_id mismatch")
    if original["source"] != searched["source"]:
        raise DiffusionGenerationError(f"Comparison source mismatch: {sample_id}")
    original_mask = original_root / str(original["mask_path"])
    searched_mask = searched_root / str(searched["mask_path"])
    if sha256_file(original_mask) != sha256_file(searched_mask):
        raise DiffusionGenerationError(f"Comparison mask mismatch: {sample_id}")
    background = paths.visa_raw / str(original["source"]["background_image"])
    if sha256_file(background) != original["source"]["background_sha256"]:
        raise DiffusionGenerationError(f"Comparison background mismatch: {sample_id}")


def render(
    *,
    paths: Paths,
    original_root: Path,
    searched_root: Path,
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    output: Path,
) -> None:
    canvas = Image.new(
        "RGB",
        (PANEL_SIZE * 4, (PANEL_SIZE + CAPTION_HEIGHT) * len(pairs) + CAPTION_HEIGHT),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    headings = ("clean", "mask", "original", "searched")
    for column, heading in enumerate(headings):
        draw.text((column * PANEL_SIZE + 6, 13), heading, fill="black")

    for row, (original, searched) in enumerate(pairs):
        assert_pair_invariants(
            paths=paths,
            original_root=original_root,
            searched_root=searched_root,
            original=original,
            searched=searched,
        )
        sample_id = str(original["sample_id"])
        original_sidecar = read_sidecar(original_root, sample_id)
        searched_sidecar = read_sidecar(searched_root, sample_id)
        with Image.open(
            paths.visa_raw / str(original["source"]["background_image"])
        ) as handle:
            background = handle.convert("RGB")
        with Image.open(original_root / str(original["mask_path"])) as handle:
            mask = handle.convert("RGB")
        with Image.open(original_root / str(original["image_path"])) as handle:
            original_image = handle.convert("RGB")
        with Image.open(searched_root / str(searched["image_path"])) as handle:
            searched_image = handle.convert("RGB")
        crop = union_crop(
            original["generation"]["crop_bbox"],
            searched["generation"]["crop_bbox"],
            width=background.width,
            height=background.height,
        )
        images = (background, mask, original_image, searched_image)
        row_top = CAPTION_HEIGHT + row * (PANEL_SIZE + CAPTION_HEIGHT)
        for column, image in enumerate(images):
            thumbnail = image.crop(crop)
            thumbnail.thumbnail((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)
            x = column * PANEL_SIZE + (PANEL_SIZE - thumbnail.width) // 2
            y = row_top + (PANEL_SIZE - thumbnail.height) // 2
            canvas.paste(thumbnail, (x, y))

        original_candidate = original_sidecar["candidates"][
            int(original_sidecar["selected_candidate_index"])
        ]
        searched_candidate = searched_sidecar["candidates"][
            int(searched_sidecar["selected_candidate_index"])
        ]
        label = (
            f"{sample_id} | original score={float(original_candidate['score']):.3f} "
            f"g={float(original_candidate['guidance_scale']):g} "
            f"c={float(original_candidate['crop_ratio']):g} | searched "
            f"score={float(searched_candidate['score']):.3f} "
            f"g={float(searched_candidate['guidance_scale']):g} "
            f"c={float(searched_candidate['crop_ratio']):g}"
        )
        draw.text((6, row_top + PANEL_SIZE + 11), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_png_atomic(output, canvas)


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths)
    if args.object not in paths.objects:
        raise DiffusionGenerationError(f"Unsupported object: {args.object}")
    root = paths.synthetic / args.out_name
    original_root = root / "original"
    searched_root = root / "searched"
    original = read_metadata(original_root / "metadata.jsonl", args.object)
    searched = read_metadata(searched_root / "metadata.jsonl", args.object)
    pairs = select_records(original, searched, args.maximum)
    render(
        paths=paths,
        original_root=original_root,
        searched_root=searched_root,
        pairs=pairs,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "object": args.object,
                "output": str(args.output.resolve()),
                "rows": len(pairs),
                "status": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
