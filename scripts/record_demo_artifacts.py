"""Run verified M22 checkpoints on frozen test images and record a local demo GIF."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file, verify_frozen_manifest  # isort: skip
from src.common.paths import Paths, load_paths  # isort: skip
from src.inference.demo_gradio import (  # isort: skip
    InspectionRuntime,
    SelectedCheckpoint,
    atomic_write_json,
    select_object_checkpoints,
)

OBJECTS = ("pcb1", "capsules")
LABELS = ("good", "bad")
FRAME_SIZE = (1280, 720)
PANEL_SIZE = (360, 420)
MAX_GIF_BYTES = 10 * 1024 * 1024


class DemoRecordingError(RuntimeError):
    """Raised when M22 evidence cannot be generated from formal checkpoints."""


class PredictionRuntime(Protocol):
    object_name: str

    def predict(
        self,
        image: Image.Image | np.ndarray | None,
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray, str]: ...


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DemoRecordingError(message)


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    payload = (
        str(value.dtype).encode("ascii")
        + b"\0"
        + ",".join(str(item) for item in value.shape).encode("ascii")
        + b"\0"
        + value.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def select_frozen_test_images(
    paths: Paths,
    *,
    object_name: str,
) -> list[dict[str, Any]]:
    """Return one deterministic normal and anomaly image from frozen high-shot test."""

    manifest, manifest_sha256 = verify_frozen_manifest(paths.splits)
    images = manifest.get("images")
    require(isinstance(images, list), "Frozen split manifest has no images list")
    selected: list[dict[str, Any]] = []
    for label in LABELS:
        matches = sorted(
            (
                item
                for item in images
                if isinstance(item, dict)
                and item.get("object") == object_name
                and item.get("set") == "test"
                and item.get("split_type") == "2cls_highshot"
                and item.get("label") == label
            ),
            key=lambda item: str(item["image_path"]),
        )
        require(bool(matches), f"No frozen {label} test image for {object_name}")
        record = dict(matches[0])
        image_path = (paths.visa_raw / str(record["image_path"])).resolve(strict=True)
        observed_sha256 = sha256_file(image_path)
        require(
            observed_sha256 == record.get("sha256"),
            f"Frozen test image hash changed: {record['image_path']}",
        )
        selected.append(
            {
                "object": object_name,
                "label": label,
                "relative_path": str(record["image_path"]).replace("\\", "/"),
                "image_path": image_path,
                "image_sha256": observed_sha256,
                "manifest_sha256": manifest_sha256,
            }
        )
    return selected


def _fit_panel(image: Image.Image | np.ndarray, *, mode: str = "RGB") -> Image.Image:
    if isinstance(image, Image.Image):
        value = image.convert(mode)
    else:
        value = Image.fromarray(np.asarray(image)).convert(mode)
    return ImageOps.contain(value, PANEL_SIZE, Image.Resampling.LANCZOS)


def build_demo_frame(
    image: Image.Image | np.ndarray,
    *,
    mask: np.ndarray,
    heatmap: np.ndarray,
    object_name: str,
    label: str,
    probabilities: Mapping[str, float],
    latency: str,
) -> Image.Image:
    """Render one legible, deterministic evidence frame for the animated GIF."""

    require(set(probabilities) == {"Defect", "Normal"}, "Unexpected classifier labels")
    canvas = Image.new("RGB", FRAME_SIZE, "#07161f")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, FRAME_SIZE[0], 92), fill="#0d2631")
    draw.text((42, 24), "DEFECTFORGE // VERIFIED LOCAL INSPECTION", fill="#77d5be")
    draw.text(
        (42, 54),
        f"{object_name.upper()}  |  FROZEN TEST LABEL: {label.upper()}",
        fill="#e8f0ec",
    )
    panels = (
        ("INPUT", _fit_panel(image)),
        ("BINARY MASK", _fit_panel(mask, mode="L").convert("RGB")),
        ("PROBABILITY HEATMAP", _fit_panel(heatmap)),
    )
    for index, (title, panel) in enumerate(panels):
        left = 42 + index * 410
        top = 128
        draw.rectangle(
            (left - 2, top - 2, left + 362, top + 422),
            outline="#315d67",
            width=2,
        )
        x = left + (360 - panel.width) // 2
        y = top + (420 - panel.height) // 2
        canvas.paste(panel, (x, y))
        draw.text((left, 562), title, fill="#f7ca48")
    defect = float(probabilities["Defect"])
    normal = float(probabilities["Normal"])
    draw.text(
        (42, 620),
        f"Defect {defect:.4f}   Normal {normal:.4f}",
        fill="#e8f0ec",
    )
    draw.text((42, 654), latency.replace("**", ""), fill="#a9c0bd")
    return canvas


def write_animated_gif(
    path: Path,
    frames: Sequence[Image.Image],
    *,
    duration_ms: int = 1800,
) -> None:
    require(len(frames) >= 2, "Demo GIF needs at least two frames")
    require(all(frame.size == FRAME_SIZE for frame in frames), "Demo GIF frame size changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frames[0].save(
        temporary,
        format="GIF",
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    require(
        temporary.stat().st_size <= MAX_GIF_BYTES,
        f"Demo GIF exceeds the tracked-file limit of {MAX_GIF_BYTES} bytes",
    )
    os.replace(temporary, path)


def _selection_evidence(
    selections: Mapping[str, tuple[SelectedCheckpoint, SelectedCheckpoint]],
    *,
    classification_results: Path,
    segmentation_results: Path,
) -> dict[str, Any]:
    return {
        "status": "passed",
        "schema_version": 1,
        "selection_is_post_evaluation_demo_only": True,
        "changes_reported_metrics": False,
        "classification_results_sha256": sha256_file(classification_results),
        "segmentation_results_sha256": sha256_file(segmentation_results),
        "objects": {
            object_name: {
                "classifier": classifier.evidence(),
                "segmenter": segmenter.evidence(),
            }
            for object_name, (classifier, segmenter) in sorted(selections.items())
        },
    }


def record_object(
    runtime: PredictionRuntime,
    samples: Sequence[Mapping[str, Any]],
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    frames: list[Image.Image] = []
    validations: list[dict[str, Any]] = []
    for sample in samples:
        image_path = sample["image_path"]
        require(isinstance(image_path, Path), "Sample image path is invalid")
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        probabilities, mask, heatmap, latency = runtime.predict(image)
        require(mask.shape == (image.height, image.width), "Demo mask shape changed")
        require(heatmap.shape == (image.height, image.width, 3), "Demo heatmap shape changed")
        require(set(np.unique(mask)) <= {0, 255}, "Demo mask is not binary")
        require(
            np.isfinite(tuple(probabilities.values())).all(),
            "Demo probabilities are nonfinite",
        )
        frames.append(
            build_demo_frame(
                image,
                mask=mask,
                heatmap=heatmap,
                object_name=str(sample["object"]),
                label=str(sample["label"]),
                probabilities=probabilities,
                latency=latency,
            )
        )
        validations.append(
            {
                "object": sample["object"],
                "label": sample["label"],
                "relative_path": sample["relative_path"],
                "image_sha256": sample["image_sha256"],
                "manifest_sha256": sample["manifest_sha256"],
                "probabilities": {
                    key: float(value) for key, value in sorted(probabilities.items())
                },
                "mask_sha256": _array_sha256(mask),
                "heatmap_sha256": _array_sha256(heatmap),
                "mask_coverage": float((mask > 0).mean()),
                "latency": latency,
            }
        )
    return frames, validations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument(
        "--classification-results",
        type=Path,
        default=Path("results/classification.csv"),
    )
    parser.add_argument(
        "--segmentation-results",
        type=Path,
        default=Path("results/segmentation.csv"),
    )
    parser.add_argument(
        "--segmentation-runs-root",
        type=Path,
        default=Path("results/colab/segmentation"),
    )
    parser.add_argument(
        "--selection-out",
        type=Path,
        default=Path("reports/demo_checkpoint_selection.json"),
    )
    parser.add_argument("--gif-out", type=Path, default=Path("assets/demo.gif"))
    parser.add_argument(
        "--validation-out",
        type=Path,
        default=Path("reports/demo_validation.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_paths(args.paths)
    selections: dict[str, tuple[SelectedCheckpoint, SelectedCheckpoint]] = {}
    samples_by_object: dict[str, list[dict[str, Any]]] = {}
    for object_name in OBJECTS:
        selections[object_name] = select_object_checkpoints(
            paths=paths,
            object_name=object_name,
            classification_results=args.classification_results,
            segmentation_results=args.segmentation_results,
            segmentation_runs_root=args.segmentation_runs_root,
        )
        samples_by_object[object_name] = select_frozen_test_images(
            paths,
            object_name=object_name,
        )

    require(torch.cuda.is_available(), "M22 demo recording requires CUDA")
    frames: list[Image.Image] = []
    validations: list[dict[str, Any]] = []
    for object_name in OBJECTS:
        classifier, segmenter = selections[object_name]
        runtime = InspectionRuntime(
            paths=paths,
            classifier_checkpoint=classifier.checkpoint,
            segmenter_checkpoint=segmenter.checkpoint,
            device=torch.device("cuda"),
        )
        object_frames, object_validations = record_object(
            runtime,
            samples_by_object[object_name],
        )
        frames.extend(object_frames)
        validations.extend(object_validations)
        del runtime
        torch.cuda.empty_cache()

    selection_payload = _selection_evidence(
        selections,
        classification_results=args.classification_results,
        segmentation_results=args.segmentation_results,
    )
    atomic_write_json(args.selection_out, selection_payload)
    write_animated_gif(args.gif_out, frames)
    validation_payload = {
        "status": "passed",
        "schema_version": 1,
        "share_enabled": False,
        "uses_frozen_test_images": True,
        "validated_outputs": len(validations),
        "objects": list(OBJECTS),
        "classification_sha256": sha256_file(args.classification_results),
        "segmentation_sha256": sha256_file(args.segmentation_results),
        "selection_sha256": sha256_file(args.selection_out),
        "demo_gif_sha256": sha256_file(args.gif_out),
        "demo_gif_frames": len(frames),
        "demo_gif_size": list(FRAME_SIZE),
        "outputs": validations,
    }
    atomic_write_json(args.validation_out, validation_payload)
    print(f"Recorded verified local demo: {args.gif_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
