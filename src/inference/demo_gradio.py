"""Local-only Gradio inspection console for the best M16 and M18 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file
from src.common.paths import Paths, load_paths
from src.training.train_classifier import (
    build_transforms as build_classifier_transforms,
)
from src.training.train_classifier import (
    load_config as load_classifier_config,
)
from src.training.train_classifier import (
    load_model as load_classifier_model,
)
from src.training.train_segmenter import (
    _load_saved_model_state as load_saved_segmenter_state,
)
from src.training.train_segmenter import load_config as load_segmenter_config
from src.training.train_segmenter import load_model as load_segmenter_model


class DemoError(RuntimeError):
    """Raised when the demo cannot prove its checkpoint or inference contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DemoError(message)


@dataclass(frozen=True, slots=True)
class VerifiedCheckpoint:
    """A SafeTensors checkpoint tied to its immutable training report."""

    weight: Path
    report: Path
    object_name: str
    model_sha256: str


@dataclass(frozen=True, slots=True)
class SelectedCheckpoint:
    """A deterministic post-evaluation choice bound back to one raw run."""

    checkpoint: VerifiedCheckpoint
    role: str
    run_name: str
    group_name: str
    primary_metric: str
    primary_value: float
    secondary_metric: str
    secondary_value: float

    def evidence(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "run_name": self.run_name,
            "group_name": self.group_name,
            "object": self.checkpoint.object_name,
            "primary_metric": self.primary_metric,
            "primary_value": self.primary_value,
            "secondary_metric": self.secondary_metric,
            "secondary_value": self.secondary_value,
            "model_sha256": self.checkpoint.model_sha256,
            "training_report_sha256": sha256_file(self.checkpoint.report),
        }


def _load_mapping(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"Expected a JSON object: {path}")
    return payload


def _weight_path(path: Path, *, role: str) -> Path:
    candidate = path.resolve(strict=False)
    if candidate.is_file():
        return candidate
    require(candidate.is_dir(), f"{role} checkpoint does not exist: {candidate}")
    relative_candidates = (Path("model.safetensors"), Path("final/model.safetensors"))
    matches = [candidate / relative for relative in relative_candidates]
    present = [item for item in matches if item.is_file()]
    require(len(present) == 1, f"{role} checkpoint must resolve to one model.safetensors")
    return present[0]


def verify_checkpoint(path: Path, *, role: str) -> VerifiedCheckpoint:
    """Verify a model against the adjacent raw training report."""

    require(role in {"classifier", "segmenter"}, f"Unknown checkpoint role: {role}")
    weight = _weight_path(path, role=role)
    report_candidates = [
        weight.parent / "training_report.json",
        weight.parent.parent / "training_report.json",
    ]
    reports = [candidate for candidate in report_candidates if candidate.is_file()]
    require(len(reports) == 1, f"{role} checkpoint needs one adjacent training_report.json")
    report_path = reports[0]
    report = _load_mapping(report_path)
    observed_sha256 = sha256_file(weight)
    require(report.get("status") == "passed", f"{role} training report did not pass")
    require(not bool(report.get("smoke")), f"{role} checkpoint is only a smoke model")
    require(report.get("mode") == "final", f"{role} checkpoint is not a final run")
    require(
        report.get("model_sha256") == observed_sha256,
        f"{role} checkpoint hash does not match its training report",
    )
    object_name = str(report.get("object", ""))
    require(bool(object_name), f"{role} training report has no object")
    return VerifiedCheckpoint(
        weight=weight,
        report=report_path,
        object_name=object_name,
        model_sha256=observed_sha256,
    )


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    require(path.is_file(), f"Missing formal result CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def _unit_metric(row: dict[str, str], metric: str, *, label: str) -> float:
    try:
        value = float(row[metric])
    except (KeyError, TypeError, ValueError) as error:
        raise DemoError(f"Invalid {label}/{metric}") from error
    require(math.isfinite(value) and 0.0 <= value <= 1.0, f"Invalid {label}/{metric}")
    return value


def select_best_checkpoint(
    *,
    results_path: Path,
    runs_root: Path,
    object_name: str,
    role: str,
) -> SelectedCheckpoint:
    """Select the best formal seed-42 physical run and rebind it to raw evidence."""

    require(role in {"classifier", "segmenter"}, f"Unknown checkpoint role: {role}")
    header, rows = _read_csv(results_path)
    if role == "classifier":
        required = {
            "run_name",
            "run_signature",
            "canonical_group",
            "object",
            "seed",
            "macro_f1",
            "auroc",
        }
        require(required <= set(header), "Classification result schema is incomplete")
        require(len(rows) == 38, "Classification selection requires all 38 formal rows")
        object_rows = [row for row in rows if row["object"] == object_name]
        require(len(object_rows) == 19, "Classification object inventory is incomplete")
        candidates = [row for row in object_rows if int(row["seed"]) == 42]
        require(len(candidates) == 15, "Classification seed-42 inventory is incomplete")
        primary_metric, secondary_metric = "macro_f1", "auroc"
    else:
        required = {
            "run_name",
            "run_signature",
            "logical_group",
            "canonical_group",
            "physical_run",
            "object",
            "seed",
            "dice",
            "aupro",
        }
        require(required <= set(header), "Segmentation result schema is incomplete")
        require(len(rows) == 18, "Segmentation selection requires all 18 logical rows")
        object_rows = [row for row in rows if row["object"] == object_name]
        require(len(object_rows) == 9, "Segmentation object inventory is incomplete")
        require(
            all(int(row["seed"]) == 42 for row in object_rows),
            "Segmentation selection requires seed 42",
        )
        candidates = [row for row in object_rows if row["physical_run"] == "true"]
        require(len(candidates) == 8, "Segmentation physical-run inventory is incomplete")
        primary_metric, secondary_metric = "dice", "aupro"
    require(
        len({row["run_name"] for row in candidates}) == len(candidates),
        f"{role} candidate run names are not unique",
    )
    scored = [
        (
            _unit_metric(row, primary_metric, label=row["run_name"]),
            _unit_metric(row, secondary_metric, label=row["run_name"]),
            row,
        )
        for row in candidates
    ]
    primary_value, secondary_value, selected = min(
        scored,
        key=lambda item: (-item[0], -item[1], item[2]["run_name"]),
    )
    checkpoint = verify_checkpoint(
        runs_root / selected["run_name"],
        role=role,
    )
    require(checkpoint.object_name == object_name, f"{role} selected the wrong object")
    report = _load_mapping(checkpoint.report)
    require(report.get("run_name") == selected["run_name"], f"{role} run name changed")
    require(report.get("run_signature") == selected["run_signature"], f"{role} signature changed")
    require(int(report.get("seed", -1)) == 42, f"{role} selected a non-seed-42 run")
    require(
        report.get("canonical_group") == selected["canonical_group"],
        f"{role} canonical group changed",
    )
    report_metrics = report.get("metrics")
    require(isinstance(report_metrics, dict), f"{role} report has no metrics")
    for metric, expected in (
        (primary_metric, primary_value),
        (secondary_metric, secondary_value),
    ):
        observed = float(report_metrics.get(metric, math.nan))
        require(
            math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12),
            f"{role} result CSV differs from its raw report for {metric}",
        )
    return SelectedCheckpoint(
        checkpoint=checkpoint,
        role=role,
        run_name=selected["run_name"],
        group_name=selected["canonical_group"],
        primary_metric=primary_metric,
        primary_value=primary_value,
        secondary_metric=secondary_metric,
        secondary_value=secondary_value,
    )


def select_object_checkpoints(
    *,
    paths: Paths,
    object_name: str,
    classification_results: Path,
    segmentation_results: Path,
    segmentation_runs_root: Path,
) -> tuple[SelectedCheckpoint, SelectedCheckpoint]:
    """Select and verify the two formal checkpoints for one demo object."""

    require(object_name in paths.objects, f"Unsupported demo object: {object_name}")
    classifier_config = load_classifier_config(paths.configs / "classifier.yaml")
    classifier_selection = select_best_checkpoint(
        results_path=classification_results,
        runs_root=paths.runs / str(classifier_config["output"]["run_subdirectory"]),
        object_name=object_name,
        role="classifier",
    )
    segmenter_selection = select_best_checkpoint(
        results_path=segmentation_results,
        runs_root=segmentation_runs_root / object_name / "runs",
        object_name=object_name,
        role="segmenter",
    )
    return classifier_selection, segmenter_selection


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(image)
    require(array.ndim in {2, 3}, "Input image must be HxW or HxWxC")
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    require(array.shape[2] == 3, "Input image must have one, three, or four channels")
    if np.issubdtype(array.dtype, np.floating):
        maximum = float(np.nanmax(array)) if array.size else 0.0
        array = array * 255.0 if maximum <= 1.0 else array
    return np.clip(array, 0, 255).astype(np.uint8)


def _colorize_probability(probability: np.ndarray) -> np.ndarray:
    """Industrial blue-to-amber heat ramp without an external colormap."""

    values = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    stops = np.asarray(
        [
            [7, 22, 31],
            [16, 103, 128],
            [77, 205, 181],
            [247, 202, 72],
            [237, 85, 59],
        ],
        dtype=np.float32,
    )
    position = values * (len(stops) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (position - lower)[..., None]
    colors = stops[lower] * (1.0 - fraction) + stops[upper] * fraction
    return np.rint(colors).astype(np.uint8)


def render_outputs(
    image: Image.Image | np.ndarray,
    *,
    anomaly_probability: float,
    pixel_probability: np.ndarray,
    threshold: float,
    elapsed_ms: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, str]:
    """Convert model outputs into the four stable M22 UI outputs."""

    rgb = _as_rgb_array(image)
    height, width = rgb.shape[:2]
    probability = np.asarray(pixel_probability, dtype=np.float32)
    require(probability.shape == (height, width), "Pixel probability shape changed")
    require(np.isfinite(probability).all(), "Pixel probability contains non-finite values")
    anomaly_probability = float(anomaly_probability)
    require(0.0 <= anomaly_probability <= 1.0, "Anomaly probability is outside [0, 1]")
    require(0.0 <= threshold <= 1.0, "Mask threshold is outside [0, 1]")

    mask = (probability >= threshold).astype(np.uint8) * 255
    color = _colorize_probability(probability)
    alpha = (0.18 + 0.52 * np.clip(probability, 0.0, 1.0))[..., None]
    heatmap = np.rint(rgb * (1.0 - alpha) + color * alpha).astype(np.uint8)
    probabilities = {
        "Defect": anomaly_probability,
        "Normal": 1.0 - anomaly_probability,
    }
    latency = (
        f"**{elapsed_ms:.1f} ms** end-to-end  ·  "
        f"mask coverage **{float((mask > 0).mean()) * 100.0:.2f}%**"
    )
    return probabilities, mask, heatmap, latency


class InspectionRuntime:
    """Load one object-matched classifier/segmenter pair and run inference."""

    def __init__(
        self,
        *,
        paths: Paths,
        classifier_checkpoint: VerifiedCheckpoint,
        segmenter_checkpoint: VerifiedCheckpoint,
        device: torch.device,
    ) -> None:
        require(
            classifier_checkpoint.object_name == segmenter_checkpoint.object_name,
            "Classifier and segmenter checkpoints target different objects",
        )
        require(device.type == "cuda" and torch.cuda.is_available(), "M22 requires CUDA")
        self.object_name = classifier_checkpoint.object_name
        self.device = device

        classifier_config = load_classifier_config(paths.configs / "classifier.yaml")
        classifier, _ = load_classifier_model(paths, classifier_config["model"])
        classifier.load_state_dict(
            load_file(str(classifier_checkpoint.weight), device="cpu"),
            strict=True,
        )
        _, self.classifier_transform = build_classifier_transforms(
            classifier_config,
            standard_augmentation=False,
        )
        self.classifier = classifier.eval().to(device)

        segmenter_config = load_segmenter_config(paths.configs / "segmenter.yaml")
        segmenter, _ = load_segmenter_model(paths, segmenter_config["model"])
        load_saved_segmenter_state(segmenter, segmenter_checkpoint.weight.parent)
        self.segmenter = segmenter.eval().to(device)
        self.segmenter_size = int(segmenter_config["model"]["input_size"])
        self.segmenter_mean = tuple(float(value) for value in segmenter_config["model"]["mean"])
        self.segmenter_std = tuple(float(value) for value in segmenter_config["model"]["std"])
        self.threshold = float(segmenter_config["metrics"]["threshold"])

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image | np.ndarray | None,
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray, str]:
        require(image is not None, "Upload an inspection image first")
        rgb = _as_rgb_array(image)
        pil = Image.fromarray(rgb, mode="RGB")
        height, width = rgb.shape[:2]

        classifier_input = self.classifier_transform(pil).unsqueeze(0).to(self.device)
        segmenter_input = tvf.resize(
            pil,
            [self.segmenter_size, self.segmenter_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        segmenter_input = (
            tvf.normalize(
                tvf.to_tensor(segmenter_input),
                mean=self.segmenter_mean,
                std=self.segmenter_std,
            )
            .unsqueeze(0)
            .to(self.device)
        )

        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            classifier_logits = self.classifier(classifier_input)
            segmenter_logits = self.segmenter(pixel_values=segmenter_input).logits
            segmenter_logits = nnf.interpolate(
                segmenter_logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        anomaly_probability = float(torch.softmax(classifier_logits.float(), dim=1)[0, 1].cpu())
        pixel_probability = torch.sigmoid(segmenter_logits.float())[0, 0].cpu().numpy()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return render_outputs(
            rgb,
            anomaly_probability=anomaly_probability,
            pixel_probability=pixel_probability,
            threshold=self.threshold,
            elapsed_ms=elapsed_ms,
        )


DEMO_CSS = """
:root {
  --df-ink: #07161f;
  --df-panel: #0d2631;
  --df-rule: rgba(119, 213, 190, 0.28);
  --df-mint: #77d5be;
  --df-amber: #f7ca48;
  --df-paper: #e8f0ec;
}
.gradio-container {
  background:
    linear-gradient(rgba(119,213,190,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(119,213,190,.035) 1px, transparent 1px),
    radial-gradient(circle at 85% 0%, rgba(247,202,72,.12), transparent 28%),
    var(--df-ink) !important;
  background-size: 28px 28px, 28px 28px, auto, auto !important;
  color: var(--df-paper) !important;
  font-family: Bahnschrift, "DIN Alternate", sans-serif !important;
}
#df-shell { max-width: 1320px; margin: 0 auto; }
#df-kicker {
  color: var(--df-mint); font-size: .76rem; letter-spacing: .22em;
  text-transform: uppercase; margin-bottom: .35rem;
}
#df-title h1 {
  color: var(--df-paper); font-size: clamp(2rem, 5vw, 4.6rem);
  line-height: .93; letter-spacing: -.035em; margin: 0 0 1rem;
}
#df-title p { color: #a9c0bd; max-width: 67ch; }
.df-panel {
  border: 1px solid var(--df-rule) !important;
  border-radius: 2px !important;
  background: rgba(13,38,49,.82) !important;
  box-shadow: 8px 8px 0 rgba(0,0,0,.22) !important;
}
#df-run {
  background: var(--df-amber) !important; color: #1c210e !important;
  border: 0 !important; border-radius: 2px !important;
  font-weight: 800 !important; letter-spacing: .08em; text-transform: uppercase;
}
#df-object {
  border-left: 3px solid var(--df-mint); padding-left: .75rem;
  color: #a9c0bd;
}
"""


def build_app(runtime: InspectionRuntime) -> Any:
    """Build the UI lazily so non-UI tests do not require a browser."""

    import gradio as gr

    with (
        gr.Blocks(title="DefectForge Inspection Console") as demo,
        gr.Column(elem_id="df-shell"),
    ):
        gr.Markdown("MACHINE VISION / LOCAL INFERENCE", elem_id="df-kicker")
        gr.Markdown(
            "# DefectForge\n"
            "A paired classification + segmentation console for few-shot industrial defects.",
            elem_id="df-title",
        )
        gr.Markdown(
            f"Object model: **{runtime.object_name}** · "
            "checkpoints verified against raw training reports",
            elem_id="df-object",
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes=["df-panel"]):
                input_image = gr.Image(
                    label="Inspection frame",
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                analyze = gr.Button("Run inspection", variant="primary", elem_id="df-run")
            with gr.Column(scale=4):
                probabilities = gr.Label(
                    label="Classification confidence",
                    num_top_classes=2,
                    elem_classes=["df-panel"],
                )
                latency = gr.Markdown(
                    "Awaiting an inspection frame.",
                    elem_classes=["df-panel"],
                )
        with gr.Row():
            mask = gr.Image(
                label="Binary defect mask",
                image_mode="L",
                elem_classes=["df-panel"],
            )
            heatmap = gr.Image(
                label="Defect probability heatmap",
                elem_classes=["df-panel"],
            )
        analyze.click(
            fn=runtime.predict,
            inputs=input_image,
            outputs=[probabilities, mask, heatmap, latency],
            concurrency_limit=1,
            show_progress="full",
        )
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument(
        "--object",
        choices=("pcb1", "capsules"),
        help="Auto-select both best formal checkpoints for this object",
    )
    parser.add_argument("--cls-ckpt", type=Path)
    parser.add_argument("--seg-ckpt", type=Path)
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
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Explicitly allow a Gradio share URL")
    parser.add_argument("--inbrowser", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(1 <= args.port <= 65535, "Port must be between 1 and 65535")
    paths = load_paths(args.paths)
    if args.object is not None:
        require(
            args.cls_ckpt is None and args.seg_ckpt is None,
            "--object cannot be mixed with explicit checkpoints",
        )
        classifier_selection, segmenter_selection = select_object_checkpoints(
            paths=paths,
            object_name=args.object,
            classification_results=args.classification_results,
            segmentation_results=args.segmentation_results,
            segmentation_runs_root=args.segmentation_runs_root,
        )
        classifier_checkpoint = classifier_selection.checkpoint
        segmenter_checkpoint = segmenter_selection.checkpoint
        atomic_write_json(
            args.selection_out,
            {
                "status": "passed",
                "schema_version": 1,
                "selection_is_post_evaluation_demo_only": True,
                "changes_reported_metrics": False,
                "object": args.object,
                "classification_results_sha256": sha256_file(args.classification_results),
                "segmentation_results_sha256": sha256_file(args.segmentation_results),
                "classifier": classifier_selection.evidence(),
                "segmenter": segmenter_selection.evidence(),
            },
        )
    else:
        require(
            args.cls_ckpt is not None and args.seg_ckpt is not None,
            "Pass --object or both --cls-ckpt and --seg-ckpt",
        )
        classifier_checkpoint = verify_checkpoint(args.cls_ckpt, role="classifier")
        segmenter_checkpoint = verify_checkpoint(args.seg_ckpt, role="segmenter")
    runtime = InspectionRuntime(
        paths=paths,
        classifier_checkpoint=classifier_checkpoint,
        segmenter_checkpoint=segmenter_checkpoint,
        device=torch.device("cuda"),
    )
    app = build_app(runtime)
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        css=DEMO_CSS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
