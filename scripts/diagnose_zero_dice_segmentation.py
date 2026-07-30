"""Explain why some M20 segmentation runs score exactly zero Dice at threshold 0.5.

Six of the sixteen physical runs produce an empty binary mask on the frozen test set.
Three of them still reach a high pixel AUROC, which suggests the probability map carried
signal and the fixed threshold emptied the mask; the other three look close to random.
This script replaces that inference with measurement.

For every run it reloads the frozen `data_manifest.json` test records and the run's own
final SafeTensors checkpoint, re-runs inference, and then does two things:

1. recomputes the published metrics and requires them to match `training_report.json`,
   so the diagnosis is provably measuring the same thing as the released numbers;
2. reports the predicted-probability distribution, above all the maximum, which decides
   whether an empty mask is arithmetically unavoidable at the pre-registered threshold.

This is post-hoc reporting on already-final results. It never fits a model, never selects
a checkpoint, a threshold or a hyperparameter, and never writes to `results/`. Evaluation
is the only stage permitted to read the frozen test set (see `.claude/skills/df-guard`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import write_text_lf  # isort: skip
from src.common.paths import load_paths  # isort: skip
from src.training.segmenter_data import SegmentationSample  # isort: skip
from src.training.train_segmenter import (  # isort: skip
    SegmentationDataset,
    _loader,
    load_config,
    segmentation_metrics,
)

# A run whose probability map ranks defect pixels this well is informative; an empty mask
# is then a threshold artefact rather than a failed model.
INFORMATIVE_PIXEL_AUROC = 0.80
# Published metrics are floats recomputed on another machine (Colab L4 vs local 4090),
# so require agreement rather than bit-identity.
METRIC_TOLERANCE = 5e-3
PERCENTILES = (50.0, 95.0, 99.0, 99.9)


class DiagnosisError(RuntimeError):
    """Raised when a run cannot be diagnosed against its own frozen evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosisError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing evidence file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"Not a JSON object: {path}")
    return payload


def samples_from_manifest(records: Sequence[Mapping[str, Any]]) -> tuple[SegmentationSample, ...]:
    samples = []
    for record in records:
        require(record["kind"] == "real", "Test partition contains a non-real record")
        samples.append(
            SegmentationSample(
                sample_id=str(record["sample_id"]),
                object_name=str(record["object_name"]),
                kind=str(record["kind"]),
                source_name=str(record["source_name"]),
                root=str(record["root"]),
                image_path=str(record["image_path"]),
                image_sha256=str(record["image_sha256"]),
                mask_path=record["mask_path"],
                mask_sha256=record["mask_sha256"],
                has_defect=bool(record["has_defect"]),
                manifest_refs=tuple(record["manifest_refs"]),
                defect_type=record.get("defect_type"),
            )
        )
    return tuple(samples)


@torch.inference_mode()
def infer(
    model: torch.nn.Module,
    loader: Any,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for images, batch_masks, _ in loader:
        images = images.to(device, non_blocking=False)
        logits = model(pixel_values=images).logits
        logits = nnf.interpolate(
            logits,
            size=batch_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy()[:, 0])
        masks.append(batch_masks.numpy()[:, 0] > 0)
    return np.concatenate(probabilities), np.concatenate(masks)


def probability_statistics(
    probabilities: np.ndarray,
    masks: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    defect_images = masks.any(axis=(1, 2))
    per_image_max = probabilities.max(axis=(1, 2))
    inside = probabilities[masks]
    outside = probabilities[~masks]
    return {
        "max_probability": float(probabilities.max()),
        "percentiles": {
            f"p{value:g}": float(np.percentile(probabilities, value)) for value in PERCENTILES
        },
        "pixels_at_or_above_threshold": int((probabilities >= threshold).sum()),
        "images_with_any_pixel_at_or_above_threshold": int((per_image_max >= threshold).sum()),
        "images_total": int(probabilities.shape[0]),
        "defect_images": int(defect_images.sum()),
        "max_probability_inside_ground_truth": float(inside.max()) if inside.size else None,
        "max_probability_outside_ground_truth": float(outside.max()) if outside.size else None,
        "mean_probability_inside_ground_truth": float(inside.mean()) if inside.size else None,
        "mean_probability_outside_ground_truth": float(outside.mean()) if outside.size else None,
    }


def classify(
    *,
    dice: float,
    max_probability: float,
    pixel_auroc: float,
    threshold: float,
    peak_inside_ground_truth: bool | None,
) -> str:
    """Name what was measured, not what it is assumed to mean.

    Every zero-Dice run observed so far peaks below the threshold, so "no pixel reaches
    the threshold" alone does not separate an under-confident but correct model from one
    whose most confident pixel is a false positive. Where the peak sits does.
    """
    if dice > 0.0:
        return "non_zero_dice"
    if max_probability >= threshold:
        return "positive_pixels_never_overlap_ground_truth"
    if peak_inside_ground_truth:
        return "underconfident_peak_inside_ground_truth"
    if pixel_auroc >= INFORMATIVE_PIXEL_AUROC:
        return "underconfident_peak_outside_ground_truth_ranking_informative"
    return "underconfident_peak_outside_ground_truth_ranking_near_random"


def diagnose_run(
    run_dir: Path,
    *,
    paths: Any,
    config: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    report = load_json(run_dir / "training_report.json")
    manifest = load_json(run_dir / "data_manifest.json")
    require(report["mode"] == "final", f"{run_dir.name} is not a final run")
    require(
        manifest["selection_sha256"] == report["selection_sha256"]
        and manifest["split_manifest_sha256"] == report["split_manifest_sha256"],
        f"{run_dir.name}: data manifest is not bound to the same frozen split",
    )
    checkpoint = run_dir / "final"
    weight_path = checkpoint / "model.safetensors"
    observed = sha256_file(weight_path)
    require(
        observed == report["model_sha256"],
        f"{run_dir.name}: checkpoint SHA256 does not match its training report",
    )

    samples = samples_from_manifest(manifest["test"])
    require(
        len(samples) == int(report["evaluation_counts"]["total"]),
        f"{run_dir.name}: frozen test count does not match the training report",
    )
    dataset = SegmentationDataset(
        paths,
        samples,
        config,
        standard_augmentation=False,
        seed=int(report["seed"]),
    )
    loader = _loader(dataset, batch_size=batch_size)
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        use_safetensors=True,
    ).to(device)
    probabilities, masks = infer(model, loader, device=device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metrics_config = config["metrics"]
    threshold = float(metrics_config["threshold"])
    recomputed = segmentation_metrics(
        probabilities,
        masks,
        threshold=threshold,
        au_pro_max_fpr=float(metrics_config["au_pro_max_fpr"]),
        au_pro_thresholds=int(metrics_config["au_pro_thresholds"]),
    )
    published = report["metrics"]
    deltas = {name: abs(recomputed[name] - float(published[name])) for name in recomputed}
    reproduced = all(delta <= METRIC_TOLERANCE for delta in deltas.values())
    require(
        reproduced,
        f"{run_dir.name}: recomputed metrics do not reproduce the published ones: {deltas}",
    )

    statistics = probability_statistics(probabilities, masks, threshold=threshold)
    inside = statistics["max_probability_inside_ground_truth"]
    outside = statistics["max_probability_outside_ground_truth"]
    peak_inside = None if inside is None or outside is None else inside > outside
    statistics["peak_inside_ground_truth"] = peak_inside
    exposure = report["sample_exposure"]
    real_defect = int(exposure["real_defect"])
    synthetic_defect = int(exposure["synthetic_defect"])
    defect_exposure = real_defect + synthetic_defect
    return {
        "run_name": report["run_name"],
        "object": report["object"],
        "canonical_group": report["canonical_group"],
        "published_metrics": {name: float(published[name]) for name in recomputed},
        "recomputed_metrics": recomputed,
        "metric_deltas": deltas,
        "reproduced_published_metrics": reproduced,
        "threshold": threshold,
        "probability": statistics,
        "defect_exposure": {
            "real": real_defect,
            "synthetic": synthetic_defect,
            "real_share": (real_defect / defect_exposure) if defect_exposure else None,
        },
        "verdict": classify(
            dice=float(published["dice"]),
            max_probability=statistics["max_probability"],
            pixel_auroc=float(published["pixel_auroc"]),
            threshold=threshold,
            peak_inside_ground_truth=peak_inside,
        ),
    }


def _seed_of(run: Mapping[str, Any]) -> str:
    """Read the seed back out of the run name, which is the only place it is recorded."""
    name = str(run["run_name"])
    _, _, tail = name.rpartition("_seed")
    return tail or "—"


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# 零 Dice 分割 run 診斷",
        "",
        "> 由 `scripts/diagnose_zero_dice_segmentation.py` 產生。",
        "> 這是對**已完成**結果的事後量測：重跑推論、先與已發佈數字對帳，再回報機率分布。",
        "> 不做任何模型、threshold 或超參選擇，也不寫入 `results/`。",
        "",
        f"預註冊 threshold：`{payload['threshold']}`　",
        (
            f"全部 {payload['runs_total']} 個 physical run 的重算指標都與已發佈值相符"
            f"（最大差異 `{payload['max_metric_delta']:.2e}`）。"
        ),
        "",
        "## 全部 run",
        "",
        (
            "| 物件 | 組別 | Seed | Dice | pixel AUROC | 最高預測機率 "
            "| ≥ threshold 的像素 | 真實瑕疵曝光佔比 |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in payload["runs"]:
        probability = run["probability"]
        share = run["defect_exposure"]["real_share"]
        share_text = "—" if share is None else f"{share * 100:.1f}%"
        lines.append(
            f"| {run['object']} | {run['canonical_group']} "
            f"| {_seed_of(run)} "
            f"| {run['published_metrics']['dice']:.4f} "
            f"| {run['published_metrics']['pixel_auroc']:.4f} "
            f"| {probability['max_probability']:.4f} "
            f"| {probability['pixels_at_or_above_threshold']:,} "
            f"| {share_text} |"
        )
    # The claim below must be derived, never asserted. A single-seed table happened to be
    # entirely ceiling-limited; a wider table need not be, and the report has to say so.
    zero_runs = [
        run for run in payload["runs"] if run["published_metrics"]["dice"] == 0.0
    ]
    ceiling_limited = [
        run
        for run in zero_runs
        if run["probability"]["pixels_at_or_above_threshold"] == 0
    ]
    missed_target = [run for run in zero_runs if run not in ceiling_limited]
    ceiling = max(
        (run["probability"]["max_probability"] for run in ceiling_limited),
        default=0.0,
    )
    floor = min(
        (
            run["probability"]["max_probability"]
            for run in payload["runs"]
            if run["published_metrics"]["dice"] > 0.0
        ),
        default=0.0,
    )
    fully_explained = not missed_target
    heading = (
        "## 最高預測機率完全決定 Dice 是否退化"
        if fully_explained
        else "## 零 Dice 有兩種成因，機率天花板只解釋其中一種"
    )
    lines += [
        "",
        heading,
        "",
        (
            f"- {len(ceiling_limited)} / {len(zero_runs)} 個零 Dice run 連一個正像素都沒有："
            f"最高預測機率**低於** threshold（其中最高的是 `{ceiling:.4f}`），"
            "空 Mask 是算術上必然"
        ),
    ]
    if missed_target:
        lines.append(
            f"- 另外 {len(missed_target)} 個 run **有**正像素，但**完全沒有落在真實瑕疵上**，"
            "因此 Dice 仍為 0。這類與機率天花板無關："
        )
        for run in missed_target:
            probability = run["probability"]
            lines.append(
                f"  - `{run['object']} / {run['canonical_group']}`（seed {_seed_of(run)}）："
                f"最高機率 `{probability['max_probability']:.4f}`、"
                f"{probability['pixels_at_or_above_threshold']:,} 個正像素、"
                f"pixel AUROC `{run['published_metrics']['pixel_auroc']:.4f}`"
            )
    lines += [
        f"- 全部非零 Dice run 的最高預測機率都**高於** threshold（最小 `{floor:.4f}`）",
        "",
        (
            (
                "也就是說：在這批 run 上，Dice 是否為 0 完全由**機率天花板**決定，"
                "與模型排序能力（pixel AUROC）無關。"
            )
            if fully_explained
            else (
                "也就是說：**機率天花板是主要但非唯一成因**。絕大多數零 Dice run 確實是"
                "整張圖沒有任何像素越過 threshold，但上列 run 證明「有正像素卻完全打偏」"
                "同樣會產生零 Dice。因此不能宣稱零 Dice 一律與模型的空間定位能力無關。"
            )
        ),
        "",
        "## 零 Dice run 的判定",
        "",
    ]
    for run in payload["runs"]:
        if run["published_metrics"]["dice"] > 0.0:
            continue
        probability = run["probability"]
        lines += [
            f"### {run['object']} / {run['canonical_group']} / seed {_seed_of(run)}",
            "",
            f"- 判定：`{run['verdict']}`",
            (
                f"- 最高預測機率 `{probability['max_probability']:.4f}`"
                f"（threshold `{run['threshold']}`）"
            ),
            (
                "- 真實瑕疵區域內的最高機率 "
                f"`{probability['max_probability_inside_ground_truth']:.4f}`，"
                f"區域外 `{probability['max_probability_outside_ground_truth']:.4f}`"
            ),
            f"- pixel AUROC `{run['published_metrics']['pixel_auroc']:.4f}`",
            "",
        ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/segmenter.yaml"))
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results/colab/segmentation"),
    )
    parser.add_argument("--object", dest="objects", action="append")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("reports/zero_dice_diagnosis.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/zero_dice_diagnosis.md"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_paths(args.paths)
    config = load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is not available")

    objects = tuple(args.objects) if args.objects else ("pcb1", "capsules")
    runs: list[dict[str, Any]] = []
    for object_name in objects:
        run_root = args.runs_root / object_name / "runs"
        require(run_root.is_dir(), f"Missing run root: {run_root}")
        for run_dir in sorted(run_root.iterdir()):
            if not (run_dir / "training_report.json").is_file():
                continue
            runs.append(
                diagnose_run(
                    run_dir,
                    paths=paths,
                    config=config,
                    device=device,
                    batch_size=args.batch_size,
                )
            )
            print(f"diagnosed {runs[-1]['run_name']}: {runs[-1]['verdict']}")

    require(bool(runs), "No final segmentation runs were found")
    zero_dice = [run for run in runs if run["published_metrics"]["dice"] == 0.0]
    payload = {
        "status": "passed",
        "schema_version": 1,
        "threshold": runs[0]["threshold"],
        "runs_total": len(runs),
        "zero_dice_runs": len(zero_dice),
        "max_metric_delta": max(max(run["metric_deltas"].values()) for run in runs),
        "verdict_counts": {
            verdict: sum(1 for run in runs if run["verdict"] == verdict)
            for verdict in sorted({run["verdict"] for run in runs})
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(
        args.output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    write_text_lf(args.report, render_markdown(payload))
    print(f"Diagnosed {len(runs)} runs ({len(zero_dice)} with zero Dice): {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
