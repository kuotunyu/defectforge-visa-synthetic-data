"""Measure how often standard augmentation removes the defect from a training pair.

`capsules/std_aug` scores exactly zero Dice while drawing 100% of its defect exposure
from real images, and differs from `real_only` only by standard augmentation. Its
probability ceiling collapses from 0.9920 to 0.3157. `pcb1/std_aug` is unaffected.

One mechanism would explain that asymmetry: the augmentation pipeline starts with a
random resized crop and then applies an affine transform, so a small defect can be pushed
out of frame. The pair survives as an image labelled "defect" whose mask is now empty,
and training on enough of those teaches the model to predict nothing.

This script measures that directly. For every real defect image in a run's frozen train
partition it replays the exact draws the trainer would have seen — same transform, same
`_stable_seed(seed, sample_id, draw_index)` sequence — and reports how often the mask
survives.

Train-side only: it reads the `train` partition and asserts the loaded records carry no
test membership. No model is loaded, no metric is recomputed, nothing is selected.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import write_text_lf  # isort: skip
from src.common.paths import load_paths  # isort: skip
from src.training.segmenter_data import (  # isort: skip
    SegmentationSample,
    resolve_image_path,
    resolve_mask_path,
)
from src.training.train_segmenter import (  # isort: skip
    _paired_transform,
    _stable_seed,
    load_config,
)

PERCENTILES = (5, 25, 50, 75, 95)


class AugmentationDiagnosisError(RuntimeError):
    """Raised when the train-side augmentation replay cannot be trusted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AugmentationDiagnosisError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing evidence file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def defect_samples(manifest: Mapping[str, Any]) -> tuple[SegmentationSample, ...]:
    samples = []
    for record in manifest["train"]:
        if not record["has_defect"] or record["kind"] != "real":
            continue
        require(record["mask_path"] is not None, "A real defect record has no mask")
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
    require(bool(samples), "No real defect images in the train partition")
    return tuple(samples)


def replay_sample(
    sample: SegmentationSample,
    *,
    paths: Any,
    config: Mapping[str, Any],
    seed: int,
    draws: int,
) -> dict[str, Any]:
    with Image.open(resolve_image_path(paths, sample)) as handle:
        image = handle.convert("RGB")
    mask_path = resolve_mask_path(paths, sample)
    require(mask_path is not None, f"{sample.sample_id}: missing mask")
    with Image.open(mask_path) as handle:
        mask = handle.convert("L")

    source_pixels = image.size[0] * image.size[1]
    source_defect = int((np.asarray(mask) > 0).sum())

    # The trainer never augments the evaluation pass; this is the reference mask area.
    _, baseline_mask = _paired_transform(
        image,
        mask,
        config=config,
        standard_augmentation=False,
        draw_seed=_stable_seed(seed, sample.sample_id, "baseline"),
    )
    baseline_area = float(baseline_mask.sum())
    require(baseline_area > 0.0, f"{sample.sample_id}: mask is empty without augmentation")

    retained: list[float] = []
    empty = 0
    for draw_index in range(draws):
        _, augmented_mask = _paired_transform(
            image,
            mask,
            config=config,
            standard_augmentation=True,
            draw_seed=_stable_seed(seed, sample.sample_id, str(draw_index)),
        )
        area = float(augmented_mask.sum())
        if area == 0.0:
            empty += 1
        retained.append(area / baseline_area)

    return {
        "sample_id": sample.sample_id,
        "source_defect_area_fraction": source_defect / source_pixels,
        "baseline_mask_pixels": baseline_area,
        "draws": draws,
        "empty_mask_draws": empty,
        "empty_mask_rate": empty / draws,
        "retained_area_ratio_mean": statistics.fmean(retained),
        "retained_area_ratio_median": statistics.median(retained),
    }


def loss_trajectory(run_dir: Path, *, window: int) -> dict[str, Any]:
    """Windowed training loss, to show whether the Dice term ever engaged."""
    report = load_json(run_dir / "training_report.json")
    history = report["loss_history"]
    require(bool(history), f"{run_dir.name}: empty loss history")
    windows = []
    for end in range(window, len(history) + 1, window):
        chunk = history[end - window : end]
        windows.append(
            {
                "step": end,
                "dice_loss": statistics.fmean(item["dice_loss"] for item in chunk),
                "bce_loss": statistics.fmean(item["bce_loss"] for item in chunk),
            }
        )
    require(len(windows) >= 2, f"{run_dir.name}: too few steps to form two windows")
    first, last = windows[0], windows[-1]
    return {
        "run_name": report["run_name"],
        "canonical_group": report["canonical_group"],
        "standard_augmentation": bool(report["standard_augmentation"]),
        "final_dice": float(report["metrics"]["dice"]),
        "windows": windows,
        "dice_loss_first_window": first["dice_loss"],
        "dice_loss_last_window": last["dice_loss"],
        "dice_loss_improvement": first["dice_loss"] - last["dice_loss"],
        "bce_loss_improvement": first["bce_loss"] - last["bce_loss"],
    }


def compare_augmentation(
    object_name: str,
    *,
    runs_root: Path,
    seed: int,
    window: int,
) -> dict[str, Any]:
    trajectories = {}
    for group in ("real_only", "std_aug"):
        run_dir = runs_root / object_name / "runs" / f"m18_{group}_{object_name}_seed{seed}"
        trajectories[group] = loss_trajectory(run_dir, window=window)
    baseline = trajectories["real_only"]
    augmented = trajectories["std_aug"]
    return {
        "object": object_name,
        "trajectories": trajectories,
        # BCE falls in every run because predicting background is easy. Only the Dice
        # term shows whether the model ever learned to overlap a defect.
        "dice_term_engaged_without_augmentation": baseline["dice_loss_improvement"] > 0.01,
        "dice_term_engaged_with_augmentation": augmented["dice_loss_improvement"] > 0.01,
    }


def summarise(object_name: str, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_draws = sum(int(item["draws"]) for item in samples)
    total_empty = sum(int(item["empty_mask_draws"]) for item in samples)
    areas = [float(item["source_defect_area_fraction"]) for item in samples]
    retained = [float(item["retained_area_ratio_mean"]) for item in samples]
    return {
        "object": object_name,
        "defect_images": len(samples),
        "total_draws": total_draws,
        "empty_mask_draws": total_empty,
        "empty_mask_rate": total_empty / total_draws,
        "images_that_ever_lose_the_defect": sum(
            1 for item in samples if int(item["empty_mask_draws"]) > 0
        ),
        "source_defect_area_fraction": {
            f"p{value}": _percentile(areas, value) for value in PERCENTILES
        },
        "retained_area_ratio_mean": statistics.fmean(retained),
        "samples": sorted(samples, key=lambda item: -float(item["empty_mask_rate"])),
    }


def _percentile(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# 標準增強是否把瑕疵切掉的診斷",
        "",
        "> 由 `scripts/diagnose_augmentation_mask_loss.py` 產生。",
        (
            "> **只讀 train partition**：重放 trainer 實際會抽到的同一組 draw"
            "（同 transform、同 `_stable_seed` 序列），量測 mask 是否還在。"
        ),
        "> 不載入模型、不重算任何指標、不做選擇。",
        "",
        f"每張瑕疵圖重放 `{payload['draws']}` 次。",
        "",
        "| 物件 | 瑕疵圖 | 原圖瑕疵面積中位數 | 空 mask 比率 | 曾經整個切掉的圖 | 保留面積比（平均） |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in payload["objects"]:
        median = summary["source_defect_area_fraction"]["p50"]
        lines.append(
            f"| {summary['object']} | {summary['defect_images']} "
            f"| {median * 100:.4f}% "
            f"| {summary['empty_mask_rate'] * 100:.2f}% "
            f"| {summary['images_that_ever_lose_the_defect']}/{summary['defect_images']} "
            f"| {summary['retained_area_ratio_mean']:.3f} |"
        )
    lines += ["", "## 每個物件最容易被切掉的瑕疵圖", ""]
    for summary in payload["objects"]:
        lines += [f"### {summary['object']}", ""]
        for item in summary["samples"][:5]:
            lines.append(
                f"- `{item['sample_id']}`：原圖瑕疵佔 "
                f"`{item['source_defect_area_fraction'] * 100:.4f}%`，"
                f"空 mask `{item['empty_mask_rate'] * 100:.1f}%`，"
                f"保留面積比 `{item['retained_area_ratio_mean']:.3f}`"
            )
        lines.append("")

    lines += [
        "## 訓練期的 Dice 項有沒有啟動",
        "",
        "BCE 在每個 run 都會下降——把整張預測成背景就能拿到低 BCE。",
        "只有 **Dice 項**能顯示模型是否真的學會與瑕疵重疊。",
        "",
        "| 物件 | 組別 | 首窗 dice_loss | 末窗 dice_loss | 改善量 | Dice 項啟動 | 最終 Dice |",
        "| --- | --- | ---: | ---: | ---: | :---: | ---: |",
    ]
    for comparison in payload["augmentation_comparison"]:
        for group in ("real_only", "std_aug"):
            trajectory = comparison["trajectories"][group]
            engaged = comparison[
                "dice_term_engaged_with_augmentation"
                if group == "std_aug"
                else "dice_term_engaged_without_augmentation"
            ]
            lines.append(
                f"| {comparison['object']} | {group} "
                f"| {trajectory['dice_loss_first_window']:.4f} "
                f"| {trajectory['dice_loss_last_window']:.4f} "
                f"| {trajectory['dice_loss_improvement']:+.4f} "
                f"| {'是' if engaged else '**否**'} "
                f"| {trajectory['final_dice']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/segmenter.yaml"))
    parser.add_argument("--runs-root", type=Path, default=Path("results/colab/segmentation"))
    parser.add_argument("--run-group", default="std_aug")
    parser.add_argument("--object", dest="objects", action="append")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss-window", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/augmentation_mask_loss.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/augmentation_mask_loss.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_paths(args.paths)
    config = load_config(args.config)
    objects = tuple(args.objects) if args.objects else ("pcb1", "capsules")

    summaries = []
    for object_name in objects:
        run_dir = args.runs_root / object_name / "runs" / (
            f"m18_{args.run_group}_{object_name}_seed{args.seed}"
        )
        manifest = load_json(run_dir / "data_manifest.json")
        require(
            manifest["standard_augmentation"] is True,
            f"{run_dir.name}: this run does not use standard augmentation",
        )
        samples = defect_samples(manifest)
        replays = [
            replay_sample(
                sample,
                paths=paths,
                config=config,
                seed=args.seed,
                draws=args.draws,
            )
            for sample in samples
        ]
        summaries.append(summarise(object_name, replays))
        print(
            f"{object_name}: {summaries[-1]['empty_mask_rate'] * 100:.2f}% of draws lose the defect"
        )

    comparisons = [
        compare_augmentation(
            object_name,
            runs_root=args.runs_root,
            seed=args.seed,
            window=args.loss_window,
        )
        for object_name in objects
    ]
    for comparison in comparisons:
        print(
            f"{comparison['object']}: Dice term engaged "
            f"without aug={comparison['dice_term_engaged_without_augmentation']}, "
            f"with aug={comparison['dice_term_engaged_with_augmentation']}"
        )

    payload = {
        "status": "passed",
        "schema_version": 1,
        "draws": args.draws,
        "seed": args.seed,
        "loss_window": args.loss_window,
        "augmentation": config["standard_augmentation"],
        "objects": summaries,
        "augmentation_comparison": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(
        args.output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    write_text_lf(args.report, render_markdown(payload))
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
