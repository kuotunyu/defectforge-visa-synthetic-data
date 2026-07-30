from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.common.integrity import sha256_file
from src.inference.demo_gradio import (
    DemoError,
    build_parser,
    render_outputs,
    select_best_checkpoint,
    verify_checkpoint,
)


def _checkpoint(root: Path, *, role: str, model_sha256: str | None = None) -> Path:
    run = root / role
    weight = run / ("final/model.safetensors" if role == "segmenter" else "model.safetensors")
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"safe-tensor-fixture")
    observed = sha256_file(weight)
    report = {
        "status": "passed",
        "smoke": False,
        "mode": "final",
        "object": "pcb1",
        "model_sha256": model_sha256 or observed,
    }
    (run / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
    return weight


def _formal_checkpoint(run: Path, *, role: str, report: dict[str, object]) -> Path:
    weight = run / ("final/model.safetensors" if role == "segmenter" else "model.safetensors")
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"formal-safe-tensor-fixture")
    report = {
        "status": "passed",
        "smoke": False,
        "mode": "final",
        "model_sha256": sha256_file(weight),
        **report,
    }
    (run / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
    return weight


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("role", ["classifier", "segmenter"])
def test_verify_checkpoint_binds_weight_to_report(tmp_path: Path, role: str) -> None:
    weight = _checkpoint(tmp_path, role=role)
    verified = verify_checkpoint(weight.parent if role == "classifier" else weight.parent.parent, role=role)
    assert verified.weight == weight
    assert verified.object_name == "pcb1"
    assert verified.model_sha256 == sha256_file(weight)


def test_verify_checkpoint_rejects_hash_mismatch(tmp_path: Path) -> None:
    weight = _checkpoint(tmp_path, role="classifier", model_sha256="0" * 64)
    with pytest.raises(DemoError, match="hash"):
        verify_checkpoint(weight, role="classifier")


def test_render_outputs_has_stable_shapes_and_probabilities() -> None:
    image = np.full((5, 7, 3), 80, dtype=np.uint8)
    probability = np.linspace(0.0, 1.0, 35, dtype=np.float32).reshape(5, 7)
    labels, mask, heatmap, latency = render_outputs(
        image,
        anomaly_probability=0.75,
        pixel_probability=probability,
        threshold=0.5,
        elapsed_ms=12.34,
    )
    assert labels == {"Defect": 0.75, "Normal": 0.25}
    assert mask.shape == (5, 7)
    assert set(np.unique(mask)) == {0, 255}
    assert heatmap.shape == image.shape
    assert heatmap.dtype == np.uint8
    assert "12.3 ms" in latency
    assert "mask coverage" in latency


def test_cli_is_local_and_private_by_default() -> None:
    args = build_parser().parse_args(
        ["--cls-ckpt", "classifier", "--seg-ckpt", "segmenter"]
    )
    assert args.port == 7860
    assert args.share is False
    assert args.inbrowser is False


def test_cli_can_request_automatic_object_selection() -> None:
    args = build_parser().parse_args(["--object", "pcb1"])
    assert args.object == "pcb1"
    assert args.cls_ckpt is None
    assert args.seg_ckpt is None


def test_classifier_selection_is_deterministic_and_report_bound(tmp_path: Path) -> None:
    seed_42_groups = (
        "real_only",
        "std_aug",
        "unfiltered_syn",
        "filtered_syn",
        "full_real",
        "real_20",
        "syn_125",
        "syn_250",
        "src_procedural",
        "src_copypaste",
        "src_diffusion",
        "procedural_norealstats",
        "bucket_original",
        "bucket_searched",
        "base_sdxl",
    )
    rows = [
        {
            "run_name": f"m16_{group}_{object_name}_seed_42",
            "run_signature": f"signature-{group}-{object_name}-42",
            "canonical_group": group,
            "object": object_name,
            "seed": "42",
            "macro_f1": "0.9" if group == "full_real" and object_name == "pcb1" else "0.5",
            "auroc": "0.8",
        }
        for group in seed_42_groups
        for object_name in ("pcb1", "capsules")
    ]
    for group in ("real_only", "filtered_syn"):
        for object_name in ("pcb1", "capsules"):
            for seed in (43, 44):
                rows.append(
                    {
                        "run_name": f"m16_{group}_{object_name}_seed_{seed}",
                        "run_signature": f"signature-{group}-{object_name}-{seed}",
                        "canonical_group": group,
                        "object": object_name,
                        "seed": str(seed),
                        "macro_f1": "1.0",
                        "auroc": "1.0",
                    }
                )
    results = tmp_path / "classification.csv"
    _write_csv(results, rows)
    run_name = "m16_full_real_pcb1_seed_42"
    run_root = tmp_path / "runs"
    _formal_checkpoint(
        run_root / run_name,
        role="classifier",
        report={
            "object": "pcb1",
            "seed": 42,
            "run_name": run_name,
            "run_signature": "signature-full_real-pcb1-42",
            "canonical_group": "full_real",
            "metrics": {"macro_f1": 0.9, "auroc": 0.8},
        },
    )
    selection = select_best_checkpoint(
        results_path=results,
        runs_root=run_root,
        object_name="pcb1",
        role="classifier",
    )
    assert selection.run_name == run_name
    assert selection.group_name == "full_real"
    assert selection.primary_value == pytest.approx(0.9)


def test_segmenter_selection_ignores_logical_alias(tmp_path: Path) -> None:
    groups = (
        "real_only",
        "std_aug",
        "unfiltered_syn",
        "filtered_syn",
        "full_real",
        "procedural_only",
        "copypaste_only",
        "diffusion_only",
        "all_mixed",
    )
    rows = []
    for group in groups:
        for object_name in ("pcb1", "capsules"):
            # ADR-032 replicated every group across three seeds; the demo must still bind
            # to the seed-42 anchor, so give the other seeds a strictly better Dice.
            for seed in (42, 43, 44):
                canonical = "filtered_syn" if group == "all_mixed" else group
                physical = group != "all_mixed"
                run_group = canonical
                if seed != 42:
                    dice = "0.99"
                elif group == "all_mixed":
                    dice = "0.95"
                elif group == "std_aug" and object_name == "pcb1":
                    dice = "0.8"
                else:
                    dice = "0.5"
                rows.append(
                    {
                        "run_name": f"m18_{run_group}_{object_name}_seed{seed}",
                        "run_signature": f"signature-{run_group}-{object_name}-{seed}",
                        "logical_group": group,
                        "canonical_group": canonical,
                        "physical_run": str(physical).lower(),
                        "object": object_name,
                        "seed": str(seed),
                        "dice": dice,
                        "aupro": "0.7",
                    }
                )
    results = tmp_path / "segmentation.csv"
    _write_csv(results, rows)
    run_name = "m18_std_aug_pcb1_seed42"
    run_root = tmp_path / "runs"
    _formal_checkpoint(
        run_root / run_name,
        role="segmenter",
        report={
            "object": "pcb1",
            "seed": 42,
            "run_name": run_name,
            "run_signature": "signature-std_aug-pcb1-42",
            "canonical_group": "std_aug",
            "metrics": {"dice": 0.8, "aupro": 0.7},
        },
    )
    selection = select_best_checkpoint(
        results_path=results,
        runs_root=run_root,
        object_name="pcb1",
        role="segmenter",
    )
    assert selection.run_name == run_name
    assert selection.group_name == "std_aug"


def test_direct_script_entrypoint_exposes_help() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "src/inference/demo_gradio.py", "--help"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--cls-ckpt" in result.stdout
    assert "--seg-ckpt" in result.stdout
    assert "--object" in result.stdout
