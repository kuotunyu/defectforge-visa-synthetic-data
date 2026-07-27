"""M14 CSV, Markdown, figures, and exact embedded-summary verification."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from src.common.integrity import sha256_file
from src.evaluation.quality_data import CropEntry
from src.evaluation.quality_pipeline import QualityRun

CSV_FIELDS = (
    "view",
    "input_name",
    "object",
    "defect_type",
    "real_scope",
    "n_real",
    "n_generated",
    "status",
    "nn_mean",
    "nn_median",
    "nn_p05",
    "nn_p95",
    "mnn_score",
    "kid",
    "fid",
)


class QualityReportError(RuntimeError):
    """A quality report artifact is malformed or inconsistent."""


def canonical_summary(run: QualityRun) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline_version": "0.1.0",
        "sanity_passed": run.sanity_passed,
        "source_audit": {
            "paths_checked": run.source_audit.paths_checked,
            "hashes_checked": run.source_audit.hashes_checked,
            "blocklist_hits": run.source_audit.blocklist_hits,
            "sha256": run.source_audit.sha256,
        },
        "model_revision": run.model_revision,
        "clean_fid_version": run.clean_fid_version,
        "metric_policy": {
            "formal_kid": "unbiased_degree3_polynomial_mmd",
            "sanity_kid": "biased_degree3_polynomial_mmd",
            "fid": "clean_fid_features_exact_low_rank",
        },
        "caches": {
            "generated_dino": {
                "path": run.generated_dino_cache,
                "sha256": sha256_file(Path(run.generated_dino_cache)),
            },
            "reference_dino": {
                "path": run.reference_dino_cache,
                "sha256": sha256_file(Path(run.reference_dino_cache)),
            },
            "clean_features": {
                "path": run.clean_feature_cache,
                "sha256": sha256_file(Path(run.clean_feature_cache)),
            },
        },
        "sanity": run.sanity,
        "rows": run.rows,
    }


def summary_sha256(summary: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        summary,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def render_markdown(summary: Mapping[str, Any]) -> str:
    digest = summary_sha256(summary)
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    lines = [
        "# M14 generation quality",
        "",
        f"<!-- generation-quality-sha256: {digest} -->",
        f"<!-- generation-quality-json: {payload} -->",
        "",
        (
            "All metrics are computed on mask-centered defect crops. DINOv2 CLS embeddings "
            "provide cosine `nn_score` and mutual-nearest-neighbor coverage. KID and FID use "
            "clean-fid 0.1.35 clean-mode Inception features."
        ),
        "",
        (
            "**KID is the primary image metric. FID is listed only for reference because the "
            "per-type real sets contain as few as three crops, making covariance estimates "
            "statistically unreliable.**"
        ),
        (
            "Formal generated-vs-real rows use unbiased KID. The identity/noise sanity table "
            "uses the biased polynomial MMD estimator: unlike the unbiased finite-sample "
            "U-statistic, it is exactly zero for a feature set compared with itself."
        ),
        "",
        (
            "Both unfiltered and filtered memberships are reported so filtering impact remains "
            "auditable instead of being inferred from hand-picked images."
        ),
        "",
        "## Mandatory sanity gate",
        "",
        f"- Overall: `{'passed' if summary['sanity_passed'] else 'FAILED'}`",
        f"- Test-blocklist hits: `{summary['source_audit']['blocklist_hits']}`",
        f"- Source paths checked: `{summary['source_audit']['paths_checked']}`",
        f"- Source-audit SHA-256: `{summary['source_audit']['sha256']}`",
        "",
        (
            "| Object | Type | n | Self NN min | Self mNN | Self KID | Self FID | "
            "Noise NN mean | tau_low | Noise KID | Noise FID | Passed |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for check in summary["sanity"]:
        lines.append(
            f"| {check['object']} | {check['defect_type']} | {check['n']} | "
            f"{check['self_nn_min']:.8f} | {check['self_mnn']:.6f} | "
            f"{check['self_kid']:.8f} | {check['self_fid']:.8f} | "
            f"{check['noise_nn_mean']:.6f} | {check['tau_low']:.6f} | "
            f"{check['noise_kid']:.6f} | {check['noise_fid']:.3f} | "
            f"{'yes' if check['passed'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Per-generator and per-type metrics",
            "",
            (
                "| View | Input | Object | Type | Real scope | Real | Generated | NN mean | "
                "NN median | mNN | KID | FID | Status |"
            ),
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["rows"]:
        if row["status"] == "empty":
            metric_values = ("—", "—", "—", "—", "—")
        else:
            metric_values = (
                f"{row['nn_mean']:.6f}",
                f"{row['nn_median']:.6f}",
                f"{row['mnn_score']:.6f}",
                f"{row['kid']:.6f}",
                f"{row['fid']:.3f}",
            )
        lines.append(
            f"| {row['view']} | {row['input_name']} | {row['object']} | "
            f"{row['defect_type']} | {row['real_scope']} | {row['n_real']} | "
            f"{row['n_generated']} | "
            f"{metric_values[0]} | {metric_values[1]} | {metric_values[2]} | "
            f"{metric_values[3]} | {metric_values[4]} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Visual evidence",
            "",
            (
                "- `reports/figures/generation_quality_nn.png`: NN distributions for each "
                "source/object before and after filtering."
            ),
            (
                "- `reports/figures/generation_quality_real_vs_generated.png`: deterministic "
                "real crop plus median-NN filtered crop from each generator."
            ),
            "",
            f"Machine-readable summary SHA-256: `{digest}`",
            "",
        ]
    )
    return "\n".join(lines)


def embedded_summary(markdown: str) -> dict[str, Any]:
    prefix = "<!-- generation-quality-json: "
    for line in markdown.splitlines():
        if line.startswith(prefix) and line.endswith(" -->"):
            return json.loads(line[len(prefix) : -4])
    raise QualityReportError("Generation quality report has no embedded summary")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise QualityReportError(f"Temporary output already exists: {temporary}")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise QualityReportError(f"Temporary CSV already exists: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "" if row[field] is None else row[field]
                    for field in CSV_FIELDS
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    _atomic_text(path, render_markdown(summary))


def write_validation(path: Path, summary: Mapping[str, Any]) -> None:
    payload = {
        "status": "passed" if summary["sanity_passed"] else "failed",
        "summary_sha256": summary_sha256(summary),
        **summary,
    }
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def plot_nn_distributions(
    distributions: Mapping[str, Sequence[float]],
    output: Path,
) -> None:
    keys = sorted(key for key in distributions if key.endswith("|__all__"))
    values = [distributions[key] for key in keys]
    labels = [
        key.replace("stageA_", "A-").replace("stageB_sd2/searched", "B-SD2")
        for key in keys
    ]
    figure, axis = plt.subplots(figsize=(16, 7))
    axis.boxplot(values, tick_labels=labels, showfliers=False)
    axis.set_ylabel("DINOv2 nearest-real cosine")
    axis.set_title("Defect-crop NN distributions: unfiltered vs filtered")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=55, labelsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _tile(path: Path | None, label: str, *, size: int = 256) -> Image.Image:
    tile = Image.new("RGB", (size, size), (20, 23, 28))
    content_height = size - 42
    if path is not None:
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            image.thumbnail((size, content_height), Image.Resampling.LANCZOS)
            tile.paste(
                image,
                ((size - image.width) // 2, (content_height - image.height) // 2),
            )
    draw = ImageDraw.Draw(tile)
    draw.text(
        (8, content_height + 10),
        label,
        fill=(240, 240, 240),
        font=ImageFont.load_default(),
    )
    return tile


def comparison_grid(
    crop_root: Path,
    entries: Sequence[CropEntry],
    representatives: Mapping[str, str],
    input_names: Sequence[str],
    output: Path,
) -> None:
    groups = sorted(
        {(entry.object_name, entry.defect_type) for entry in entries if entry.kind == "real"}
    )
    columns = 1 + len(input_names)
    tile_size = 256
    canvas = Image.new(
        "RGB",
        (columns * tile_size, len(groups) * tile_size),
        (14, 16, 20),
    )
    for row_index, (object_name, defect_type) in enumerate(groups):
        real = next(
            entry
            for entry in entries
            if entry.kind == "real"
            and entry.object_name == object_name
            and entry.defect_type == defect_type
        )
        canvas.paste(
            _tile(
                crop_root / real.relative_path,
                f"REAL {object_name}/{defect_type}",
                size=tile_size,
            ),
            (0, row_index * tile_size),
        )
        for column_index, input_name in enumerate(input_names, start=1):
            key = f"filtered|{input_name}|{object_name}|{defect_type}"
            relative_path = representatives.get(key)
            if relative_path is None and input_name == "stageA_procedural":
                relative_path = representatives.get(
                    f"filtered|{input_name}|{object_name}|__all__"
                )
            label = input_name if relative_path is not None else f"{input_name} EMPTY"
            canvas.paste(
                _tile(
                    None if relative_path is None else crop_root / relative_path,
                    label,
                    size=tile_size,
                ),
                (column_index * tile_size, row_index * tile_size),
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
