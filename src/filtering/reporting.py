"""Deterministic M13 summaries, Markdown, and visual audits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.filtering.rules import RULE_ORDER, RejectReason

REASONS_BY_RULE = {
    "roi": {RejectReason.ROI_OVERFLOW},
    "area": {RejectReason.AREA_OUT_OF_RANGE},
    "aspect": {RejectReason.ASPECT_OUT_OF_RANGE},
    "phash": {RejectReason.PHASH_DUPLICATE},
    "dinov2": {
        RejectReason.NN_TOO_LOW,
        RejectReason.NN_TOO_HIGH_COPY,
        RejectReason.EMBEDDING_OUTLIER,
    },
    "seam": {RejectReason.SEAM_POOR},
}


class FilterReportError(RuntimeError):
    """Published filter metadata cannot produce a trustworthy report."""


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise FilterReportError(f"Invalid JSON at {path}:{line_number}") from error
            filter_value = record.get("filter")
            if not isinstance(filter_value, dict) or not isinstance(
                filter_value.get("reject_reasons"),
                list,
            ):
                raise FilterReportError(f"Missing filter decision at {path}:{line_number}")
            records.append(record)
    return records


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the exact machine-readable payload rendered in the report."""

    valid_reasons = {str(reason) for reason in RejectReason}
    reason_counts: Counter[str] = Counter()
    first_reason_counts: Counter[str] = Counter()
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    accepted = 0
    for record in records:
        decision = record["filter"]
        reasons = [str(value) for value in decision["reject_reasons"]]
        unknown = set(reasons) - valid_reasons
        if unknown:
            raise FilterReportError(f"Unknown reject reasons: {sorted(unknown)}")
        passed = bool(decision["passed"])
        if passed != (not reasons):
            raise FilterReportError(f"passed/reasons disagree for {record['sample_id']}")
        accepted += int(passed)
        reason_counts.update(reasons)
        if reasons:
            first_reason_counts[reasons[0]] += 1
        groups[
            (
                str(decision["input_name"]),
                str(record["object"]),
                str(record["generator"]),
                str(record["defect_type"]),
            )
        ].append(record)

    rows: list[dict[str, Any]] = []
    for (input_name, object_name, generator, defect_type), group in sorted(groups.items()):
        counts = Counter(
            reason
            for record in group
            for reason in record["filter"]["reject_reasons"]
        )
        accumulated: set[str] = set()
        group_funnel: dict[str, int] = {}
        for rule in RULE_ORDER:
            accumulated.update(str(reason) for reason in REASONS_BY_RULE[rule])
            group_funnel[f"after_{rule}"] = sum(
                not (set(record["filter"]["reject_reasons"]) & accumulated)
                for record in group
            )
        rows.append(
            {
                "input": input_name,
                "object": object_name,
                "generator": generator,
                "defect_type": defect_type,
                "total": len(group),
                "accepted": sum(bool(record["filter"]["passed"]) for record in group),
                **group_funnel,
                **{str(reason): counts[str(reason)] for reason in RejectReason},
            }
        )

    survivors = len(records)
    funnel: list[dict[str, int | str]] = []
    accumulated: set[str] = set()
    for rule in RULE_ORDER:
        accumulated.update(str(reason) for reason in REASONS_BY_RULE[rule])
        survivors = sum(
            not (set(record["filter"]["reject_reasons"]) & accumulated)
            for record in records
        )
        funnel.append({"rule": rule, "survivors": survivors})

    thresholds: dict[str, Any] = {}
    for record in records:
        thresholds.setdefault(
            str(record["object"]),
            record["filter"]["thresholds"],
        )
    return {
        "schema_version": 1,
        "total": len(records),
        "accepted": accepted,
        "rejected": len(records) - accepted,
        "first_reason_counts": dict(sorted(first_reason_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "funnel": funnel,
        "thresholds": dict(sorted(thresholds.items())),
        "rows": rows,
    }


def summary_sha256(summary: Mapping[str, Any]) -> str:
    payload = json.dumps(
        summary,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render a compact human report with an exact embedded verification payload."""

    digest = summary_sha256(summary)
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    lines = [
        "# M13 synthetic quality filter",
        "",
        f"<!-- filter-summary-sha256: {digest} -->",
        f"<!-- filter-summary-json: {payload} -->",
        "",
        (
            "The six rules run in the locked order: ROI, area, aspect, pHash, "
            "DINOv2, then seam. Counts below are reconstructed from published "
            "`unfiltered/metadata.jsonl`; accepted samples are hardlinked into `filtered/`."
        ),
        "",
        "## Outcome",
        "",
        f"- Total: {summary['total']}",
        f"- Accepted: {summary['accepted']}",
        f"- Rejected: {summary['rejected']}",
        f"- Summary SHA-256: `{digest}`",
        "",
        "## Funnel",
        "",
        "| Rule | Survivors |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {row['rule']} | {row['survivors']} |"
        for row in summary["funnel"]
    )
    lines.extend(
        [
            "",
            "## Reject reasons",
            "",
            (
                "A sample may trigger multiple rules. `First rejects` classifies each rejected "
                "sample once by the locked funnel order; `All triggers` counts every triggered "
                "rule and can therefore exceed the rejected total."
            ),
            "",
            "| Reason | First rejects | All triggers |",
            "|---|---:|---:|",
        ]
    )
    reason_counts = summary["reason_counts"]
    first_reason_counts = summary["first_reason_counts"]
    lines.extend(
        f"| {reason} | {first_reason_counts.get(str(reason), 0)} | "
        f"{reason_counts.get(str(reason), 0)} |"
        for reason in RejectReason
    )
    lines.extend(
        [
            "",
            "## Generator and defect-type detail",
            "",
            (
                "| Input | Object | Generator | Type | Generated | ROI | Area | "
                "Aspect | pHash | DINOv2 | Seam | Final | Pass rate |"
            ),
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        pass_rate = row["accepted"] / row["total"]
        lines.append(
            f"| {row['input']} | {row['object']} | {row['generator']} | "
            f"{row['defect_type']} | {row['total']} | {row['after_roi']} | "
            f"{row['after_area']} | {row['after_aspect']} | {row['after_phash']} | "
            f"{row['after_dinov2']} | {row['after_seam']} | {row['accepted']} | "
            f"{pass_rate:.2%} |"
        )
    lines.extend(
        [
            "",
            (
                "The table above contains survivors after each rule. Exact per-group all-trigger "
                "counts remain embedded in the machine-readable summary and are checked by "
                "`scripts/verify_filter_report.py`."
            ),
        ]
    )
    lines.extend(["", "## Locked thresholds", ""])
    for object_name, thresholds in summary["thresholds"].items():
        lines.append(f"### {object_name}")
        lines.append("")
        for name, value in thresholds.items():
            lines.append(f"- `{name}`: {value}")
        lines.append("")
    lines.extend(
        [
            "## Visual audit",
            "",
            (
                "The accepted and rejected sheets are deterministic, evenly spaced samples "
                "from their respective populations. They are audit views, not hand-picked examples."
            ),
            "",
            "- `reports/figures/filter_accepted.png`",
            "- `reports/figures/filter_rejected.png`",
            "",
        ]
    )
    return "\n".join(lines)


def embedded_summary(markdown: str) -> dict[str, Any]:
    prefix = "<!-- filter-summary-json: "
    for line in markdown.splitlines():
        if line.startswith(prefix) and line.endswith(" -->"):
            return json.loads(line[len(prefix) : -4])
    raise FilterReportError("Report has no embedded filter summary")


def deterministic_selection(
    records: Sequence[Mapping[str, Any]],
    *,
    passed: bool,
    count: int,
) -> list[Mapping[str, Any]]:
    candidates = [record for record in records if bool(record["filter"]["passed"]) == passed]
    if count < 1 or not candidates:
        return []
    if len(candidates) <= count:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[int(index)] for index in indices]


def _annotated_tile(root: Path, record: Mapping[str, Any], *, tile_size: int) -> Image.Image:
    image_path = root / str(record["image_path"])
    mask_path = root / str(record["mask_path"])
    with Image.open(image_path) as image_handle:
        image = np.asarray(image_handle.convert("RGB"))
    with Image.open(mask_path) as mask_handle:
        mask = np.asarray(mask_handle.convert("L")) > 0
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    outlined = image.copy()
    cv2.drawContours(outlined, contours, -1, (255, 48, 48), thickness=4)
    content_height = tile_size - 64
    thumbnail = Image.fromarray(outlined).copy()
    thumbnail.thumbnail((tile_size, content_height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_size, tile_size), (24, 27, 32))
    tile.paste(
        thumbnail,
        ((tile_size - thumbnail.width) // 2, (content_height - thumbnail.height) // 2),
    )
    draw = ImageDraw.Draw(tile)
    reasons = record["filter"]["reject_reasons"]
    label = (
        f"{record['object']} / {record['defect_type']}\n"
        f"{'ACCEPT' if not reasons else ', '.join(reasons[:2])}"
    )
    draw.text((8, content_height + 5), label, fill=(240, 240, 240), font=ImageFont.load_default())
    return tile


def contact_sheet(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    passed: bool,
    count: int,
    columns: int = 4,
    tile_size: int = 320,
) -> None:
    selected = deterministic_selection(records, passed=passed, count=count)
    if not selected:
        raise FilterReportError("No records available for requested contact sheet")
    rows = (len(selected) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_size, rows * tile_size), (18, 20, 24))
    for index, record in enumerate(selected):
        tile = _annotated_tile(root, record, tile_size=tile_size)
        canvas.paste(tile, ((index % columns) * tile_size, (index // columns) * tile_size))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
