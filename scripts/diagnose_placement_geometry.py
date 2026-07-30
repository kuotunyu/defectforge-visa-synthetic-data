"""Measure whether M9 synthetic placements match the real defect distribution.

v3 (ADR-036) found that, on the only object where validation Macro-F1 can discriminate,
pasting real defect texture at synthetic locations costs far more than replacing the texture
with a generated one. That points at the placement bundle, so this script measures the
placement stage directly instead of inferring it from downstream metrics.

Three families of measurement, per object:

1. **Geometry** — the placed mask area distribution against the real defect component area
   distribution, plus the affine scale and rotation actually applied.
2. **Containment** — every placement source is checked against the frozen test blocklist, so
   the diagnosis itself cannot read test data.
3. **Site appearance** — pixel statistics under the placed mask on its background image,
   against pixel statistics under real defect masks on their own images. Real defects sit on
   the object by definition, so a systematic gap here means the legal ROI admits pixels that
   do not look like defect sites.

Every conclusion in the report is derived from the measurements. Following ADR-034, no claim
is pre-written as a string.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import write_text_lf  # isort: skip
from src.common.paths import load_paths  # isort: skip

# The real p5-p95 band is the reference interval; it is a description of the real data, not a
# tuned threshold, and it is reported alongside every derived fraction.
LOW_QUANTILE = 0.05
HIGH_QUANTILE = 0.95
# Width of the defect-free band measured around each mask. See context_ring for why the
# comparison has to happen outside the mask rather than under it.
RING_WIDTH_PX = 12


class PlacementDiagnosisError(RuntimeError):
    """Raised when the placement stage cannot be measured safely."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlacementDiagnosisError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values: Sequence[float], fraction: float) -> float:
    require(bool(values), "Cannot take a quantile of an empty series")
    ordered = sorted(float(value) for value in values)
    return ordered[int(fraction * (len(ordered) - 1))]


def load_placements(root: Path, object_name: str) -> list[dict[str, Any]]:
    path = root / "synthetic/placements" / object_name / "placements.jsonl"
    require(path.is_file(), f"Missing placement records: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(bool(records), f"Placement records are empty: {path}")
    for record in records:
        require(record["object"] == object_name, f"Placement object changed: {path}")
    return records


def assert_sources_are_not_test(
    records: Sequence[Mapping[str, Any]],
    *,
    raw_root: Path,
    blocklist: set[str],
) -> dict[str, Any]:
    """Fail closed if the placement stage ever touched a frozen test file."""
    sources = sorted(
        {str(record["source_mask"]) for record in records}
        | {str(record["background_image"]) for record in records}
    )
    hits = [name for name in sources if sha256_file(raw_root / name) in blocklist]
    require(not hits, f"Placement source is in the test blocklist: {hits[:3]}")
    return {"checked_sources": len(sources), "blocklist_hits": 0}


def _components(mask: np.ndarray) -> list[int]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    return [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)]


def context_ring(mask: np.ndarray, *, width_px: int = RING_WIDTH_PX) -> np.ndarray:
    """The band just outside the mask.

    Pixels *under* a real mask already contain the defect, while pixels under a placed mask
    are still clean background, so measuring under the masks compares two different things.
    The surrounding band is defect-free on both sides, which makes the comparison symmetric:
    it asks what kind of surface the defect sits on, not what the defect looks like.
    """
    binary = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width_px * 2 + 1,) * 2)
    return (cv2.dilate(binary, kernel) > 0) & (binary == 0)


def _site_statistics(image: np.ndarray, mask: np.ndarray) -> tuple[float, float] | None:
    selected = context_ring(mask)
    if not selected.any():
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (
        float(np.median(hsv[:, :, 1][selected])),
        float(np.median(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[selected])),
    )


def measure_object(
    *,
    records: Sequence[Mapping[str, Any]],
    raw_root: Path,
    placement_root: Path,
) -> dict[str, Any]:
    real_areas: list[int] = []
    real_saturation: list[float] = []
    real_intensity: list[float] = []
    for relative in sorted({str(record["source_mask"]) for record in records}):
        mask = cv2.imread(str(raw_root / relative), cv2.IMREAD_GRAYSCALE)
        require(mask is not None, f"Unreadable real mask: {relative}")
        real_areas.extend(_components(mask))
        image_relative = relative.replace("/Masks/", "/Images/").replace(".png", ".JPG")
        image = cv2.imread(str(raw_root / image_relative), cv2.IMREAD_COLOR)
        if image is None:
            continue
        statistics = _site_statistics(image, mask)
        if statistics is not None:
            real_saturation.append(statistics[0])
            real_intensity.append(statistics[1])

    by_background: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_background[str(record["background_image"])].append(record)

    synthetic_saturation: list[float] = []
    synthetic_intensity: list[float] = []
    for relative, group in by_background.items():
        image = cv2.imread(str(raw_root / relative), cv2.IMREAD_COLOR)
        if image is None:
            continue
        for record in group:
            mask = cv2.imread(
                str(placement_root / str(record["mask_path"])),
                cv2.IMREAD_GRAYSCALE,
            )
            if mask is None:
                continue
            statistics = _site_statistics(image, mask)
            if statistics is not None:
                synthetic_saturation.append(statistics[0])
                synthetic_intensity.append(statistics[1])

    synthetic_areas = [int(record["mask_area_px"]) for record in records]
    low = quantile(real_areas, LOW_QUANTILE)
    high = quantile(real_areas, HIGH_QUANTILE)
    in_band = sum(low <= area <= high for area in synthetic_areas)
    return {
        "placements": len(records),
        "real_components": len(real_areas),
        "area": {
            "real_median": quantile(real_areas, 0.5),
            "synthetic_median": quantile(synthetic_areas, 0.5),
            "median_ratio": quantile(synthetic_areas, 0.5) / quantile(real_areas, 0.5),
            "real_band": [low, high],
            "synthetic_in_real_band_fraction": in_band / len(synthetic_areas),
            "synthetic_min": min(synthetic_areas),
            "synthetic_min_exceeds_real_median": min(synthetic_areas)
            > quantile(real_areas, 0.5),
        },
        "transform": {
            "scale_median": quantile(
                [record["affine"]["scale"] for record in records], 0.5
            ),
            "scale_below_half_fraction": sum(
                record["affine"]["scale"] < 0.5 for record in records
            )
            / len(records),
            "abs_rotation_median": quantile(
                [abs(record["affine"]["rotation_deg"]) for record in records], 0.5
            ),
        },
        "site_appearance": {
            "real_masks_measured": len(real_saturation),
            "synthetic_masks_measured": len(synthetic_saturation),
            "real_saturation_median": quantile(real_saturation, 0.5)
            if real_saturation
            else None,
            "synthetic_saturation_median": quantile(synthetic_saturation, 0.5)
            if synthetic_saturation
            else None,
            "real_intensity_median": quantile(real_intensity, 0.5)
            if real_intensity
            else None,
            "synthetic_intensity_median": quantile(synthetic_intensity, 0.5)
            if synthetic_intensity
            else None,
        },
    }


def derive_findings(objects: Mapping[str, Any]) -> dict[str, Any]:
    """Every statement here is computed; nothing is asserted in advance (ADR-034)."""
    out_of_band = sorted(
        name
        for name, item in objects.items()
        if item["area"]["synthetic_in_real_band_fraction"] < 0.5
    )
    oversized = sorted(
        name
        for name, item in objects.items()
        if item["area"]["synthetic_min_exceeds_real_median"]
    )
    in_band = sorted(
        name
        for name, item in objects.items()
        if item["area"]["synthetic_in_real_band_fraction"] >= 0.95
    )
    return {
        "objects_with_majority_out_of_band_area": out_of_band,
        "objects_whose_smallest_placement_exceeds_the_real_median": oversized,
        "objects_with_area_fully_in_band": in_band,
        "area_explains_every_object": not in_band,
    }


def build_report(payload: Mapping[str, Any]) -> str:
    objects = payload["objects"]
    findings = payload["findings"]
    lines = [
        "# M9 放置階段幾何診斷",
        "",
        (
            "由 `scripts/diagnose_placement_geometry.py` 產生。這是對**已完成**產物的事後量測："
            "不重新生成、不訓練、不讀 frozen test。動機見 "
            "[ADR-036](../docs/decisions.md#adr-036)。"
        ),
        "",
        "## 防洩漏",
        "",
    ]
    for name, item in objects.items():
        guard = item["blocklist"]
        lines.append(
            f"- `{name}`：檢查 {guard['checked_sources']} 個來源檔，"
            f"test blocklist 命中 **{guard['blocklist_hits']}**"
        )
    lines += [
        "",
        "## 面積：合成放置 vs 真實瑕疵元件",
        "",
        (
            f"參考區間是**真實**面積的 p{LOW_QUANTILE * 100:.0f}–p{HIGH_QUANTILE * 100:.0f}，"
            "它是對真實資料的描述，不是調出來的門檻。"
        ),
        "",
        "| 物件 | 真實中位數 | 合成中位數 | 比值 | 真實 p5–p95 | 合成落在區間內 | 合成最小值 > 真實中位數 |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for name, item in objects.items():
        area = item["area"]
        band = area["real_band"]
        lines.append(
            f"| {name} | {area['real_median']:.0f} | {area['synthetic_median']:.0f} "
            f"| {area['median_ratio']:.2f}× | {band[0]:.0f}–{band[1]:.0f} "
            f"| {area['synthetic_in_real_band_fraction'] * 100:.1f}% "
            f"| {'**是**' if area['synthetic_min_exceeds_real_median'] else '否'} |"
        )
    lines += [
        "",
        "## 實際套用的變換",
        "",
        "| 物件 | scale 中位數 | scale < 0.5 的比例 | \\|rotation\\| 中位數 |",
        "|---|---:|---:|---:|",
    ]
    for name, item in objects.items():
        transform = item["transform"]
        lines.append(
            f"| {name} | {transform['scale_median']:.3f} "
            f"| {transform['scale_below_half_fraction'] * 100:.1f}% "
            f"| {transform['abs_rotation_median']:.1f}° |"
        )
    lines += [
        "",
        "## 落點的局部外觀（mask 外的環狀區域）",
        "",
        (
            f"兩邊都量 mask **外**寬 {RING_WIDTH_PX} px 的環狀帶，而不是 mask 底下的像素。"
            "理由：真實 mask 底下已經有瑕疵，放置 mask 底下還是乾淨背景，"
            "量 mask 底下等於拿兩種不同的東西相比。環狀帶在兩邊都是無瑕疵的物件表面，"
            "問的是「瑕疵坐落在什麼樣的表面上」。"
        ),
        "",
        "| 物件 | 真實周圍 saturation | 合成周圍 saturation | 真實周圍 intensity | 合成周圍 intensity |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in objects.items():
        site = item["site_appearance"]

        def show(value: float | None) -> str:
            return "—" if value is None else f"{value:.1f}"

        lines.append(
            f"| {name} | {show(site['real_saturation_median'])} "
            f"| {show(site['synthetic_saturation_median'])} "
            f"| {show(site['real_intensity_median'])} "
            f"| {show(site['synthetic_intensity_median'])} |"
        )
    lines += [
        "",
        (
            "⚠️ **這組數字是描述性的，不構成判定。** 本次沒有為「周圍表面差多少才算系統性不同」"
            "預先訂任何門檻，事後訂一個等於用結果決定標準。逐物件的差值如下，"
            "只作為假說生成之用："
        ),
        "",
    ]
    for name, item in objects.items():
        site = item["site_appearance"]
        pairs = (
            ("saturation", "real_saturation_median", "synthetic_saturation_median"),
            ("intensity", "real_intensity_median", "synthetic_intensity_median"),
        )
        parts = []
        for label, real_key, synthetic_key in pairs:
            real_value, synthetic_value = site[real_key], site[synthetic_key]
            if real_value is None or synthetic_value is None:
                parts.append(f"{label} —")
            else:
                parts.append(f"{label} `{synthetic_value - real_value:+.1f}`")
        lines.append(f"- `{name}`：合成周圍 − 真實周圍 = " + "、".join(parts))

    out_of_band = findings["objects_with_majority_out_of_band_area"]
    fully_in_band = findings["objects_with_area_fully_in_band"]
    oversized = findings["objects_whose_smallest_placement_exceeds_the_real_median"]
    lines += ["", "## 由量測導出的結論", ""]
    if out_of_band:
        lines.append(
            f"- **{'、'.join(out_of_band)}** 的合成面積有過半落在真實 p5–p95 之外，"
            "放置幾何在這些物件上偏離真實分布"
        )
    if oversized:
        lines.append(
            f"- **{'、'.join(oversized)}** 連最小的合成放置都比真實瑕疵的中位數大，"
            "代表偏離不是尾端少數樣本造成的"
        )
    if fully_in_band:
        lines.append(
            f"- **{'、'.join(fully_in_band)}** 的合成面積**全部**落在真實區間內，"
            "因此面積幾何**無法**解釋這些物件的下游退步"
        )
    if not findings["area_explains_every_object"]:
        lines.append(
            "- 因此「放置幾何偏離分布」**不是所有物件共通的解釋**。"
            "任何以此為單一主因的說法都不成立"
        )
    lines.append("")
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_text_lf(temporary, text)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--object", dest="objects", action="append")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/placement_geometry.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/placement_geometry.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_paths(args.paths)
    object_names = tuple(args.objects) if args.objects else tuple(paths.objects)
    data_root = Path(paths.data_root)
    raw_root = data_root / "raw/VisA"
    blocklist = {
        str(value)
        for value in json.loads(
            (paths.splits / "test_blocklist.json").read_text(encoding="utf-8")
        )["sha256"]
    }
    measured: dict[str, Any] = {}
    for object_name in object_names:
        records = load_placements(data_root, object_name)
        guard = assert_sources_are_not_test(
            records,
            raw_root=raw_root,
            blocklist=blocklist,
        )
        measured[object_name] = {
            "blocklist": guard,
            **measure_object(
                records=records,
                raw_root=raw_root,
                placement_root=data_root / "synthetic/placements" / object_name,
            ),
        }
    payload = {
        "status": "passed",
        "schema_version": 1,
        "reads_frozen_test": False,
        "real_band_quantiles": [LOW_QUANTILE, HIGH_QUANTILE],
        "objects": measured,
        "findings": derive_findings(measured),
    }
    atomic_write(args.report, build_report(payload))
    atomic_write(
        args.output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(payload["findings"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
