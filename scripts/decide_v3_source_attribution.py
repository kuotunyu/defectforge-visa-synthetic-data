"""Apply the ADR-035 attribution rule that was fixed before the v3 pilot ran.

The rule is transcribed from ADR-035 and must not be edited to fit an observed result.
For each object, on validation Macro-F1:

    placement penalty   P = real_only    - db_copypaste
    appearance penalty  A = db_copypaste - db_diffusion

`P > A` on both objects means placement/blending dominates; `A > P` on both means
appearance dominates; anything else, including a tie on either object, means no single
cause dominates.

Copy-paste keeps real defect pixels but its own feathering/Poisson seam is synthetic, so
`P` bundles placement together with blending. This attributes to two bundles, not to three
clean factors, and the report says so.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import write_text_lf  # isort: skip

BASELINE = "real_only"
REAL_APPEARANCE = "db_copypaste"
GENERATED_APPEARANCE = "db_diffusion"
ATTRIBUTION_METRIC = "macro_f1"


class AttributionError(RuntimeError):
    """Raised when the pilot result cannot support the pre-registered rule."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AttributionError(message)


def load_pilot(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing v3 pilot result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"Invalid v3 pilot result: {path}")
    return payload


def _metric(
    payload: Mapping[str, Any],
    *,
    candidate: str,
    object_name: str,
    metric: str = ATTRIBUTION_METRIC,
) -> float:
    candidates = payload["candidates"]
    require(candidate in candidates, f"Pilot result is missing candidate: {candidate}")
    per_object = candidates[candidate]
    require(object_name in per_object, f"Missing {candidate}/{object_name}")
    value = float(per_object[object_name]["metrics"][metric])
    require(math.isfinite(value), f"Non-finite {metric}: {candidate}/{object_name}")
    return value


def attribute(payload: Mapping[str, Any]) -> dict[str, Any]:
    objects = [str(value) for value in payload["objects"]]
    require(len(objects) >= 2, "Attribution needs at least two objects")
    per_object: dict[str, Any] = {}
    for object_name in objects:
        baseline = _metric(payload, candidate=BASELINE, object_name=object_name)
        real_appearance = _metric(
            payload,
            candidate=REAL_APPEARANCE,
            object_name=object_name,
        )
        generated = _metric(
            payload,
            candidate=GENERATED_APPEARANCE,
            object_name=object_name,
        )
        placement = baseline - real_appearance
        appearance = real_appearance - generated
        # Guard against float representation only, not a scientific threshold: two
        # differences that are the same number can print unequal (0.8-0.6 vs 0.6-0.4), and
        # without this the tie branch is unreachable and 1e-17 of noise picks a winner.
        if math.isclose(placement, appearance, rel_tol=1e-9, abs_tol=1e-12):
            dominant = "tie"
        elif placement > appearance:
            dominant = "placement"
        else:
            dominant = "appearance"
        per_object[object_name] = {
            "placement_penalty": placement,
            "appearance_penalty": appearance,
            "dominant": dominant,
        }
    dominants = {item["dominant"] for item in per_object.values()}
    if dominants == {"placement"}:
        verdict = "placement_dominates"
    elif dominants == {"appearance"}:
        verdict = "appearance_dominates"
    else:
        verdict = "object_dependent"
    return {
        "rule": (
            "P = real_only - db_copypaste; A = db_copypaste - db_diffusion; "
            "one cause dominates only when the same side wins on every object"
        ),
        "metric": ATTRIBUTION_METRIC,
        "objects": per_object,
        "verdict": verdict,
        "placement_penalty_mean": statistics.fmean(
            item["placement_penalty"] for item in per_object.values()
        ),
        "appearance_penalty_mean": statistics.fmean(
            item["appearance_penalty"] for item in per_object.values()
        ),
        "caveat": (
            "copy-paste keeps real defect pixels but blends them with a synthetic seam, "
            "so the placement penalty bundles placement with blending"
        ),
    }


def decide(payload: Mapping[str, Any]) -> dict[str, Any]:
    gate = payload.get("gate", {})
    return {
        "status": "passed",
        "schema_version": 1,
        "source": "results/v3/pilot_classification.json",
        "preregistered_in": "ADR-035",
        "attribution": attribute(payload),
        "confirmatory_run_authorized_by_gate": bool(
            gate.get("confirmatory_run_authorized_by_gate", False)
        ),
        "gate_status": str(gate.get("status", "unknown")),
    }


def build_report(payload: Mapping[str, Any]) -> str:
    attribution = payload["attribution"]
    labels = {
        "placement_dominates": "放置／縫合為主因",
        "appearance_dominates": "外觀為主因",
        "object_dependent": "依物件而異，無單一主因",
    }
    lines = [
        "# v3 來源歸因 pilot 的預註冊判定",
        "",
        (
            "本報告只執行 [ADR-035](../docs/decisions.md#adr-035) 在**執行前**寫死的規則。"
            "數字由 `scripts/decide_v3_source_attribution.py` 從 "
            "`results/v3/pilot_classification.json` 產生。"
        ),
        "",
        "## 兩個懲罰量（validation Macro-F1）",
        "",
        "- 放置懲罰 `P = real_only − db_copypaste`",
        "- 外觀懲罰 `A = db_copypaste − db_diffusion`",
        "",
        "| 物件 | P（放置＋縫合） | A（外觀） | 該物件較大者 |",
        "|---|---:|---:|---|",
    ]
    for object_name, item in attribution["objects"].items():
        lines.append(
            f"| {object_name} | {item['placement_penalty']:+.4f} "
            f"| {item['appearance_penalty']:+.4f} | {item['dominant']} |"
        )
    lines += [
        "",
        f"**判定：{labels[attribution['verdict']]}**",
        "",
        (
            f"兩物件平均：P `{attribution['placement_penalty_mean']:+.4f}`、"
            f"A `{attribution['appearance_penalty_mean']:+.4f}`。"
        ),
        "",
        "## 這個歸因的界限",
        "",
        (
            "copy-paste 的瑕疵像素是**真實**的，但它的羽化／Poisson 縫合是合成的。"
            "因此 `P` 綁著「放置」與「縫合」兩件事，**不是**純粹的放置效應。"
            "這是兩個 bundle 的歸因，不是三個乾淨因子的分解。"
        ),
        "",
        "## Confirmatory gate",
        "",
        (
            f"- gate 狀態：`{payload['gate_status']}`"
        ),
        (
            "- 是否授權 confirmatory test："
            f"**{'是' if payload['confirmatory_run_authorized_by_gate'] else '否'}**"
        ),
        "",
        (
            "歸因結論成立**不等於** gate 通過。gate 未過時一律不得讀 frozen test、"
            "不得跑 3 seeds（ADR-035）。"
        ),
        "",
    ]
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_text_lf(temporary, text)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot",
        type=Path,
        default=Path("results/v3/pilot_classification.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v3_source_attribution.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/v3_source_attribution.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = decide(load_pilot(args.pilot))
    atomic_write(args.report, build_report(payload))
    atomic_write(
        args.output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(payload["attribution"]["verdict"], ensure_ascii=False))
    print(f"confirmatory_authorized={payload['confirmatory_run_authorized_by_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
