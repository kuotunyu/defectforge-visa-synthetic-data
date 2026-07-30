"""Generate the ADR-038 v4 pilot config deterministically from frozen artefacts.

The two arms differ only in which filtered synthetic samples they may draw from:

* `db_current`  — the whole filtered pool, as v1 and v3 used it.
* `db_inband`   — only samples whose placement area falls inside the real defect
  p5-p95 band measured by `diagnose_placement_geometry.py`.

Both arms get the same sample count, set by the object that can supply the fewest in-band
samples, so a difference cannot come from synthetic volume. Selection inside each arm is a
seed-42 shuffle, which spreads defect types and generators instead of taking a sorted prefix
that would collapse onto one type.

Restricting the area also shifts the generator mix, so the contrast attributes to the in-band
subset as a bundle, not to area alone. ADR-038 states this and so does the report.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import write_text_lf  # isort: skip
from src.common.paths import load_paths  # isort: skip

SELECTION_SEED = 42
VIEW = "filtered"
# Carried forward from ADR-026's own preregistered ranking; not re-tuned here.
REAL_BAD_SHARE = 0.75


class PilotConfigError(RuntimeError):
    """Raised when the v4 arms cannot be built from frozen artefacts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotConfigError(message)


def load_filtered(paths: Any) -> list[dict[str, Any]]:
    path = paths.synthetic / VIEW / "metadata.jsonl"
    require(path.is_file(), f"Missing filtered metadata: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(bool(records), f"Filtered metadata is empty: {path}")
    return records


def real_bands(report: Path) -> dict[str, tuple[float, float]]:
    require(report.is_file(), f"Missing placement geometry report: {report}")
    payload = json.loads(report.read_text(encoding="utf-8"))
    return {
        name: (float(item["area"]["real_band"][0]), float(item["area"]["real_band"][1]))
        for name, item in payload["objects"].items()
    }


def _shuffled(values: Sequence[str]) -> list[str]:
    ordered = sorted(values)
    generator = random.Random(SELECTION_SEED)
    generator.shuffle(ordered)
    return ordered


def build_arms(
    records: Sequence[Mapping[str, Any]],
    bands: Mapping[str, tuple[float, float]],
    object_names: Sequence[str],
) -> dict[str, Any]:
    pools: dict[str, dict[str, list[str]]] = {}
    for object_name in object_names:
        require(object_name in bands, f"No measured band for {object_name}")
        low, high = bands[object_name]
        subset = [row for row in records if row["object"] == object_name]
        require(bool(subset), f"No filtered samples for {object_name}")
        in_band = [
            str(row["sample_id"])
            for row in subset
            if low <= float(row["placement"]["mask_area_px"]) <= high
        ]
        pools[object_name] = {
            "current": [str(row["sample_id"]) for row in subset],
            "inband": in_band,
        }
    count = min(len(pool["inband"]) for pool in pools.values())
    require(count > 0, "No object has any in-band filtered sample")
    selection = {
        arm: {
            object_name: sorted(_shuffled(pool[arm])[:count])
            for object_name, pool in pools.items()
        }
        for arm in ("current", "inband")
    }
    for arm, per_object in selection.items():
        for object_name, ids in per_object.items():
            require(
                len(ids) == count,
                f"{arm}/{object_name} supplied {len(ids)} ids, expected {count}",
            )
    return {"count": count, "pools": pools, "selection": selection}


def render_base_config(
    base: Mapping[str, Any],
    arms: Mapping[str, Any],
) -> dict[str, Any]:
    """Add one group per arm, each pinned to an explicit sample-id list.

    `build_classification_group` already honours `sample_ids_by_object`, so pinning the arms
    here means the pilot runner needs no new CLI surface and interfaces.md stays unchanged.
    """
    payload = json.loads(json.dumps(base))
    template = payload["groups"]["filtered_syn"]
    for arm, per_object in arms["selection"].items():
        group = json.loads(json.dumps(template))
        group["synthetic"] = {
            "count": arms["count"],
            "view": VIEW,
            "sample_ids_by_object": per_object,
        }
        payload["groups"][f"v4_{arm}"] = group
    return payload


def render_config(
    arms: Mapping[str, Any],
    object_names: Sequence[str],
    base_config: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "preregistered",
        "purpose": (
            "Validation-only pilot. Both arms draw the same number of filtered synthetic "
            "samples; only the placement-area distribution differs. Decision rules are "
            "fixed in ADR-038 before execution."
        ),
        "base_config": base_config.as_posix(),
        "objects": list(object_names),
        "seed": 42,
        "total_steps": 100,
        "generated_by": "scripts/build_v4_pilot_config.py",
        "selection_seed": SELECTION_SEED,
        "samples_per_object": arms["count"],
        "candidates": [
            {
                "name": "real_only",
                "group": "real_only",
                "sampler_strategy": "class_balanced",
                "real_bad_share": 0.50,
            },
            *(
                {
                    "name": f"db_{arm}",
                    "group": f"v4_{arm}",
                    "sampler_strategy": "domain_balanced",
                    "real_bad_share": REAL_BAD_SHARE,
                }
                for arm in ("current", "inband")
            ),
        ],
        "selection": {
            "candidates": ["db_current", "db_inband"],
            "primary_metric": "macro_f1",
            "secondary_metric": "auroc",
        },
        "gate": {
            "per_object_macro_f1_tolerance": 0.01,
            "per_object_auroc_tolerance": 0.02,
            "mean_macro_f1_min_gain": 0.01,
            "require_both_objects_noninferior": True,
        },
        "output": {
            "run_subdirectory": "cls_v4_pilot",
            "result": "results/v4/pilot_classification.json",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument(
        "--geometry",
        type=Path,
        default=Path("reports/placement_geometry.json"),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/classifier.yaml"),
    )
    parser.add_argument(
        "--base-out",
        type=Path,
        default=Path("configs/classifier_v4_base.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/classifier_v4_pilot.yaml"),
    )
    return parser


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_text_lf(temporary, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_paths(args.paths)
    object_names = tuple(paths.objects)
    arms = build_arms(load_filtered(paths), real_bands(args.geometry), object_names)
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    require(isinstance(base, dict), f"Invalid base config: {args.base_config}")
    _write_yaml(args.base_out, render_base_config(base, arms))
    _write_yaml(args.output, render_config(arms, object_names, args.base_out))
    print(
        json.dumps(
            {
                "samples_per_object": arms["count"],
                "pool_sizes": {
                    name: {arm: len(ids) for arm, ids in pool.items()}
                    for name, pool in arms["pools"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
