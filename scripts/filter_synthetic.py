"""Run the deterministic M13 quality funnel over configured synthetic inputs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import load_paths
from src.filtering.dataset import publish_views, read_filter_inputs
from src.filtering.pipeline import load_yaml, run_filter_pipeline
from src.filtering.rules import RULE_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/filters.yaml"))
    parser.add_argument("--stage-a-config", type=Path, default=Path("configs/stage_a.yaml"))
    parser.add_argument(
        "--placement-config",
        type=Path,
        default=Path("configs/placement.yaml"),
    )
    parser.add_argument(
        "--real-mask-stats",
        type=Path,
        default=Path("reports/real_mask_stats.json"),
    )
    parser.add_argument("--object", action="append", dest="objects")
    parser.add_argument("--disable", action="append", default=[], choices=RULE_ORDER)
    parser.add_argument("--limit-per-input", type=int)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--validation-out", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    paths = load_paths(args.paths)
    config = load_yaml(args.config)
    samples = read_filter_inputs(
        paths,
        [str(value) for value in config["inputs"]],
        objects=None if args.objects is None else set(args.objects),
        limit_per_input=args.limit_per_input,
    )
    result = run_filter_pipeline(
        paths,
        samples,
        filter_config=config,
        stage_a_config=load_yaml(args.stage_a_config),
        placement_config=load_yaml(args.placement_config),
        stats_path=args.real_mask_stats,
        disabled=set(args.disable),
    )
    output = config["output"]
    published = None
    if args.publish:
        filtered_root, unfiltered_root = publish_views(
            paths,
            samples,
            result.records,
            filtered_name=str(output["filtered_name"]),
            unfiltered_name=str(output["unfiltered_name"]),
            link_mode=str(output["link_mode"]),
        )
        published = {
            "filtered_root": str(filtered_root),
            "unfiltered_root": str(unfiltered_root),
        }
    summary = {
        "schema_version": 1,
        "pipeline_version": "0.2.0",
        "disabled_rules": sorted(set(args.disable)),
        "counts": result.counts,
        "thresholds": result.thresholds,
        "model_revision": result.model_revision,
        "published": published,
    }
    serialized = json.dumps(summary, indent=2, sort_keys=True)
    print(serialized)
    if args.validation_out is not None:
        args.validation_out.parent.mkdir(parents=True, exist_ok=True)
        args.validation_out.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
