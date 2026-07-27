"""Run M14 DINO correspondence and clean-fid crop quality evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import sha256_file
from src.common.paths import load_paths
from src.evaluation.quality_data import (
    audit_sources_against_blocklist,
    crop_cache_key,
    load_defect_types,
    materialize_quality_crops,
)
from src.evaluation.quality_pipeline import (
    align_decisions,
    evaluate_groups,
    extract_clean_features,
    load_dino_embeddings,
)
from src.evaluation.quality_reporting import (
    canonical_summary,
    comparison_grid,
    plot_nn_distributions,
    write_csv,
    write_report,
    write_validation,
)
from src.filtering.dataset import read_filter_inputs
from src.filtering.pipeline import load_yaml
from src.filtering.reporting import read_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/quality.yaml"))
    parser.add_argument(
        "--filter-config",
        type=Path,
        default=Path("configs/filters.yaml"),
    )
    parser.add_argument(
        "--defect-types",
        type=Path,
        default=Path("splits/defect_types.json"),
    )
    parser.add_argument(
        "--blocklist",
        type=Path,
        default=Path("splits/test_blocklist.json"),
    )
    parser.add_argument(
        "--filter-validation",
        type=Path,
        default=Path("reports/filter_validation.json"),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--sanity-check", action="store_true")
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
    filter_config = load_yaml(args.filter_config)
    defect_types = load_defect_types(args.defect_types)
    input_names = [str(value) for value in filter_config["inputs"]]
    samples = read_filter_inputs(paths, input_names)
    decision_path = paths.synthetic / str(config["input_metadata"])
    decisions = read_records(decision_path)
    align_decisions(samples, decisions)

    source_audit = audit_sources_against_blocklist(
        paths,
        samples,
        defect_types,
        args.blocklist,
    )
    cache_key = crop_cache_key(
        metadata_sha256=sha256_file(decision_path),
        defect_types_sha256=sha256_file(args.defect_types),
        source_audit_sha256=source_audit.sha256,
        ratio=float(config["crop"]["ratio"]),
    )
    crop_root, entries = materialize_quality_crops(
        paths.cache,
        cache_key,
        paths,
        samples,
        defect_types,
        ratio=float(config["crop"]["ratio"]),
        seed=paths.seed,
        source_audit_sha256=source_audit.sha256,
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "source_audit": {
                        "paths_checked": source_audit.paths_checked,
                        "hashes_checked": source_audit.hashes_checked,
                        "blocklist_hits": source_audit.blocklist_hits,
                        "sha256": source_audit.sha256,
                    },
                    "crop_root": str(crop_root),
                    "crops": len(entries),
                },
                indent=2,
            )
        )
        return
    if not torch.cuda.is_available():
        print(json.dumps({"status": "resource_error", "reason": "CUDA unavailable"}))
        raise SystemExit(4)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    existing_vram_gib = (total_bytes - free_bytes) / 1024**3
    maximum_existing = float(config["resources"]["maximum_existing_vram_gib"])
    if existing_vram_gib > maximum_existing:
        print(
            json.dumps(
                {
                    "status": "resource_error",
                    "reason": "shared GPU is busy",
                    "existing_vram_gib": existing_vram_gib,
                    "maximum_existing_vram_gib": maximum_existing,
                },
                indent=2,
            )
        )
        raise SystemExit(4)
    clean_features, clean_cache, clean_version = extract_clean_features(
        paths,
        crop_root,
        entries,
        config=config["clean_fid"],
    )
    (
        generated_embeddings,
        reference_embeddings,
        generated_dino_cache,
        reference_dino_cache,
        model_revision,
    ) = load_dino_embeddings(
        paths,
        crop_root,
        entries,
        samples,
        config=config["dinov2"],
        crop_ratio=float(config["crop"]["ratio"]),
    )
    with args.filter_validation.open("r", encoding="utf-8") as handle:
        filter_validation = json.load(handle)
    run = evaluate_groups(
        paths,
        entries,
        samples,
        decisions,
        generated_embeddings,
        reference_embeddings,
        clean_features,
        input_names=input_names,
        views=[str(value) for value in config["views"]],
        defect_types=defect_types,
        thresholds=filter_validation["thresholds"],
        sanity_config=config["sanity"],
        source_audit=source_audit,
        model_revision=model_revision,
        clean_fid_version=clean_version,
        generated_dino_cache=generated_dino_cache,
        reference_dino_cache=reference_dino_cache,
        clean_feature_cache=clean_cache,
    )
    summary = canonical_summary(run)
    output = config["output"]
    validation_path = paths.project_root / str(output["validation"])
    write_validation(validation_path, summary)
    if not run.sanity_passed:
        print(json.dumps({"status": "failed", "sanity": run.sanity}, indent=2))
        raise SystemExit(2)
    if args.sanity_check:
        print(json.dumps({"status": "passed", "sanity": run.sanity}, indent=2))
        return

    write_csv(paths.project_root / str(output["csv"]), run.rows)
    write_report(paths.project_root / str(output["report"]), summary)
    plot_nn_distributions(
        run.nn_distributions,
        paths.project_root / str(output["nn_figure"]),
    )
    comparison_grid(
        crop_root,
        entries,
        run.representatives,
        input_names,
        paths.project_root / str(output["comparison_figure"]),
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "rows": len(run.rows),
                "sanity_checks": len(run.sanity),
                "crop_root": str(crop_root),
                "report": str(paths.project_root / str(output["report"])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
