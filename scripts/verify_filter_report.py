"""Verify M13 published payloads and report counts from first principles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import load_paths
from src.filtering.pipeline import load_yaml
from src.filtering.reporting import (
    embedded_summary,
    read_records,
    summarize,
    summary_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/filters.yaml"))
    args = parser.parse_args()
    paths = load_paths(args.paths)
    config = load_yaml(args.config)
    output = config["output"]
    filtered_root = paths.synthetic / str(output["filtered_name"])
    unfiltered_root = paths.synthetic / str(output["unfiltered_name"])
    all_records = read_records(unfiltered_root / "metadata.jsonl")
    accepted_records = read_records(filtered_root / "metadata.jsonl")
    accepted_ids = {
        (record["filter"]["input_name"], record["sample_id"])
        for record in all_records
        if record["filter"]["passed"]
    }
    published_ids = {
        (record["filter"]["input_name"], record["sample_id"])
        for record in accepted_records
    }
    if accepted_ids != published_ids:
        raise RuntimeError("Filtered metadata differs from accepted unfiltered records")
    hardlinks_checked = 0
    for root, records in (
        (unfiltered_root, all_records),
        (filtered_root, accepted_records),
    ):
        for record in records:
            source_root = paths.synthetic / str(record["filter"]["input_name"])
            for input_field, output_field in (
                ("input_image_path", "image_path"),
                ("input_mask_path", "mask_path"),
            ):
                source = source_root / record["filter"][input_field]
                destination = root / record[output_field]
                if not destination.is_file():
                    raise RuntimeError(f"Missing published payload: {destination}")
                if not os.path.samefile(source, destination):
                    raise RuntimeError(f"Payload is not a source hardlink: {destination}")
                hardlinks_checked += 1

    observed = summarize(all_records)
    report_path = paths.project_root / str(output["report"])
    embedded = embedded_summary(report_path.read_text(encoding="utf-8"))
    if embedded != observed:
        raise RuntimeError("Markdown summary differs from published metadata")
    result = {
        "status": "passed",
        "total": len(all_records),
        "accepted": len(accepted_records),
        "rejected": len(all_records) - len(accepted_records),
        "summary_sha256": summary_sha256(observed),
        "hardlinks_checked": hardlinks_checked,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
