"""Build M13 funnel Markdown and deterministic accepted/rejected contact sheets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import load_paths
from src.filtering.pipeline import load_yaml
from src.filtering.reporting import contact_sheet, read_records, render_markdown, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/filters.yaml"))
    args = parser.parse_args()
    paths = load_paths(args.paths)
    config = load_yaml(args.config)
    output = config["output"]
    root = paths.synthetic / str(output["unfiltered_name"])
    records = read_records(root / "metadata.jsonl")
    report_path = paths.project_root / str(output["report"])
    report_path.write_text(render_markdown(summarize(records)), encoding="utf-8")
    contact_sheet(
        root,
        records,
        paths.project_root / str(output["accepted_contact_sheet"]),
        passed=True,
        count=int(output["contact_sheet_n"]),
    )
    contact_sheet(
        root,
        records,
        paths.project_root / str(output["rejected_contact_sheet"]),
        passed=False,
        count=int(output["contact_sheet_n"]),
    )
    print(report_path)


if __name__ == "__main__":
    main()
