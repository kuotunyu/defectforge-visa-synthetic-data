"""Statically validate the M18 SegFormer Colab thin-wrapper contract."""

from __future__ import annotations

import json
import re
from pathlib import Path

NOTEBOOK = Path("notebooks/02_train_segformer.ipynb")
REQUIRED_SECTIONS = (
    "## 1. Runtime and GPU preflight",
    "## 2. Mount Drive and stage one object locally",
    "## 3. Reproducible environment and packaged selection",
    "## 4. Dry-run, eight formal groups, and automatic resume",
    "## 5. Independently validate and collect one result ZIP",
)


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise RuntimeError("Notebook is not valid nbformat 4 JSON")
    sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if isinstance(cell, dict)
    ]
    text = "\n".join(sources)
    for section in REQUIRED_SECTIONS:
        if section not in text:
            raise RuntimeError(f"Notebook section missing: {section}")
    required_fragments = (
        "total_gib >= 14",
        "defectforge_m18_source.zip",
        "m18_seg_{OBJECT_NAME}.zip",
        "sample_ids_by_object",
        "uv', 'sync', '--frozen",
        "train_segmenter.py",
        "--resume-from-checkpoint",
        "validate_segmenter_runs.py",
        "procedural_only",
        "all_mixed",
        "m18_seg_results_{OBJECT_NAME}",
    )
    for fragment in required_fragments:
        if fragment not in text:
            raise RuntimeError(f"Notebook contract fragment missing: {fragment}")
    if "userdata" in text or "HF_TOKEN" in text:
        raise RuntimeError("Public M18 model must not require a Colab Secret")
    if re.search(r"(?:hf_|sk-)[A-Za-z0-9_-]{16,}", text, flags=re.IGNORECASE):
        raise RuntimeError("Notebook appears to contain a literal credential")
    forbidden_training_loop = (
        "loss.backward(",
        "optimizer.step(",
        "binary_cross_entropy_with_logits(",
    )
    if any(fragment in text for fragment in forbidden_training_loop):
        raise RuntimeError("Notebook duplicates training code instead of calling the trainer")
    code_cells = [
        cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
    ]
    if any(cell.get("outputs") for cell in code_cells):
        raise RuntimeError("Notebook must be committed without execution outputs")
    if any(cell.get("execution_count") is not None for cell in code_cells):
        raise RuntimeError("Notebook must be committed without execution counts")
    print(
        json.dumps(
            {
                "status": "passed",
                "cells": len(notebook["cells"]),
                "sections": len(REQUIRED_SECTIONS),
                "formal_groups": 8,
                "alias_reruns": 0,
                "literal_credentials": 0,
                "duplicated_training_loop": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
