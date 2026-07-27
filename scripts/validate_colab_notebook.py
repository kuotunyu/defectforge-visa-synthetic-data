"""Statically validate the M11 SDXL Colab thin-wrapper contract."""

from __future__ import annotations

import json
import re
from pathlib import Path

NOTEBOOK = Path("notebooks/01_train_inpaint_lora_sdxl.ipynb")
REQUIRED_SECTIONS = (
    "## 1. Runtime and GPU preflight",
    "## 2. Mount Drive and stage data locally",
    "## 3. Secrets and reproducible environment",
    "## 4. Dry-run, training, and automatic resume",
    "## 5. Validate and collect results",
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
        "total_gib >= 20",
        "userdata.get('HF_TOKEN')",
        "uv', 'sync', '--frozen",
        "train_inpaint_lora.py",
        "--resume-from-checkpoint",
        "validate_lora_run.py",
        "m11_sdxl_inputs.zip",
        "results/lora_sdxl",
    )
    for fragment in required_fragments:
        if fragment not in text:
            raise RuntimeError(f"Notebook contract fragment missing: {fragment}")
    if re.search(r"(?:hf_|sk-)[A-Za-z0-9_-]{16,}", text, flags=re.IGNORECASE):
        raise RuntimeError("Notebook appears to contain a literal credential")
    forbidden_training_loop = ("accelerator.backward(", "noise_scheduler.add_noise(", "optimizer.step(")
    if any(fragment in text for fragment in forbidden_training_loop):
        raise RuntimeError("Notebook duplicates the training loop instead of calling the trainer")
    if any(cell.get("outputs") for cell in notebook["cells"] if cell.get("cell_type") == "code"):
        raise RuntimeError("Notebook must be committed without execution outputs")
    print(
        json.dumps(
            {
                "status": "passed",
                "cells": len(notebook["cells"]),
                "sections": len(REQUIRED_SECTIONS),
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
