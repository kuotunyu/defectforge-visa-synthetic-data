---
name: df-sdg
description: Generate, resume, independently validate, or audit DefectForge M12 full-resolution SD2 Stage B original-bucket samples. Use when working with generate_diffusion.py, validate_diffusion.py, generate_sd2.yaml, M9 placements, SD2 LoRA inference, stageB_sd2/original, or diffusion contact sheets.
---

# DefectForge SDG Inference

Run from the DefectForge repository root. Invoke `df-guard` first. Keep full-resolution images,
masks, candidate sidecars, and metadata under `D:/sdg-data/01-defectforge/synthetic`; never
put bulk outputs in Git. Check GPU occupancy before loading weights because SafeSynth and
FormosaNLU may share the RTX 4090.

## Frozen preconditions

Require:

- manifest SHA256
  `3d3c385cf0ff78479ecf90b4faf25fc07c88830e043616fb15aefb1282983e8c`;
- few-shot selection SHA256
  `7021234d0bef51926832591d60c205fa7273e0cc32fd0ae5348740094b060ea2`;
- defect-types SHA256
  `0c7669287fb1b8f48b3f6aff202eaefc914e8bfc1d892b9e63f88b20996cb41a`;
- M9 placement JSONL SHA256:
  pcb1 `58532c6da7795bc97cb9f5dee5193a7f2f3ba208e1c350e70d63130782c141d5`,
  capsules `148f427ec3549f2f187b13afc6ec0ade8589842b0a977ca10786d391f01ea323`;
- M9 independent validation `status=passed` with zero geometry, sibling-overlap, and
  test-blocklist failures;
- M10 final adapter hashes exactly matching `configs/generate_sd2.yaml`;
- base model `sd2-community/stable-diffusion-2-inpainting` at revision
  `5f74973cbb64c8568780732c17f43eb269d63a0d`;
- original-bucket pipeline version `0.5.0` (searched uses the separately audited v0.6.0
  contract in `df-refine`).

Do not use test images, masks, hashes, embeddings, thresholds, or visual choices. The first
500 frozen placements per object are a deterministic prefix with exact quotas:
pcb1 348/152 and capsules 375/125 for type0/type1.

## Preflight and smoke

Check VRAM, then dry-run both objects:

```powershell
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
uv run --frozen python src/synthetic/generate_diffusion.py --object pcb1 --dry-run
uv run --frozen python src/synthetic/generate_diffusion.py --object capsules --dry-run
```

Use a fresh ignored output for an end-to-end smoke:

```powershell
uv run --frozen python src/synthetic/generate_diffusion.py --object pcb1 --n 2 --out-name stageB_sd2_smoke
uv run --frozen python src/synthetic/generate_diffusion.py --object capsules --n 2 --out-name stageB_sd2_smoke
uv run --frozen python scripts/validate_diffusion.py --bucket original --object pcb1 --object capsules --n 2 --out-name stageB_sd2_smoke
```

Require full-resolution image/mask pairs, one atomic `.records/<sample_id>.json` per sample,
canonical metadata rebuilt from sidecars, exact background/mask provenance, zero pixels
changed outside declared blend support, zero blocklist hits, and unique output hashes.

## Resume and byte determinism

Resume only with identical effective parameters:

```powershell
uv run --frozen python src/synthetic/generate_diffusion.py --object pcb1 --n 2 --out-name stageB_sd2_smoke --resume
```

Require `generated=0`, `skipped=2`, and `peak_vram_gib=0`; a complete resume must not load the
model. Every sidecar locks config SHA, pipeline version, model revision, placement checksum,
seed, placement index, and effective inference/blend parameters.

Generate two fresh two-sample outputs with the same seed and compare SHA256 for every image
and mask. Require zero mismatches. `created_at` timestamps are provenance and are outside the
byte-identity claim.

## Formal original bucket

Run one object at a time:

```powershell
uv run --frozen python src/synthetic/generate_diffusion.py --object pcb1 --contact-sheet reports/figures/diffusion_original_pcb1.png
uv run --frozen python src/synthetic/generate_diffusion.py --object capsules --contact-sheet reports/figures/diffusion_original_capsules.png
uv run --frozen python scripts/validate_diffusion.py --bucket original --output reports/stageB_sd2_original_validation.json
```

Require 1,000 records, images, masks, and sidecars; exact quotas; 1,000 unique image and mask
hashes; 40 inference steps; full-resolution GT masks byte-identical to M9 placements; and no
test-blocklist hit. The canonical metadata must contain only exact-schema records and no
temporary files may remain.

Open both crop-level contact sheets. Preserve bad examples such as text distortion,
semantically incompatible placements, rings, near-normal changes, or hallucinated geometry.
The original bucket is intentionally unfiltered; do not delete or hand-pick failures. M13
must make rejection reproducible.

## Closeout

Run Ruff, the full pytest suite, `git diff --check`, and the independent validator again.
Update M12 reports, PLAN, worklog, troubleshooting, interfaces, and this skill. Commit only
code, configs, tests, reports, JSON validation evidence, and canonical contact sheets.

Audit all Git authors and committers as
`kuotunyu <61350295+kuotunyu@users.noreply.github.com>`. Add no co-author trailer. Do not add
a remote, create a GitHub repository, push, or open a pull request until the user wakes.
