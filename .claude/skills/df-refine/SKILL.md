---
name: df-refine
description: Search, resume, independently validate, visually review, or audit DefectForge M12 guidance-scale and crop-ratio refinement into stageB_sd2/searched. Use when working with --refine, candidate sidecars, boundary-gradient scores, original-versus-searched contact sheets, or refine search settings in generate_sd2.yaml.
---

# DefectForge Refine Search

Run from the DefectForge repository root after `df-sdg` preflight. Refine uses the same first
500 frozen M9 placements per object as the original bucket. It changes only inference
parameters and generator seed; it never changes the full-resolution GT mask or selects data
using the test split.

## Locked search contract

Use `configs/generate_sd2.yaml` and require:

- four candidates per sample;
- guidance grid `{5.0, 7.5, 10.0, 12.5}`;
- crop-ratio grid `{1.8, 2.5, 3.5}`;
- searched pipeline version `0.6.0` (`original` remains the frozen `0.5.0` bucket);
- candidate 0 is the byte-reproducible original baseline at guidance 7.5, crop ratio 2.5,
  and candidate index 0's seed, so refinement cannot score below the original;
- a deterministic stratified schedule that covers all four guidance values and all three
  crop ratios across the baseline plus three exploratory runs;
- `boundary-gradient-v2` scoring;
- weights: visible change 0.25, seam continuity 0.65, clipped-pixel artifact 0.10;
- 2px mask dilation plus 3px Gaussian feathering;
- one sample-local `torch.Generator` seed per candidate.

The selected candidate must be the maximum recorded score. This heuristic only ranks the
four candidates; it is not the final M13 quality filter and must not be described as one.

## Smoke and score calibration

Use a fresh ignored output:

```powershell
uv run --frozen python src/synthetic/generate_diffusion.py --object pcb1 --n 4 --out-name stageB_sd2_refine_smoke
uv run --frozen python src/synthetic/generate_diffusion.py --object capsules --n 4 --out-name stageB_sd2_refine_smoke
uv run --frozen python src/synthetic/generate_diffusion.py --object pcb1 --n 4 --refine --out-name stageB_sd2_refine_smoke --contact-sheet reports/figures/diffusion_refine_smoke_pcb1.png
uv run --frozen python src/synthetic/generate_diffusion.py --object capsules --n 4 --refine --out-name stageB_sd2_refine_smoke --contact-sheet reports/figures/diffusion_refine_smoke_capsules.png
uv run --frozen python scripts/validate_diffusion.py --bucket searched --object pcb1 --object capsules --n 4 --out-name stageB_sd2_refine_smoke
```

Inspect crop-level clean/mask/generated panels. Reject a scoring version if it consistently
prefers checkerboards, circular halos, text/logo artifacts, or invisible edits over a
plausible alternative. Record the observed candidate metrics before changing weights. Any
weight, grid, blend, or schedule change requires a pipeline-version bump and a fresh output;
never resume artifacts made under another rule.

## Formal searched bucket

Run sequentially:

```powershell
uv run --frozen python src/synthetic/generate_diffusion.py --object pcb1 --refine --contact-sheet reports/figures/diffusion_searched_pcb1.png
uv run --frozen python src/synthetic/generate_diffusion.py --object capsules --refine --contact-sheet reports/figures/diffusion_searched_capsules.png
uv run --frozen python scripts/validate_diffusion.py --bucket searched --output reports/stageB_sd2_searched_validation.json
uv run --frozen python scripts/compare_diffusion.py --object pcb1 --output reports/figures/diffusion_original_vs_searched_pcb1.png
uv run --frozen python scripts/compare_diffusion.py --object capsules --output reports/figures/diffusion_original_vs_searched_capsules.png
```

Require 1,000 selected outputs backed by 4,000 candidate evaluations, exactly four unique
parameter pairs in every sidecar, candidate 0 equality with the original evidence,
selected-score equality with the candidate maximum and no score below candidate 0,
full-resolution mask identity, exact metadata inventory, no changed pixels outside blend
support, unique output hashes, zero test-blocklist hits, and no `.tmp` residue.

## Original-versus-searched review

View both formal contact sheets for each object at crop scale. Record:

- where search removes hard seams, checkerboards, or halos;
- where it merely weakens the change;
- where semantic placement still distorts PCB silkscreen or creates capsule text/highlights;
- per-type guidance/crop selection counts and score distributions.

Do not delete visually poor searched samples. M13 must reject them with documented rules, and
the original/searched buckets must remain paired by identical sample ID and GT mask for the
ablation. If searched is not materially better in aggregate, keep the evidence and revise the
score in a new pipeline version rather than rewriting history.

## Closeout

Re-run independent validation, Ruff, pytest, and Git attribution audits. Track only code,
config, tests, reports, validation JSON, contact sheets, and this skill. Keep all generated
images, masks, and candidate sidecars on D:.

All commits must use
`kuotunyu <61350295+kuotunyu@users.noreply.github.com>` as both author and committer, with no
co-author trailer. Do not create a remote or push before the user wakes.
