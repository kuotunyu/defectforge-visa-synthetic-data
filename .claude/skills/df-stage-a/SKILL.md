---
name: df-stage-a
description: Generate, resume, independently validate, or audit DefectForge M7-M8 Stage A copy-paste and procedural datasets, including the ADR-011 no-real-stats control. Use when working with copy_paste.py, procedural.py, validate_synthetic.py, validate_procedural.py, stage_a.yaml, Stage A metadata, contact sheets, or Stage A output directories.
---

# DefectForge Stage A

Run from the DefectForge repository root. Invoke `df-guard` before any generation.
All bulk outputs stay under the `synthetic` path from `configs/paths.yaml`; never put them in Git.

## Preconditions

Require:

- frozen manifest SHA256
  `3d3c385cf0ff78479ecf90b4faf25fc07c88830e043616fb15aefb1282983e8c`;
- frozen defect-types SHA256
  `0c7669287fb1b8f48b3f6aff202eaefc914e8bfc1d892b9e63f88b20996cb41a`;
- every intended train-good background and real defect component clears
  `splits/test_blocklist.json`;
- M6 is committed, D: has at least 1.5x the estimated output space, and no remote exists
  until the user creates the repository.

Never use test data for synthesis, thresholds, ROI calibration, or visual selection.

## M7 copy-paste

Run:

```powershell
uv run --frozen python src/synthetic/copy_paste.py --n 500
uv run --frozen python scripts/validate_synthetic.py --n 500 --report reports/stageA_copypaste_validation.json
```

Use `--resume` only for the same output name, seed, count, blend, and config. Require exactly
500 records per object, frozen type quotas 348/152 for pcb1 and 375/125 for capsules,
binary non-empty masks, exact image/mask pairing, 100% ROI containment, and zero blocklist hits.

Open both canonical `stageA_copypaste_grid_<object>.png` files. Reject any rectangular seam,
PCB defect outside the board, or capsule defect on cloth/background.

## M8 procedural

Run both disclosed variants:

```powershell
uv run --frozen python src/synthetic/procedural.py --n 500
uv run --frozen python src/synthetic/procedural.py --n 500 --no-real-stats
uv run --frozen python scripts/validate_procedural.py --n 500 --report reports/stageA_procedural_validation.json
uv run --frozen python scripts/validate_procedural.py --n 500 --no-real-stats --report reports/stageA_procedural_norealstats_validation.json
```

Require exactly 125 each of perlin, crack, scratch, and spot per object. The standard branch
uses only aggregate p05/p95 area-ratio and aspect-ratio statistics from the 10 training masks.
The no-real-stats branch installs a Python audit hook: any attempt to open
`reports/real_mask_stats.json` is fatal. Do not remove, bypass, or weaken this guard.

Require area/aspect outlier rates below 10% (the locked run achieved 0%), binary non-empty
masks, exact inventory, frozen train-good background provenance, null real-defect source
fields, 100% ROI containment, unique output hashes, and zero blocklist hits.

Open all four canonical `stageA_procedural*_grid_<object>.png` files. Confirm all four shapes
appear, PCB masks stay on the board, capsule masks stay on capsules, and no rectangular seam
is visible.

## Reproducibility and closeout

For any generator change, run two independent small outputs with identical parameters and
compare SHA256 for every image and mask; require zero mismatches. Timestamps in metadata are
provenance and are outside the byte-identity claim.

Run:

```powershell
uv run --frozen ruff check .
uv run --frozen pytest -q
```

Update `PLAN.md`, `docs/worklog.md`, and `docs/troubleshooting.md`. Commit only source,
configs, reports, JSON validation evidence, and canonical grids. Keep smoke data on D: and
ignored. Audit all authors and committers as
`kuotunyu <61350295+kuotunyu@users.noreply.github.com>`, add no co-author trailer, and do not
add a remote or push.
