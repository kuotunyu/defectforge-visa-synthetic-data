---
name: df-prep-testcase
description: Prepare, resume, independently validate, or audit DefectForge M9 SDG test cases from frozen train-good backgrounds and frozen real-mask components. Use when working with mask_placement.py, validate_placements.py, placement.yaml, DINOv2/Otsu ROI intersection, placements.jsonl, placement contact sheets, or the synthetic/placements output.
---

# DefectForge Prepare Testcase

Run from the DefectForge repository root. Invoke `df-guard` before generation. Keep all
bulk masks and DINOv2 caches under paths from `configs/paths.yaml`; never put them in Git.

## Preconditions

Require:

- frozen manifest SHA256
  `3d3c385cf0ff78479ecf90b4faf25fc07c88830e043616fb15aefb1282983e8c`;
- frozen defect-types SHA256
  `0c7669287fb1b8f48b3f6aff202eaefc914e8bfc1d892b9e63f88b20996cb41a`;
- frozen few-shot selection and `reports/real_mask_stats.json` point to that manifest;
- every intended train-good background and real source image/mask clears
  `splits/test_blocklist.json`;
- M8 is committed, D: has at least 1.5x estimated output space, and the target output does
  not exist unless this is an exact-parameter resume.

Never use test images, test masks, test-derived thresholds, or visual selection from test data.
Keep DINOv2 locked to `facebook/dinov2-base` revision
`f9e44c814b77203eaa57a6bdbbd535f21ede1415`.

## Smoke test

Use a fresh ignored output name:

```powershell
uv run --frozen python src/synthetic/mask_placement.py --limit-backgrounds 8 --n-per-image 3 --out-name placements_smoke --viz-n 24
uv run --frozen python scripts/validate_placements.py --limit-backgrounds 8 --n-per-image 3 --out-name placements_smoke --report reports/placements_smoke_validation.json
```

Open both `placements_smoke_check_<object>.png` figures. Cyan is the legal ROI; red is the
placed mask. Reject PCB masks outside the board, capsule masks on cloth/background, empty or
clipped masks, and red pixels outside cyan.

Prove resume before the formal run:

```powershell
uv run --frozen python src/synthetic/mask_placement.py --limit-backgrounds 8 --n-per-image 3 --out-name placements_smoke --viz-n 24 --resume
```

Use `--resume` only when object list, seed, background limit, `n-per-image`, ROI method,
config, and output name are unchanged.

## Formal generation and validation

Run:

```powershell
uv run --frozen python src/synthetic/mask_placement.py --n-per-image 3 --out-name placements --viz-n 24
uv run --frozen python scripts/validate_placements.py --n-per-image 3 --out-name placements --report reports/placements_validation.json
```

Require exactly:

- pcb1: 602 train-good backgrounds, 1,806 placements, type quotas 1,256/550;
- capsules: 361 train-good backgrounds, 1,083 placements, type quotas 812/271;
- total: 963 backgrounds and 2,889 placements.

Require every placement to have:

- an exact frozen train-good background SHA256;
- an exact frozen source component and trigger token;
- a non-empty binary PNG at the background resolution;
- 100% containment in the recomputed `intersect` ROI;
- area ratio and aspect ratio inside frozen real-mask p05-p95 bounds;
- deterministic type schedule, placement ID, and sample-local PCG64 seed;
- no pixel overlap with either sibling variant on the same background;
- no input or output SHA256 in the frozen test blocklist.

Require zero geometry outliers, zero sibling overlaps, zero blocklist hits, unchanged frozen
checksums, exact tree inventory, and independently rebuilt ROI for all 963 backgrounds. Open
both canonical `placement_check_<object>.png` figures and inspect all 24 panels.

## Determinism and closeout

Generate two fresh small outputs with identical parameters and compare SHA256 for every PNG.
Require zero mismatches. Timestamps in JSONL are provenance and are outside the byte-identity
claim.

Run:

```powershell
uv run --frozen ruff check .
uv run --frozen pytest -q
```

Update `PLAN.md`, `docs/worklog.md`, and `docs/troubleshooting.md`. Commit only source,
configs, validation JSON, reports, canonical figures, tests, and this skill. Keep smoke and
determinism outputs on D: and ignored.

Audit all reachable Git authors and committers as
`kuotunyu <61350295+kuotunyu@users.noreply.github.com>`. Add no co-author trailer. Do not add
a remote, create a GitHub repository, push, or open a pull request until the user wakes and
explicitly handles repository creation.
