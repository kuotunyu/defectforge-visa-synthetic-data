---
name: df-split
description: Prepare, resume, or audit DefectForge VisA M3-M5 data splits, pHash manifest freezing, test blocklist creation, and deterministic few-shot/validation selection. Use when working with prepare_splits.py, freeze_manifest.py, sample_fewshot.py, split_manifest.json, fewshot_selection.json, or the M3-M5 reports and contact sheets.
---

# DefectForge Split

Use only the project scripts below. All paths come from `configs/paths.yaml`; raw/prepared
images stay under `data_root` and never enter Git.

## Decide where to resume

1. Run `git status --short` and inspect `PLAN.md`.
2. If prepared split folders are missing, run M3.
3. If `splits/MANIFEST.sha256` exists, treat M4 as frozen: invoke `df-guard`; never rerun
   `freeze_manifest.py` and never use `--force`.
4. If the manifest is valid but selection is missing, run M5.

## M3: official split preparation

```powershell
uv run --frozen python scripts/prepare_splits.py --split-type both
```

Require all eight locked counts, highshot train/test disjointness,
`fewshot_train` contained in `highshot_train`, and exact bad-image/mask stem pairing. The wrapper must call
the official spot-diff utility; do not recreate its split algorithm.

## M4: one-time freeze

Only when no frozen artifacts exist:

```powershell
uv run --frozen python scripts/freeze_manifest.py --dry-run --phash-threshold 6
uv run --frozen python scripts/freeze_manifest.py --phash-threshold 6
```

Review dry-run nearest-distance metrics before the second command. The locked run retained
threshold 6: pcb1 median nearest distance 20 with one qualifying pair; capsules median 92
with no qualifying pair. Final frozen manifest SHA256 is
`3d3c385cf0ff78479ecf90b4faf25fc07c88830e043616fb15aefb1282983e8c`.

Require 1,806 images, 1,805 groups, no cross-partition group, 723 test images, 80 test masks,
and 803 unique blocked hashes.

## M5: deterministic sidecar

Run `df-guard`, then:

```powershell
uv run --frozen python scripts/sample_fewshot.py --seed 42 --k 10
```

This writes `fewshot_selection.json`; it must not update the manifest. Expected selection
SHA256 is `7021234d0bef51926832591d60c205fa7273e0cc32fd0ae5348740094b060ea2`.
Re-run once and require the same digest. Validation counts are pcb1 60 good / 6 bad and
capsules 36 good / 6 bad.

Open both `reports/figures/fewshot_contact_sheet_*.png`. Require 10 panels per object,
visible red GT contours, non-empty masks, and contours aligned with visually plausible defects.

## Closeout

Run:

```powershell
uv run --frozen ruff check src scripts tests
uv run --frozen pytest -q
```

Verify all Git authors/committers remain
`kuotunyu <61350295+kuotunyu@users.noreply.github.com>`, add no co-author trailer, and leave
the repository without a remote until the user creates the GitHub repository.
