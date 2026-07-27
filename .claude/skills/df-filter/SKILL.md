---
name: df-filter
description: Run, resume, audit, report, visually inspect, or verify DefectForge M13 six-stage synthetic quality filtering. Use for configs/filters.yaml, filtered or unfiltered membership, reject reasons, DINO thresholds, pHash, seam scores, funnel counts, hardlink publication, or filter contact sheets.
---

# DefectForge Quality Filter

Run from the repository root. Treat `stageA_copypaste`, `stageA_procedural`, and
`stageB_sd2/searched` as immutable inputs. Never alter their metadata or payloads.

## Locked contract

Apply rules in this order:

1. ROI containment;
2. real-reference mask area;
3. real-reference aspect ratio;
4. pHash distance from previously finally accepted crops;
5. DINOv2 nearest-real similarity, copy ceiling, and centroid outlier distance;
6. outer-band seam continuity.

Use the eight enum values in `src/filtering/rules.py`. A sample may have multiple reasons.
Classify it once by the first reason for funnel rejects, and separately report all triggers.

Use `facebook/dinov2-base` at revision
`f9e44c814b77203eaa57a6bdbbd535f21ede1415`. Calibrate `tau_low` from the 5th percentile
of real leave-one-out nearest cosine, keep `tau_copy=0.98`, and calibrate the outlier ceiling
from the real centroid-distance 95th percentile.

## Workflow

Run a no-publish smoke first:

```powershell
uv run python scripts/filter_synthetic.py --limit-per-input 2
```

Run the full decision pass, then publish only after its counts are plausible:

```powershell
uv run python scripts/filter_synthetic.py --validation-out reports/filter_validation.json
uv run python scripts/filter_synthetic.py --publish --validation-out reports/filter_validation.json
uv run python scripts/report_filters.py
uv run python scripts/verify_filter_report.py
```

The first full pass must not create `filtered/` or `unfiltered/`. Publication must hardlink
payloads, never copy or replace an unrelated existing file. `unfiltered/metadata.jsonl`
contains all decisions; `filtered/metadata.jsonl` contains exactly the accepted subset.

## Required review

Open both canonical contact sheets and inspect red mask contours, defect visibility, semantic
fit, and seams. Investigate any generator/type with zero accepted records using a dedicated
deterministic sheet. Do not relax a threshold merely to make coverage nonzero. Record any
calibration change and bump the relevant pipeline version.

Require the verifier to prove exact report reconstruction, known reason enums, accepted
membership equality, and source hardlink identity for every published image and mask.

## Closeout

Run Ruff and the full pytest suite. Track only code, config, tests, reports, JSON, and small
figures; keep full payloads on D:. Commit only as
`kuotunyu <61350295+kuotunyu@users.noreply.github.com>` for both author and committer.
Require zero co-author trailers, zero remotes, and a clean worktree.
