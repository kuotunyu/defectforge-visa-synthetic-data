---
name: df-types
description: Resume or audit DefectForge M6 connected-component extraction, DINOv2 plus morphology clustering, temporary trigger tokens, cluster figures, and frozen defect_types.json. Use before any stage consumes defect types or when checking M6 outputs and naming.
---

# DefectForge Types

Run `df-guard` first. M6 may read only the 20 records frozen in
`splits/fewshot_selection.json`; all image and mask hashes must clear the test blocklist.

## Reproduce M6 only before freeze

If `splits/defect_types.json` does not exist:

```powershell
uv run --frozen python scripts/cluster_defect_types.py --seed 42 --k-range 1 5 --min-cluster-size 3 --min-component-area 32 --auto-name
```

The script uses `facebook/dinov2-base` revision
`f9e44c814b77203eaa57a6bdbbd535f21ede1415`, CLS embeddings from
`last_hidden_state[:, 0, :]`, and eight morphology features. Each standardized block is divided
by the square root of its dimension before concatenation.

Never overwrite an existing `defect_types.json`. Verify it against
`splits/DEFECT_TYPES.sha256`.

## Expected frozen result

- Defect types SHA256:
  `0c7669287fb1b8f48b3f6aff202eaefc914e8bfc1d892b9e63f88b20996cb41a`.
- pcb1: 23 retained components, 3 tiny regions filtered; k=2 with sizes 16 and 7.
- capsules: 12 retained components, none filtered; k=2 with sizes 9 and 3.
- Every cluster has at least 3 components.
- Temporary tokens are `<pcb1-type0>`, `<pcb1-type1>`, `<capsules-type0>`, and
  `<capsules-type1>`.

`confirmed_by_user` remains false. A later review may change only `type_name`; never change
the trigger-token strings after training begins.

## Visual audit

Open both `reports/figures/defect_type_cluster_*_2.png`.

- Red contours must align with non-empty GT components.
- pcb1 type0 is dominated by pins/larger structural regions; type1 by smaller local regions.
- capsules type0 is dominated by local spots/dents; type1 by larger damage.
- Blurry small crops are expected for tiny real defects; empty background crops are not.

## Closeout

Run Ruff and pytest, then verify manifest and selection checksums are unchanged. Keep all Git
author and committer identities as
`kuotunyu <61350295+kuotunyu@users.noreply.github.com>` with no co-author trailer.
