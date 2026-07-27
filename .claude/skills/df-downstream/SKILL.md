---
name: df-downstream
description: Expand, preflight, train, resume, audit, or aggregate DefectForge Phase 2 downstream experiments. Use for M16 ConvNeXt classification groups, classifier hyperparameter freezing, classification.csv, M18 SegFormer Colab groups, segmentation.csv, M20 aggregation, synthetic-volume/source/base/refine ablations, or any downstream test-leakage check.
---

# DefectForge Downstream Experiments

Run from the repository root. Read `docs/experiment_protocol.md` and the relevant section of
`docs/interfaces.md` before executing a run. Never select settings from the frozen test set.

## Mandatory preflight

1. Run `df-guard`; require the frozen manifest checksum, 1,806 images, 1,805 single-partition
   pHash groups, 803 unique blocklist hashes, sufficient disk, and sufficient free VRAM.
2. Require M15 `reports/phase1_acceptance.json` status `passed`.
3. Load all paths through `configs/paths.yaml`; do not hardcode local paths in source or outputs.
4. Expand the exact group before loading a model. The expansion must:
   - use only high-shot `set=train` records for fitting;
   - use only frozen real validation during development;
   - omit test entirely during development;
   - use exactly high-shot `set=test` during formal evaluation;
   - hash every selected file and prove zero train/test overlap;
   - preserve synthetic provenance back to train-side manifest records.
5. Stop on a missing source such as `stageB_sdxl/searched`; do not silently substitute another group.

## M16 classification

Use the locked `configs/classifier.yaml` model repository, revision, weight SHA256, 384 input,
balanced sampler, and fixed optimizer-step budget.

Run the two-object real-model smoke before tuning:

```powershell
uv run --frozen python src/training/train_classifier.py --object pcb1 --group real_only --mode development --smoke --run-name m16_smoke_pcb1_v1
uv run --frozen python src/training/train_classifier.py --object capsules --group real_only --mode development --smoke --run-name m16_smoke_capsules_v1
```

Require `status=passed`, one executed step, a non-empty validation metric, locked base-weight
hash, portable `data_manifest.json`, and peak VRAM below the local 24 GB budget.
Then run `uv run --frozen python scripts/verify_classifier_smoke.py`; it must pass without
loading a model or allocating GPU memory.

Tune only `real_only` in `--mode development`. Development must never create test metrics.
Freeze one common learning rate, weight decay, total steps, augmentation policy, and model lock
in `configs/classifier.yaml`; record the Real-only validation evidence before setting
`hyperparameters_frozen: true`.

Formal runs use:

```powershell
uv run --frozen python src/training/train_classifier.py --object OBJECT --group GROUP --mode final --seed SEED
```

Do not run aliases twice: `real_60` cites `full_real`, `syn_500` cites `filtered_syn`, and
`base_sd2` cites `bucket_searched`. Run all canonical groups at seed 42, then add seeds 43 and
44 only for `real_only` and the preregistered main `filtered_syn` group. Every formal run must
append one unique long-format row to `results/classification.csv`.

## Fairness and reporting

- Keep total optimizer steps fixed across group sizes.
- Use identical real images in groups 1-4.
- Record exact sampled exposures for real good, real bad, and synthetic bad records.
- Report Macro-F1, anomaly F1, AUROC, and normal false-positive rate.
- Keep unfavorable and null improvements. Never retune a synthetic group.
- Treat `filtered_syn` as the preregistered main group for three-seed reporting; do not choose
  a group after reading test scores.

## M18 and M20 boundary

Do not invent SegFormer commands before `src/training/train_segmenter.py` and its notebook exist.
At M18, extend this skill only after the local 1-step smoke passes. Require the notebook to call
the shared trainer and fill all five concrete handoff fields in `instructions_for_me.md`.
At M20, rebuild `results/segmentation.csv` from raw metrics rather than notebook output text.

## Closeout

Run Ruff, full pytest, the independent M16 verifier, Contributor audit, and secret scan. Inspect
all generated plots. Only after every PLAN verification is green: check M16, append worklog and
ADR evidence, commit as the sole `kuotunyu` author/committer without co-author trailers, and give
the user an exact action list.
