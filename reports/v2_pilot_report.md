# M26 v2 domain-balanced pilot

**Status:** completed; confirmatory M27 correctly stopped

**Protocol:** `configs/classifier_v2_pilot.yaml` committed before execution

**Scope:** development validation only; the frozen test partition was not loaded

## Question

Did v1 fail because label-balanced sampling let 500 synthetic anomalies overwhelm the
10 real anomalies inside the positive class?

The v1 sampler assigned 50% probability to good and 50% to bad, but sampled uniformly
within bad. That produced only **14 real-bad exposures** versus **769 synthetic-bad
exposures** in a 1,600-sample schedule. The v2 sampler kept good at 50%, then explicitly
reserved either 50% or 75% of the bad-class budget for real anomalies.

## Validation results

| Candidate | pcb1 Macro-F1 | pcb1 AUROC | capsules Macro-F1 | capsules AUROC | Real / synthetic bad exposure |
|---|---:|---:|---:|---:|---:|
| Real-only | 0.6944 | 0.9167 | 0.8133 | 0.9120 | 818 / 0 |
| v1 class-balanced | 0.5537 | 0.3139 | 0.4545 | 0.2083 | 14 / 769 |
| Domain-balanced 50% | 0.6944 | **0.9389** | 0.6571 | 0.7500 | 405 / 383 |
| Domain-balanced 75% | 0.6944 | 0.8806 | 0.6571 | **0.8611** | 613 / 215 |

All rows use the same ConvNeXt-Tiny base, filtered 500-image synthetic set, seed 42,
100-step budget, optimizer, transform, and frozen train-side real validation split.

## Gate decision

The preregistered ranking selected **Domain-balanced 75%** because both domain-balanced
variants tied on mean Macro-F1 and 75% had higher mean AUROC.

- pcb1 vs real-only: Macro-F1 `+0.0000`, AUROC `-0.0361`
- capsules vs real-only: Macro-F1 `-0.1562`, AUROC `-0.0509`
- two-object mean Macro-F1 gain: `-0.0781`

This fails all three confirmatory conditions: per-object Macro-F1 tolerance, per-object
AUROC tolerance, and minimum mean Macro-F1 gain. Therefore
`confirmatory_run_authorized_by_gate=false`; no test evaluation, three-seed run, A100,
or Colab spend is permitted for this hypothesis.

## Interpretation

The exposure hypothesis was **partly correct**. Domain balancing completely recovered
pcb1 Macro-F1 and the 50% variant improved pcb1 AUROC over real-only. It also recovered
substantial performance on capsules relative to v1 mixing. However, capsules remained
materially below real-only, so exposure collapse is not the sole cause. The residual
domain gap is consistent with object-dependent synthetic appearance or mask-placement
quality, but this pilot was not designed to distinguish those mechanisms.

The honest result is therefore: better mixing can prevent catastrophic forgetting, but
the current synthetic set still does not justify a confirmatory claim of improvement.

## Reproduction

```powershell
uv run --frozen python scripts/run_v2_classifier_pilot.py
uv run --frozen python scripts/run_v2_classifier_pilot.py --execute
uv run --frozen python scripts/verify_v2_classifier_pilot.py
```

The runner is resumable from complete raw reports and fails on partial output. The
independent verifier recomputes the gate from all eight raw development reports, checks
that every data manifest has an empty test list and real-only validation, and binds
metrics, exposure counts, signatures, and model hashes.
