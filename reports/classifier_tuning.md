# M16 Classifier Hyperparameter Tuning

**Status:** `passed`

**Selection data:** Real-only frozen validation only; test inventory was empty for every run.

| Learning rate | Mean Macro-F1 | Mean AUROC | pcb1 best step | capsules best step |
|---:|---:|---:|---:|---:|
| 1e-05 | 0.799986 | 0.939352 | 100 | 75 |
| 3e-05 | 0.755161 | 0.941667 | 50 | 50 |
| 1e-04 | 0.773602 | 0.880093 | 100 | 25 |

## Frozen common setting

- Learning rate: `1e-05`
- Weight decay: `0.05`
- Final refit steps: `100`
- Batch size: `16`
- Selection rule: mean Macro-F1, then mean AUROC, then lower learning rate
- The final step budget is the maximum best step across both objects.
- All formal groups use this exact setting; no synthetic group may be retuned.
