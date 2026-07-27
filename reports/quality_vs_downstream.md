# M17 generation quality vs downstream improvement

Preregistered join: the three unfiltered source-only generation rows are matched to their seed-42 M16 source-ablation classifiers. Improvement is measured against the same object's seed-42 Real-only Macro-F1.

| Object | Source | KID ↓ | NN mean ↑ | Real-only F1 | Source F1 | Δ Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| capsules | src_copypaste | -0.005203 | 0.763209 | 0.572789 | 0.485881 | -0.086908 |
| capsules | src_procedural | 0.005402 | 0.723610 | 0.572789 | 0.472295 | -0.100494 |
| capsules | src_diffusion | 0.014334 | 0.670441 | 0.572789 | 0.371365 | -0.201424 |
| pcb1 | src_copypaste | 0.100150 | 0.630112 | 0.682569 | 0.421294 | -0.261275 |
| pcb1 | src_procedural | 0.190637 | 0.710234 | 0.682569 | 0.541737 | -0.140832 |
| pcb1 | src_diffusion | 0.181205 | 0.705260 | 0.682569 | 0.211317 | -0.471252 |

Across the six preregistered points, Pearson r(KID, ΔMacro-F1) = -0.6084. This descriptive six-point statistic is not treated as a significance test.

The sign and strength are reported as observed; no source, metric, or object is removed after reading downstream results.
