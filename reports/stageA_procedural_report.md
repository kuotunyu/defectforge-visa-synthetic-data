# Stage A Procedural Generation Report

- Output: `${synthetic}/stageA_procedural`
- Stats mode: `real_stats`
- Runtime real-stats access guard: `False`
- Validated samples: `1000`

> **Zero real defect pixels.** The procedural-only group never sees a single real
> defect pixel. It does use *aggregate shape statistics* (area ratio and aspect
> ratio percentiles) computed from the 10 few-shot training masks — that is the
> entire leakage surface, and it is disclosed here.

| object | shape | count | area-ratio range | aspect-ratio range |
|---|---|---:|---:|---:|
| capsules | perlin | 125 | 0.00010200–0.00745867 | 0.67010–1.12857 |
| capsules | crack | 125 | 0.00010333–0.00331933 | 0.66667–1.14151 |
| capsules | scratch | 125 | 0.00011467–0.00761600 | 0.66667–1.14286 |
| capsules | spot | 125 | 0.00010200–0.01145467 | 0.66667–1.13978 |
| capsules | bounds | — | 0.00010117–0.01578007 | 0.66352–1.14792 |
| pcb1 | perlin | 125 | 0.00127406–0.03676212 | 0.93976–3.59406 |
| pcb1 | crack | 125 | 0.00132199–0.03323881 | 0.93243–3.56667 |
| pcb1 | scratch | 125 | 0.00130801–0.03693785 | 0.94382–3.56471 |
| pcb1 | spot | 125 | 0.00127739–0.03597465 | 0.92118–3.56923 |
| pcb1 | bounds | — | 0.00126741–0.03703214 | 0.91717–3.61198 |
