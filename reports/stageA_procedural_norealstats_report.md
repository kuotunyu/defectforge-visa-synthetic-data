# Stage A Procedural Generation Report

- Output: `${synthetic}/stageA_procedural_norealstats`
- Stats mode: `no_real_stats`
- Runtime real-stats access guard: `True`
- Validated samples: `1000`

- This control uses only the fixed hand-tuned bounds in `configs/stage_a.yaml`.
- A Python audit hook made any access to `real_mask_stats.json` fatal.

| object | shape | count | area-ratio range | aspect-ratio range |
|---|---|---:|---:|---:|
| capsules | perlin | 125 | 0.00010267–0.00976267 | 0.60215–1.23913 |
| capsules | crack | 125 | 0.00010200–0.00462133 | 0.60227–1.24468 |
| capsules | scratch | 125 | 0.00010667–0.00605133 | 0.60140–1.23333 |
| capsules | spot | 125 | 0.00010333–0.01109133 | 0.60000–1.25000 |
| capsules | bounds | — | 0.00010000–0.01200000 | 0.60000–1.25000 |
| pcb1 | perlin | 125 | 0.00102511–0.02998775 | 0.76829–3.97674 |
| pcb1 | crack | 125 | 0.00102178–0.02972615 | 0.76552–3.93506 |
| pcb1 | scratch | 125 | 0.00104108–0.02943260 | 0.78333–3.94521 |
| pcb1 | spot | 125 | 0.00102178–0.02915968 | 0.75743–3.94545 |
| pcb1 | bounds | — | 0.00100000–0.03000000 | 0.75000–4.00000 |
