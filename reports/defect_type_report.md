# M6 Defect-type Clustering Report

- Backbone: `facebook/dinov2-base` (`Apache-2.0`)
- Model revision: `f9e44c814b77203eaa57a6bdbbd535f21ede1415`
- Semantic representation: `last_hidden_state[:, 0, :]` (CLS, 768-D)
- Minimum component area: `32` px
- Minimum cluster size: `3`
- Feature fusion: each block standardized, then divided by sqrt(block dimension)
- Naming: stable temporary `typeN` tokens; user display-name review is non-blocking

## pcb1

- Retained components: `23`
- Tiny components filtered: `3`
- Selected k: `2`
- Fallback applied: `false`

| k | silhouette | cluster sizes | eligible |
|---:|---:|---|---|
| 2 | 0.124927 | {0: 16, 1: 7} | True |
| 3 | 0.141689 | {0: 15, 1: 7, 2: 1} | False |
| 4 | 0.141048 | {0: 5, 1: 10, 2: 7, 3: 1} | False |
| 5 | 0.161938 | {0: 5, 1: 5, 2: 5, 3: 7, 4: 1} | False |

### Frozen temporary types

- `<pcb1-type0>`: 16 components
- `<pcb1-type1>`: 7 components

## capsules

- Retained components: `12`
- Tiny components filtered: `0`
- Selected k: `2`
- Fallback applied: `false`

| k | silhouette | cluster sizes | eligible |
|---:|---:|---|---|
| 2 | 0.237409 | {0: 9, 1: 3} | True |
| 3 | 0.247153 | {0: 9, 1: 1, 2: 2} | False |
| 4 | 0.225302 | {0: 7, 1: 2, 2: 1, 3: 2} | False |
| 5 | 0.208510 | {0: 4, 1: 2, 2: 1, 3: 3, 4: 2} | False |

### Frozen temporary types

- `<capsules-type0>`: 9 components
- `<capsules-type1>`: 3 components

## Assertions

- Every input image and mask cleared the test blocklist: **passed**
- Every retained component area meets the minimum: **passed**
- Every selected cluster meets the minimum size, or single-type fallback applied: **passed**
- Frozen manifest and few-shot selection checksums unchanged: **passed**
