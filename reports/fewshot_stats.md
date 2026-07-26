# M5 Few-shot Selection and Mask Statistics

- Manifest SHA256: `3d3c385cf0ff78479ecf90b4faf25fc07c88830e043616fb15aefb1282983e8c`
- Selection SHA256: `7021234d0bef51926832591d60c205fa7273e0cc32fd0ae5348740094b060ea2`
- Seed: `42`
- Few-shot k per object: `10`
- Validation: 10% per object × label from highshot train, excluding official fewshot pool

## Frozen partition counts

| object | train good | train bad | test good | test bad | val good | val bad |
|---|---:|---:|---:|---:|---:|---:|
| pcb1 | 602 | 60 | 402 | 40 | 60 | 6 |
| capsules | 361 | 60 | 241 | 40 | 36 | 6 |

## k=10 seed mask distribution

| object | metric | min | p05 | median | p95 | max |
|---|---|---:|---:|---:|---:|---:|
| pcb1 | area_px | 1661 | 1904 | 3938.5 | 55632.6 | 95936 |
| pcb1 | area_ratio | 0.00110565 | 0.00126741 | 0.00262168 | 0.0370321 | 0.0638603 |
| pcb1 | aspect_ratio | 0.847059 | 0.917172 | 1.25768 | 3.61198 | 4.78571 |
| capsules | area_px | 118 | 151.75 | 2444.5 | 23670.1 | 24121 |
| capsules | area_ratio | 7.86667e-05 | 0.000101167 | 0.00162967 | 0.0157801 | 0.0160807 |
| capsules | aspect_ratio | 0.631799 | 0.66352 | 1.03947 | 1.14792 | 1.16667 |

The raw per-mask values, bounding boxes, normalized centroids, and image sizes are in
`reports/real_mask_stats.json` for downstream sampling and audit.

## Assertions

- Selection regenerated twice with byte-identical canonical JSON: **passed**
- Few-shot seed and validation are disjoint: **passed**
- No selected image or mask hash is test-blocklisted: **passed**
- Frozen manifest checksum unchanged before/after M5: **passed**
