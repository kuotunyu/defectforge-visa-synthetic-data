# Generation Report

## M7 Stage A copy-paste

- Output: `${synthetic}/stageA_copypaste`
- Validated samples: `1000`
- Quotas use largest-remainder proportional allocation and sum exactly to n.
- Every output image/mask pair passed size, non-empty, ROI containment, schema,
  and test-blocklist assertions.

| object | type | quota | poisson | feather |
|---|---|---:|---:|---:|
| pcb1 | type0 | 348 | 158 | 190 |
| pcb1 | type1 | 152 | 79 | 73 |
| capsules | type0 | 375 | 196 | 179 |
| capsules | type1 | 125 | 60 | 65 |

## M12 SD2 Stage B

### Scope and locked inputs

M12 pairs the first 500 frozen M9 placements for each object across two buckets:

- `original` pipeline v0.5.0 evaluates guidance 7.5 and crop ratio 2.5 once;
- `searched` pipeline v0.6.0 evaluates the byte-reproducible original as candidate 0,
  then three deterministic grid-covering alternatives, and retains the maximum
  `boundary-gradient-v2` score.

Both buckets use the same full-resolution source image and byte-identical M9 GT mask for
each sample ID. Generated images, masks, and atomic candidate sidecars remain on D:; Git
tracks only code, configuration, validation summaries, reports, and visual evidence.

| lock | value |
|---|---|
| base model | `sd2-community/stable-diffusion-2-inpainting` |
| immutable revision | `5f74973cbb64c8568780732c17f43eb269d63a0d` |
| generation config SHA256 | `5950f5fef7041b07c982578be88daec33849e8e16618b6b2113839bc3b480bef` |
| manifest SHA256 | `3d3c385cf0ff78479ecf90b4faf25fc07c88830e043616fb15aefb1282983e8c` |
| few-shot selection SHA256 | `7021234d0bef51926832591d60c205fa7273e0cc32fd0ae5348740094b060ea2` |
| defect types SHA256 | `0c7669287fb1b8f48b3f6aff202eaefc914e8bfc1d892b9e63f88b20996cb41a` |

The object adapters and placement JSONL files are independently hash-locked in
`configs/generate_sd2.yaml`.

### Original formal bucket

| object | records | type0 / type1 | wall time | peak VRAM |
|---|---:|---:|---:|---:|
| pcb1 | 500 | 348 / 152 | 866.33 s | 3.051 GiB |
| capsules | 500 | 375 / 125 | 859.41 s | 3.051 GiB |
| **total** | **1,000** | **723 / 277** | **1,725.75 s** | **3.051 GiB** |

Independent validation passed all 1,000 records: 1,000 unique image hashes, 1,000 unique
mask hashes, zero test-blocklist hits, 374 unique frozen source files rehashed, exact type
quotas, exact M9 mask identity, full-resolution shape, binary mask area, declared blend
support, and no temporary residue. Mean candidate score was `0.690561736667016`; mean
visible change was `0.10074130138102919`.

Visual review deliberately retained mixed results. PCB samples include plausible scratches
and local spots, but also checkerboards, silkscreen distortion, component-like rectangles,
and RGB stripes. Capsules include plausible scratches and blemishes, but also circular halos,
white label-like highlights, and radial hole borders. These are honest unfiltered inputs to
the M13 ablation, not hand-picked publishable samples.

### Refine calibration and corrective audit

The first formal searched schedule covered the full parameter grid but did not guarantee that
the original pair was among its four candidates. An audit after 111 aligned PCB samples found:

| outcome versus original score | samples |
|---|---:|
| improved | 97 |
| equal | 5 |
| regressed | 9 |

The run was stopped rather than accepting a positive average that hid per-sample regressions.
Its recoverable artifacts were moved out of the canonical tree.
[ADR-015](../docs/decisions.md#adr-015) records the correction: searched v0.6.0 fixes
candidate 0 to the original parameters and seed, then greedily covers the remaining guidance
and crop dimensions with three candidates.

The two recoverable diagnostic prefixes are
`${synthetic}/stageB_sd2/searched_aborted_prebaseline_20260727` (364 files, 213,025,842
bytes) and `${synthetic}/stageB_sd2/searched_aborted_unversioned_baseline_20260727`
(24 files, 13,990,336 bytes). Neither is part of canonical metadata or Git.

A fresh four-sample GPU smoke proved that candidate 0 evidence was byte-for-byte equal to the
formal original evidence. All four selected scores were non-decreasing, with improvements
from `+0.0348047091261758` to `+0.6608829064481`; the independent validator, unit tests,
and the v0.6.0 sidecar/record version audit all passed.

### Searched formal bucket

| object | records | wall time | peak VRAM | improved / equal / regressed | mean score | mean delta |
|---|---:|---:|---:|---:|---:|---:|
| pcb1 | 500 | 3,228.21 s | 3.051 GiB | 381 / 119 / 0 | 0.772419 | +0.067680 |
| capsules | 500 | 3,200.48 s | 3.051 GiB | 397 / 103 / 0 | 0.783759 | +0.107375 |
| **total** | **1,000** | **6,428.69 s** | **3.051 GiB** | **778 / 222 / 0** | **0.778089** | **+0.087527** |

PCB independently passed 500 original-baseline comparisons, exact type quotas, 500 unique
image and mask hashes, zero blocklist hits, and all geometric/provenance checks. Selected
guidance counts were 5.0: 198, 7.5: 119, 10.0: 102, 12.5: 81; selected crop-ratio counts
were 1.8: 93, 2.5: 170, 3.5: 237. All four candidate indices were selected in practice.

The type-balanced four-column PCB comparison shows material improvements rather than only a
score increase: obvious checkerboards, black circular/structural patches, and RGB stripes
were replaced by low-contrast scratches or local changes. Search safely retained original
outputs when no alternative scored better. Some results remain nearly invisible, and some
silkscreen/text semantics remain questionable; those are preserved for documented M13
filtering rather than removed here.

PCB visual evidence: [original](figures/diffusion_original_pcb1.png),
[searched](figures/diffusion_searched_pcb1.png), and
[aligned comparison](figures/diffusion_original_vs_searched_pcb1.png).

Capsules independently contributed 397 improved and 103 baseline selections. Selected
guidance counts were 5.0: 163, 7.5: 103, 10.0: 129, 12.5: 105; selected crop-ratio counts
were 1.8: 158, 2.5: 163, 3.5: 179. The aligned comparison shows several circular halos
replaced by thin scratches or local blemishes. It also exposes remaining high-contrast purple
spots, dark circular patches, and one stamp/text-like artifact. These failures confirm that
the refine ranker is useful but does not replace M13 semantic, text, near-copy, and seam
filters.

Capsules visual evidence: [original](figures/diffusion_original_capsules.png),
[searched](figures/diffusion_searched_capsules.png), and
[aligned comparison](figures/diffusion_original_vs_searched_capsules.png).

The combined independent validator passed all 1,000 searched records with exact quotas,
1,000 original-baseline evidence comparisons, 1,000 unique image hashes, 1,000 unique mask
hashes, 374 frozen source files rehashed, zero test-blocklist hits, and zero score regressions.
Across both objects, selected guidance counts were 5.0: 361, 7.5: 222, 10.0: 231,
12.5: 186; crop counts were 1.8: 251, 2.5: 333, 3.5: 416.

A complete formal `--resume` audit then skipped 500/500 records for each object with
`generated=0` and `peak_vram_gib=0.0`; neither run loaded the model or rewrote an output.

### Revalidation

```powershell
uv run --frozen python scripts/validate_diffusion.py `
  --bucket original `
  --output reports/stageB_sd2_original_validation.json

uv run --frozen python scripts/validate_diffusion.py `
  --bucket searched `
  --output reports/stageB_sd2_searched_validation.json
```

Both validators must finish with `"status": "passed"`. The searched report additionally
must show 1,000 original baselines compared and zero regressed score deltas.

## SDXL searched comparison

The preregistered SDXL comparison uses the Colab-trained adapters and the same monotonic
searched protocol, with 250 samples per object. The base model is
`diffusers/stable-diffusion-xl-1.0-inpainting-0.1` at revision
`115134f363124c53c7d878647567d04daf26e41e`.

### PCB formal bucket

| object | records | wall time | peak VRAM | improved / equal / regressed | mean score | mean delta |
|---|---:|---:|---:|---:|---:|---:|
| pcb1 | 250 | 11,472.76 s | 9.000 GiB | 188 / 62 / 0 | 0.727411 | +0.121870 |
| capsules | 250 | 10,725.78 s | 9.000 GiB | 210 / 40 / 0 | 0.548296 | +0.144467 |
| **total** | **500** | **22,198.54 s** | **9.000 GiB** | **398 / 102 / 0** | **0.637854** | **+0.133169** |

The independent validator passed all 250 records, 250 unique image hashes, 250 unique mask
hashes, 104 frozen source files, and zero test-blocklist hits. The selected guidance counts
were 5.0: 68, 7.5: 62, 10.0: 65, 12.5: 55; selected crop-ratio counts were 1.8: 14,
2.5: 90, 3.5: 146. All four candidate indices were selected.

Automatic integrity and monotonic refine checks do not imply semantic realism. Human review
found a mixed bucket: some local burns, stains, scratches, and discolouration are plausible,
but multiple edits resemble beetles, insects, or inserted IC components. The formal bucket
is retained unchanged so M16/M17 can measure its actual downstream utility. See
[`sdxl_searched_visual_review.md`](sdxl_searched_visual_review.md) and
[`figures/sdxl_searched_pcb1.png`](figures/sdxl_searched_pcb1.png).

Capsules independently passed 250 records, 250 unique image hashes, 250 unique mask hashes,
104 frozen source files, and zero test-blocklist hits. Selected guidance counts were
5.0: 98, 7.5: 40, 10.0: 60, 12.5: 52; crop-ratio counts were 1.8: 87, 2.5: 64,
3.5: 99. All four candidate indices were selected.

Capsules visual review found plausible type-1 corrosion, roughness, discolouration, and
coating-loss examples. Type-0, however, repeatedly hallucinated purple jewellery, buttons,
lenses, or mechanical rings across several masks and parameter choices. This systematic
semantic failure is retained in the formal bucket and recorded in
[`figures/sdxl_searched_capsules.png`](figures/sdxl_searched_capsules.png).

The combined validator passed all 500 records, 500 unique image hashes, 500 unique mask
hashes, 208 frozen source files, 500 original-baseline comparisons, zero blocklist hits, and
zero refine-score regressions. Across both objects, selected guidance counts were 5.0: 166,
7.5: 102, 10.0: 125, 12.5: 107; crop counts were 1.8: 101, 2.5: 154, 3.5: 245.
The automatic searched objective improved 398 records and retained 102 baselines, but the
human review demonstrates that this monotonic score is not a semantic-validity metric.
