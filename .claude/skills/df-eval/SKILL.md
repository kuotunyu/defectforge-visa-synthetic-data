---
name: df-eval
description: Prepare, run, audit, report, visualize, or verify DefectForge M14 generation-quality evaluation. Use for mask-centered crops, DINOv2 nn_score or mnn_score, clean-fid Inception features, KID, FID, real-self or noise sanity checks, test-blocklist source audits, generation_quality.csv, or generation_quality.md.
---

# DefectForge Generation Quality Evaluation

Run from the repository root after M13 publication and verification. Read
`docs/filtering_spec.md` and `configs/quality.yaml`; do not evaluate whole images.

## Locked metric contract

- Crop around the full-resolution mask at ratio 2.5.
- Use DINOv2-base CLS embeddings at the pinned revision, L2-normalized for cosine.
- `nn_score(g)` is maximum generated-to-real cosine.
- `mnn_score` is the fraction of real crops participating in a mutual nearest-neighbor pair.
- Use clean-fid 0.1.35 clean-mode Inception features for KID and FID.
- Compute deterministic unbiased degree-3 polynomial KID for formal generated-vs-real rows.
- Use the biased degree-3 polynomial MMD for real-self and noise sanity controls. It includes
  kernel diagonals, so a finite feature set compared with itself is exactly zero; unbiased KID
  is not an identity check because its same-set U-statistic is negative for tiny samples.
- Compute exact low-rank FID; clean-fid 0.1.35 directly calls a removed SciPy `sqrtm`
  parameter, and full 2048-dimensional covariance sqrtm is unstable for three real crops.
- Report KID as primary and FID only as a statistically unreliable reference.
- Report every generator by object by defect type and an object-wide aggregate for both
  unfiltered and filtered memberships. Mark empty groups explicitly.
- Match copy-paste and diffusion `type0/type1` to the same frozen real type. Procedural
  `crack/perlin/scratch/spot` uses all real components for that object and must be labeled
  `real_scope=object_all`; never imply the taxonomies are equivalent.

## Resource-aware workflow

Before any GPU work, inspect shared VRAM. If another project is using the GPU, prepare on CPU:

```powershell
uv run python scripts/evaluate_generation_quality.py --prepare-only
```

This must hash every unique generated payload and provenance source, prove zero test-blocklist
hits, and atomically materialize the immutable crop cache. Do not load DINO or Inception while
shared VRAM is busy.

When the GPU is safely available, run the mandatory gate:

```powershell
uv run python scripts/evaluate_generation_quality.py --sanity-check
```

The runner must exit 4 before model loading when existing shared VRAM exceeds
`configs/quality.yaml`'s limit.

Require every real-self group to have NN near 1, mNN near 1, biased KID exactly near 0, and FID near 0.
Require deterministic noise NN below `tau_low` by the configured margin and noise KID/FID
above self. Exit 2 on any failure; do not publish formal metrics.

Then run and verify:

```powershell
uv run python scripts/evaluate_generation_quality.py
uv run python scripts/verify_generation_quality.py
```

The verifier must match CSV rows to the Markdown embedded summary, confirm the validation
digest and mandatory gate, re-hash all feature caches, and require both figures.

## Review and closeout

Inspect the NN distribution plot and the real-versus-median-generated grid. Keep unfavorable
metrics and empty groups; never change membership or frozen types to improve the table.
Run Ruff and full pytest, document exact sanity values and cache hashes, then use only the
`kuotunyu` noreply author/committer identity with no co-author trailer or remote.
