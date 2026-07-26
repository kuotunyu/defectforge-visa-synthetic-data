# M10 SD2 Inpainting LoRA Report

## Outcome

M10 passed on the local RTX 4090. Both object-specific adapters trained for 400 steps,
produced four complete checkpoints, alternated both frozen trigger tokens in periodic
samples, and reloaded from disk through `PeftModel`.

- Base model: `sd2-community/stable-diffusion-2-inpainting`
- Locked revision: `5f74973cbb64c8568780732c17f43eb269d63a0d`
- Resolution: 512
- UNet LoRA: rank 4, alpha 4, attention projections only
- Text conditioning: two PEFT `TRAINABLE_TOKENS` rows per object
- Seed: 42
- Formal root: `D:/sdg-data/01-defectforge/runs/lora_sd2`

The original `stabilityai/stable-diffusion-2-inpainting` repository was unavailable during
M10. ADR-014 records the immutable community preservation mirror and the expected LFS SHA256
for the UNet, text encoder, and VAE.

## Formal results

| object | source images | component samples | type counts | steps | training time | wall time | peak VRAM |
|---|---:|---:|---|---:|---:|---:|---:|
| pcb1 | 10 | 23 | 16 / 7 | 400 | 128.66 s | 134.82 s | 3.203 GiB |
| capsules | 10 | 12 | 9 / 3 | 400 | 125.65 s | 131.66 s | 3.203 GiB |

Both runs were far below ADR-008's 30-minute local-training cutoff, so the Colab fallback was
not activated.

### Final adapter SHA256

| object | UNet LoRA | trainable-token adapter |
|---|---|---|
| pcb1 | `471cc2e38a4bdb7024919f46c9469b5c60c328d8756a11ae7b88b6ba9dab1770` | `68b8af46a35c4a5666ddd8f8c3f019d3f847daf5b240d04641a3e92cf3e441e0` |
| capsules | `3025b8753df9d27015a4a257b72092d4abfbadcfe14a45a5a6e6e41bde7ef64f` | `c4b3ef4c64c2ac1cdca726d85318e246393b8447f3f678329e2a1169fe79af26` |

FP16 + TF32 training is seed-controlled but not claimed to be bitwise deterministic across
independent CUDA runs. A preserved baseline followed the same sample order and nearly
identical loss trajectory but produced different adapter bytes. M12 generation, unlike
training, retains its separate byte-determinism requirement.

## Smoke, resume, and reload evidence

- One-step GPU smoke passed independently for `pcb1` and `capsules`.
- Peak smoke VRAM was 3.18 GiB.
- A controlled two-step run stopped after step 1, saved both adapters plus tokenizer,
  optimizer, scheduler, and trainer state, then resumed from `latest` at step 1 and completed
  step 2.
- The final validator independently checked all checkpoint inventories, frozen data hashes,
  sample sidecars, panel hashes, background hashes, test-blocklist exclusion, PEFT configs,
  adapter hashes, and a fresh `PeftModel.from_pretrained` reload for both objects.
- Machine-readable evidence: `reports/lora_sd2_validation.json`.

## Visual review

Every formal run contains four three-panel images: clean crop, binary placement mask, and raw
inpainting output. Sidecars record the exact prompt, trigger token, placement, crop, generator
seed, model revision, background SHA256, and panel SHA256. The token order is
type0 / type1 / type0 / type1 for both objects.

The samples are not copies of any single few-shot seed, so the PLAN M10 overfit rollback
condition was not triggered. Raw patch quality is mixed:

- `pcb1` type0 can create a component-like region when a structurally heterogeneous type is
  placed on an incompatible board location; type1 produces local trace, hole, or solder-like
  changes.
- `capsules` type0 produces local discoloration/bubble-like changes; type1 can produce
  implausible embossed or text-like artifacts.
- Binary mask seams remain visible in some raw panels.

These raw samples are diagnostic, not publishable synthetic outputs. M12 must search
guidance/crop settings and blend the selected patch back at full resolution; M13 must reject
text artifacts, semantic mismatches, seam discontinuities, and near-copies. The mixed result
is retained rather than hidden because it motivates the refine and filtering ablations.

## Revalidation

```powershell
uv run --frozen python scripts/validate_lora_run.py `
  --reload `
  --output reports/lora_sd2_validation.json
```

The validator must finish with `"status": "passed"`.
