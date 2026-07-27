# M11 SDXL Inpainting LoRA Colab Report

## Outcome

The M11 formal Colab L4 run passed for both objects. The immutable SDXL inpainting model
completed 400 steps per object, produced four checkpoints and four held-out sample panels
per object,
saved the UNet plus both trainable-token adapters, and reloaded each final bundle through a
fresh `PeftModel`.

- Base model: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- Locked revision: `115134f363124c53c7d878647567d04daf26e41e`
- Pipeline version: `0.3.0`
- Resolution: 1024
- UNet LoRA: rank 4, alpha 4
- Text conditioning: two CLIP encoders and two PEFT `TRAINABLE_TOKENS` adapters
- Colab runtime: L4, 24 GB
- Downloaded result root: `results/colab/lora_sdxl`

The result archive was downloaded as
`lora_sdxl-20260727T054640Z-1-001.zip` (147,164,017 bytes, 49 members), with SHA256
`9b4ff4f06cc2e0d3fdde84c257d0b709b8c2ac81487e0bc5ca8a89dec9da8660`.

## Formal results

| object | source images | components | steps | training time | wall time | notebook wrapper | steps/s | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pcb1 | 10 | 23 | 400 | 826.28 s | 910.91 s | 922.2 s | 0.4841 | 9.680 GiB |
| capsules | 10 | 12 | 400 | 830.61 s | 846.21 s | 855.8 s | 0.4816 | 9.680 GiB |

Formal training time was 1,656.89 seconds (27 min 36.89 s); the two notebook wrapper calls
totalled 1,778.0 seconds (29 min 38.0 s). Colab compute-unit consumption was not recorded
before the run and cannot be reconstructed honestly from the artifacts.

### Final adapter SHA256

| object | UNet LoRA | text encoder 1 | text encoder 2 |
|---|---|---|---|
| pcb1 | `af124d460c44341012e0034a8da02100c203ebec1284cc40bf23e6d3199f5582` | `09f52b40e3f59f416e0255433f67a84d3bbf189e3ff34946b4c72e424a4a6f42` | `2ff88cef857959dd90ee2ad769fe54804b6e76a3da54c4b547d9819f115e0c03` |
| capsules | `6a84185f531ad664f7f0ff729a1e8ad8a2a8395253b38a8c5072e48482093f75` | `2b020eb6f8866224b335075ee800a4f289892348402e3290767b90004cb0567a` | `01a7fa219885167e699aeb277e3b5a3dc17adf89c7bf01eccb93dfdef1a4494b` |

## Independent CPU import validation

The local verifier does not load the SDXL base model and therefore does not allocate GPU
memory. It independently checked:

- the downloaded ZIP CRC, member safety, byte count and SHA256;
- all 49 extracted files and the upstream validation JSON;
- model revision, pipeline version, local config SHA256 and all three frozen input hashes;
- the 803-entry test blocklist and each held-out background hash;
- both trainer states, final bundle inventories, PEFT configs and three adapter hashes;
- dual tokenizers, four PNG/JSON pairs per object, panel dimensions and hashes;
- type0/type1 rotation, prompts, placements and Colab-recorded checkpoint inventory;
- the Colab validator's fresh `PeftModel` reload result for both objects.

Machine-readable evidence is in `reports/lora_sdxl_import_validation.json`; reproduce it with:

```powershell
uv run --frozen python scripts/verify_colab_lora_results.py `
  --archive C:/Users/3Hml/Downloads/lora_sdxl-20260727T054640Z-1-001.zip
```

## Visual review

All eight 3072 × 1024 panels were opened and inspected. None is a direct copy of one
few-shot seed, but raw quality is mixed and is not represented as publication-ready:

- pcb1 type0 creates local bright connector-like or blurred structures near the circular
  component; type1 can distort silkscreen text and nearby traces;
- capsules type0 produces strong circular halos, tick marks and, at step 300, an implausible
  green orb; type1 is much weaker and sometimes close to a normal capsule;
- binary placement boundaries remain perceptible in several panels.

This negative evidence is preserved. The adapters are valid experiment inputs, while later
generation must search guidance/crop parameters, blend into full-resolution backgrounds, and
filter text-like, halo, seam and semantically incompatible outputs.

## Remaining M11 gates

The formal no-checkpoint branch and final fresh reload passed, but M11 remains unchecked:

1. PLAN and `df-finetune` explicitly require a one-step **local** SDXL smoke.
2. The checkpoint-resume path was not exercised. The formal run created checkpoints
   100/200/300/400, but it began fresh and therefore does not prove restoration of both text
   adapters plus optimizer/scheduler state.

No additional formal Colab training is needed. These two short checks should run after
FormosaNLU releases the local GPU; until then, no CUDA process should be started.
