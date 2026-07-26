---
name: df-finetune
description: Train, smoke-test, resume, independently validate, or audit DefectForge per-object SD2 inpainting LoRA and learned trigger-token adapters. Use when working with train_inpaint_lora.py, validate_lora_run.py, lora_sd2.yaml, M10 checkpoints, held-out samples, or the runs/lora_sd2 output.
---

# DefectForge Fine-Tune

Run from the DefectForge repository root. Invoke `df-guard` before any GPU work. Keep model
weights, optimizer state, samples, and checkpoints under paths from `configs/paths.yaml`; never
put them in Git. Before starting CUDA, check available VRAM because SafeSynth and FormosaNLU
may be using the same machine.

## Frozen preconditions

Require:

- frozen manifest SHA256
  `3d3c385cf0ff78479ecf90b4faf25fc07c88830e043616fb15aefb1282983e8c`;
- frozen few-shot selection SHA256
  `7021234d0bef51926832591d60c205fa7273e0cc32fd0ae5348740094b060ea2`;
- frozen defect-types SHA256
  `0c7669287fb1b8f48b3f6aff202eaefc914e8bfc1d892b9e63f88b20996cb41a`;
- exactly 803 unique SHA256 values in `splits/test_blocklist.json`;
- M9 placements independently validated with zero geometry, overlap, or blocklist failures;
- sufficient ignored space under `D:/sdg-data/01-defectforge/runs`.

Use the preservation mirror `sd2-community/stable-diffusion-2-inpainting` at revision
`5f74973cbb64c8568780732c17f43eb269d63a0d`. Require these LFS payload hashes:

- text encoder:
  `cce6febb0b6d876ee5eb24af35e27e764eb4f9b1d0b7c026c8c3333d4cfc916c`;
- UNet:
  `9bcbb17f54b039f58bf78677fab8cd8a35dd686f6c9dd553e3646a8b0aaff41a`;
- VAE:
  `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`.

Do not substitute another revision, source model, split, placement set, trigger-token mapping,
or test-derived input. `--dry-run` performs these locks and run-signature checks without
loading weights.

## Preflight and dry-run

Check the GPU without reserving it:

```powershell
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
```

Then validate both objects:

```powershell
uv run --frozen python src/training/train_inpaint_lora.py --object pcb1 --dry-run
uv run --frozen python src/training/train_inpaint_lora.py --object capsules --dry-run
```

Stop if a frozen checksum, remote model lock, blocklist assertion, or run signature differs.

## Smoke and controlled resume

Use fresh ignored output directories so smoke evidence cannot overwrite a formal run:

```powershell
uv run --frozen python src/training/train_inpaint_lora.py --object pcb1 --smoke --output-dir D:/sdg-data/01-defectforge/runs/lora_sd2_smoke/pcb1
uv run --frozen python src/training/train_inpaint_lora.py --object capsules --smoke --output-dir D:/sdg-data/01-defectforge/runs/lora_sd2_smoke/capsules
```

Require one optimizer step, saved UNet and token adapters, a tokenizer, a sample PNG plus JSON
sidecar, and successful fresh `PeftModel` reload. The measured M10 reference is approximately
3.18 GiB peak VRAM per one-step smoke.

Prove exact resume on a fresh directory:

```powershell
uv run --frozen python src/training/train_inpaint_lora.py --object pcb1 --max-train-steps 2 --sample-every 2 --stop-after-steps 1 --output-dir D:/sdg-data/01-defectforge/runs/lora_sd2_resume_check/pcb1
uv run --frozen python src/training/train_inpaint_lora.py --object pcb1 --max-train-steps 2 --sample-every 2 --resume-from-checkpoint latest --output-dir D:/sdg-data/01-defectforge/runs/lora_sd2_resume_check/pcb1
```

Require the second process to state that it resumed at step 1 and completed step 2. Resume only
when the saved run signature exactly matches model revision, config, seed, object, frozen
inputs, trigger tokens, and effective hyperparameters. Never repair a mismatch by editing
trainer state.

## Formal local training

Run one object at a time:

```powershell
uv run --frozen python src/training/train_inpaint_lora.py --object pcb1
uv run --frozen python src/training/train_inpaint_lora.py --object capsules
```

For each object require 400 optimizer steps; checkpoints 100, 200, 300, and 400; a complete
final adapter bundle; and four 20-step held-out samples. Sample sidecars must alternate type0
and type1 and record the exact prompt, token, placement, crop, generator seed, model revision,
background SHA256, and panel SHA256. The M10 reference time is about 129 seconds of training
for pcb1 and 126 seconds for capsules, with roughly 3.20 GiB peak allocated VRAM.

Do not claim byte-identical independent CUDA training: FP16/TF32 kernels are not bitwise
deterministic. Exact resume and sample-local generator reproducibility are the applicable
claims.

## Independent validation and visual review

Run:

```powershell
uv run --frozen python scripts/validate_lora_run.py --reload --output reports/lora_sd2_validation.json
```

Require `status: passed`, exact frozen/model/config locks, both objects, all four checkpoints,
all bundle files, expected PEFT configs, adapter hashes, four PNG/JSON sample pairs per object,
type rotation, panel/background hashes, zero blocklist hits, and fresh adapter reload.

Open every sample panel. Reject any claim that the raw patch is already publishable. In M10,
some samples showed semantically incompatible PCB changes, subtle near-normal changes,
capsule text/logo artifacts, or binary-mask seams. Preserve this negative evidence. M12 must
search guidance scale and crop ratio and blend into the full-resolution background; M13 must
filter semantic failures, text artifacts, seams, and near-copies.

## Closeout

Run:

```powershell
uv run --frozen ruff check .
uv run --frozen pytest -q
git diff --check
```

Update `PLAN.md`, `reports/lora_sd2_report.md`, `docs/worklog.md`,
`docs/troubleshooting.md`, and the machine-readable validation report. Commit only code,
config, tests, reports, and this skill; keep all weights and run artifacts on D:.

Audit every reachable Git author and committer as
`kuotunyu <61350295+kuotunyu@users.noreply.github.com>`. Add no co-author trailer. Do not add
a remote, create a GitHub repository, push, or open a pull request until the user wakes and
explicitly handles repository creation.
