---
name: df-setup
description: Set up and verify the DefectForge Windows/uv/CUDA environment and download, checksum, safely extract, and inventory the VisA dataset. Use for milestones M1-M2, when resuming a fresh DefectForge checkout, when CUDA or package imports fail, or when VisA data/checksum state is missing or uncertain.
---

# DefectForge Setup

Run the verified M1-M2 setup without touching GitHub, Colab, or another project.
Read `docs/environment.md` and `docs/data_protocol.md` before changing versions or paths.

## Preconditions

1. Work from the DefectForge repository root on native Windows.
2. Read `configs/paths.yaml`; do not hardcode a replacement data root.
3. Require repo-local Git identity
   `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`.
4. Require at least 8 GiB free on the drive containing `data_root`.
5. Treat an explicit unattended-work/download instruction as authorization. Otherwise report the
   1,929,840,640-byte download once before starting.
6. Do not create a remote, push, upload, use Colab, or print secrets.

## M1: Lock and verify the environment

Run:

```powershell
uv lock --python 3.12
uv sync --frozen --python 3.12
uv run --frozen ruff check src scripts tests
uv run --frozen pytest -q
uv run --frozen python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run --frozen python -c "import accelerate, cleanfid, cv2, diffusers, imagehash, peft, sklearn, timm, torchvision, transformers; from diffusers import AutoPipelineForInpainting; print('imports ok')"
```

Require:

- CPython 3.12
- `torch==2.13.0+cu130`
- CUDA available
- device exactly `NVIDIA GeForce RTX 4090`
- all imports succeed

Do not replace the explicit cu130 index with PyPI torch. Do not add the unmaintained
`noise==1.2.2`; procedural noise uses NumPy/scikit-image.

## M2: Download and verify VisA

Preview:

```powershell
uv run --frozen python scripts\download_visa.py --dry-run
```

Download or resume, then verify and extract:

```powershell
uv run --frozen python scripts\download_visa.py
```

If the exact tar already exists:

```powershell
uv run --frozen python scripts\download_visa.py --skip-download
```

The script must:

- require exactly `1,929,840,640` tar bytes;
- compute SHA256 and write `splits/source_checksums.json`;
- reject tar members escaping the extraction root;
- extract into `${visa_raw}` because the official tar has object folders at its root;
- verify pcb1 = 1,004 normal / 100 anomaly and capsules = 602 normal / 100 anomaly;
- require one mask per anomaly for both objects.

The verified 2026-07-27 tar SHA256 is
`2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`.
Treat a different digest as a source change: stop and investigate rather than overwriting the
record.

## Completion

Run `git diff --check`, scan for secrets and non-`kuotunyu` authors, then:

1. Mark only the verified milestone complete in `PLAN.md`.
2. Append the exact versions, digest, inventory, elapsed time, and any retry to
   `docs/worklog.md`.
3. Record solved failures in `docs/troubleshooting.md`.
4. Commit with author and committer both set to `kuotunyu`; never add a co-author trailer.

Stop only for an assertion failure that remains after diagnosing a fixable implementation issue,
insufficient disk, an unexpected upstream digest, a paid/external action, or a request requiring
the user's own login.
