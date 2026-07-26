---
name: df-guard
description: Verify DefectForge frozen split integrity and prevent test-image or test-mask leakage. Use before M5 and every later clustering, synthesis, filtering, training, or evaluation stage; also use whenever split_manifest.json, MANIFEST.sha256, test_blocklist.json, or a stage input list is inspected or changed.
---

# DefectForge Guard

Run this preflight from the DefectForge repository root before a stage reads images.
Treat every failure as blocking; never repair or regenerate a frozen artifact automatically.

## 1. Verify the frozen partition

Run:

```powershell
uv run --frozen python -c "from pathlib import Path; from src.common.integrity import verify_frozen_manifest; m,h=verify_frozen_manifest(Path('splits')); print(h, len(m['images']))"
```

Require:

- checksum verification succeeds;
- exactly 1,806 manifest images are present for the locked pcb1/capsules scope;
- `splits/test_blocklist.json` has the same `manifest_sha256`;
- every pHash `group_id` belongs to exactly one final `set`.

Do not use `scripts/freeze_manifest.py --force` in unattended work.

## 2. Guard every intended training-side file

Build the exact `Path` list that the stage will read, then call:

```python
from src.common.integrity import assert_not_blocklisted

assert_not_blocklisted(files, paths.splits / "test_blocklist.json")
```

This check applies to images and masks. The stage implementation must repeat it internally;
the skill preflight is the first layer, not a substitute.

Evaluation is the only stage allowed to read records whose manifest `set` is `test`.
Evaluation must not pass those files to fitting, clustering, generation, threshold calibration,
model selection, or hyperparameter selection.

## 3. Check operational boundaries

- Load paths with `src.common.paths.load_paths`; do not hardcode C: or D: paths.
- Confirm the previous milestone is committed and inspect `git status --short`.
- Confirm free D: space exceeds estimated new output by 1.5x.
- Require CUDA only for GPU stages; report the active GPU and free VRAM.
- Search staged changes for `.env` content, tokens, school email, `Co-Authored-By`, or other authors.
- Keep the repository remote-free until the user creates the GitHub repository.

## 4. Failure behavior

On checksum mismatch, blocklist hit, missing input, insufficient disk/VRAM, or an assertion
failure: stop that stage, preserve the frozen files, and write the concrete path/hash/count
to the handoff report. Do not lower a threshold, skip an input, or edit the manifest to pass.
