---
title: DefectForge VisA Demo
emoji: 🏭
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: other
short_description: Verified few-shot industrial defect classification and segmentation.
---

# DefectForge · VisA Inspection Console

Public research demo for
[`kuotunyu/01-defectforge-visa`](https://github.com/kuotunyu/01-defectforge-visa).
Upload a `pcb1` or `capsules` inspection image to run the formally selected
ConvNeXt-Tiny classifier and SegFormer-B0 segmenter.

## Evidence boundary

- Each packaged SafeTensors file is checked against a generated SHA256 manifest at load time.
- Selection is post-evaluation and changes none of the published metrics.
- The formal segmentation threshold is `0.50`. The UI slider affects visualization only.
- Uploaded images are processed in volatile application memory; this app code does not save them.
- The v1 study reported negative mean gains from filtered synthetic data and does not hide them.

## License boundary

- Space application code: MIT.
- ConvNeXt base model: Apache-2.0.
- SegFormer base and derived segmenter checkpoints: NVIDIA Source Code License for SegFormer,
  **non-commercial research/evaluation use only**.
- VisA source dataset: CC BY 4.0. No original VisA image is included in this Space.

See `THIRD_PARTY_NOTICES.md` and `licenses/` for the complete notices.

This demo is not a production AOI, quality-control, medical, or safety system.
