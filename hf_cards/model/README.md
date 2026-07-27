---
license: openrail++
language:
- en
- zh
library_name: diffusers
base_model:
- sd2-community/stable-diffusion-2-inpainting
- diffusers/stable-diffusion-xl-1.0-inpainting-0.1
tags:
- stable-diffusion
- stable-diffusion-xl
- lora
- inpainting
- industrial
---

# DefectForge VisA Inpainting LoRAs

Per-object SD2 and SDXL inpainting LoRA adapters trained from 10 real anomalous VisA
training images for each of `pcb1` and `capsules`.

**繁中摘要：**每個物件只使用 10 張真實瑕疵訓練圖，分別訓練 SD2／SDXL
inpainting LoRA。權重必須搭配 ROI crop、binary mask 與 blend-back 流程使用；直接對
整張 AOI 影像做一般文字生圖，不是本模型的支援用法。

## Files and base locks

```text
lora_sd2/
  pcb1/final/
  capsules/final/
lora_sdxl/
  pcb1/final/
  capsules/final/
```

Each final directory includes the UNet adapter, learned token embedding adapter(s),
tokenizer state, and final training configuration.

| Family | Base repository | Immutable revision | Resolution |
|---|---|---|---:|
| SD2 | `sd2-community/stable-diffusion-2-inpainting` | `5f74973cbb64c8568780732c17f43eb269d63a0d` | 512 |
| SDXL | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | `115134f363124c53c7d878647567d04daf26e41e` | 1024 |

## Trigger tokens

The tokens are frozen unsupervised pseudo-types, not official VisA labels.

| Object | Token | Frozen training-mask components |
|---|---|---:|
| pcb1 | `<pcb1-type0>` | 16 |
| pcb1 | `<pcb1-type1>` | 7 |
| capsules | `<capsules-type0>` | 9 |
| capsules | `<capsules-type1>` | 3 |

## Supported inference path

Use the repository pipeline so inference crops around the placed defect mask, runs the
matching base model and adapter at its native resolution, then blends the generated ROI
back into the original image. This preserves the surrounding AOI frame.

```powershell
uv run python src/synthetic/generate_diffusion.py `
  --config configs/generate_sd2.yaml `
  --object pcb1 `
  --n 1 `
  --refine
```

The equivalent Python flow is:

```python
# 1. Load a frozen normal image and a placement mask.
# 2. Expand the mask bounding box by the config crop_ratio.
# 3. Resize that ROI + mask to the base model's native resolution.
# 4. Inpaint with the object-specific LoRA and trigger token.
# 5. Resize the generated ROI back and feather/Poisson blend only inside the mask.
#
# The versioned implementation and exact metadata contract live in:
# src/synthetic/generate_diffusion.py
```

Do not bypass crop-to-ROI and blend-back by sending a full-resolution industrial frame
directly to a generic text-to-image call; that is outside this release's tested
interface.

## Training data and leakage boundary

- Each object uses 10 real anomalous images selected with seed 42.
- Selections are published in `splits/fewshot_selection.json` in the GitHub repository.
- The single test partition is VisA `2cls_highshot` test.
- No frozen test image, mask, or embedding is read during adapter training or generation.
- Every test image/mask SHA-256 is published in `splits/test_blocklist.json`.

## Limitations

- Ten training images per object create a high risk of overfitting and limited defect
  diversity.
- Trigger tokens are pseudo-types and may mix multiple visual failure mechanisms.
- SDXL `pcb1-type0` can hallucinate component- or insect-like shapes; search improves
  boundary fit but does not guarantee semantic correctness.
- Inpainting quality is sensitive to mask geometry, crop ratio, guidance scale, and
  background domain.
- These adapters cover only `pcb1` and `capsules`; they are not general industrial
  anomaly models.
- Outputs remain subject to the base models' Open RAIL++-M use restrictions.

## License chain

<!-- BEGIN VERIFIED LICENSE_CHAIN -->
| Asset | License | DefectForge obligation |
|---|---|---|
| VisA source dataset | CC BY 4.0 | Attribute VisA and its paper; do not include the original images in our HF dataset |
| `sd2-community/stable-diffusion-2-inpainting` | CreativeML Open RAIL++-M | Preserve the use-based restrictions and disclose the preservation mirror |
| `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | CreativeML Open RAIL++-M | Preserve the use-based restrictions |
| `facebook/dinov2-base` | Apache-2.0 | Attribute the model and DINOv2 paper |
| DefectForge synthetic images | CC BY 4.0 | Treat as VisA derivatives; retain VisA attribution and disclose diffusion base licenses |
| DefectForge LoRA weights | CreativeML Open RAIL++-M | Inherit the corresponding base-model restrictions and link the license |
| DefectForge source code | MIT | The MIT grant applies to code only, not data or model weights |
<!-- END VERIFIED LICENSE_CHAIN -->

## Citation and provenance

Methodology, checksums, training reports, and limitations are published in
[kuotunyu/01-defectforge-visa](https://github.com/kuotunyu/01-defectforge-visa).
The project is an independent open-source replication and is not affiliated with or
endorsed by NVIDIA.
