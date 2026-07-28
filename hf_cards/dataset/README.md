---
license: cc-by-4.0
language:
- en
- zh
pretty_name: DefectForge VisA Synthetic Defects
tags:
- anomaly-detection
- image-segmentation
- synthetic-data
- industrial
- visa
---

# DefectForge VisA Synthetic Defects

Synthetic defect images and generation-time masks for the VisA `pcb1` and `capsules`
objects. Each object is generated from only 10 real anomalous training images, while the
frozen high-shot test partition is never visible to generation, filtering, or quality
reference sets.

**繁中摘要：**這是 VisA `pcb1`／`capsules` 的少樣本工業瑕疵合成資料。每個物件只用
10 張真實瑕疵訓練圖；mask 是生成時使用的標註，不是模型事後預測。資料同時提供
filtered 與 unfiltered 版本，並公開 provenance 與 test SHA-256 blocklist。

## What is included

```text
data/
  filtered/{images,masks}/
  unfiltered/{images,masks}/
  ablations/stageA_procedural_norealstats/{images,masks}/
  diagnostics/stageB_sd2/original/{images,masks}/
  diagnostics/stageB_sdxl/{original,searched}/{images,masks}/
splits/
  split_manifest.json
  defect_types.json
  test_blocklist.json
  *.sha256
```

Each leaf dataset also contains `metadata.jsonl`. No original VisA image is
redistributed. Source image fields are relative VisA paths plus SHA-256 values so users
can reconstruct provenance after obtaining VisA independently.

The formal downstream pool has 3,000 unfiltered samples: copy-paste, procedural, and
searched SD2, 500 per object and source. The frozen M13 filter accepts 1,770 of them.
The 1,000 no-real-statistics procedural samples and 2,000 SD2/SDXL diagnostic samples
are published separately and were not assigned a retrospective pass/fail label.

## Generation

- Stage A copy-paste places connected components from the 10 frozen few-shot masks.
- Stage A procedural synthesis samples texture and geometry under the frozen placement
  contract.
- Stage B uses SD2 or SDXL inpainting LoRA, crop-to-ROI inference, and blend-back at the
  source resolution.
- The searched bucket evaluates a preregistered guidance-scale/crop-ratio grid and keeps
  the candidate chosen by the frozen boundary/artifact score.

The complete method, immutable revisions, and our interpretation of NVIDIA Cosmos
AnomalyGen are documented in the
[GitHub repository](https://github.com/kuotunyu/defectforge-visa-synthetic-data).

## Labels and provenance

The binary mask is the exact mask used to create each sample. It is fully automatic and
requires no manual annotation. A metadata record includes:

- sample ID, object, pseudo-type, trigger token, generator, and bucket;
- relative background/source paths and SHA-256 values;
- ROI, mask box, affine placement, mask area, and area ratio;
- model/revision, adapter path, prompt, seed, inference parameters, crop, and blend mode;
- filtering decision, raw scores, and rejection reasons;
- pipeline version and creation timestamp.

See `docs/synthesis_spec.md` in the GitHub repository for the normative schema.

## Filtered and unfiltered views

The unfiltered formal view retains all 3,000 downstream-pool samples and the full
rejection evidence. The filtered formal view contains only samples that pass the
preregistered DINOv2-neighbor,
containment, morphology, seam, and pHash rules. Thresholds were frozen before downstream
results. Both views are provided so the filtering claim can be tested rather than
assumed. Ablation and diagnostic sets outside that formal pool remain explicitly
unfiltered; they are not relabelled after downstream results.

## Leakage controls

- The single evaluation partition is VisA `2cls_highshot` test.
- Generation, defect pseudo-typing, filtering references, and quality references use
  training data only.
- `splits/test_blocklist.json` contains every frozen test image and bad-mask SHA-256.
- `split_manifest.json` records pHash groups, final set, source set, and checksums.
- Downstream train/test disjointness is checked by content hash, not path strings.

## Limitations

- Only two VisA objects and 10 real anomalous seeds per object are covered.
- `type0` and `type1` are unsupervised pseudo-types inferred from DINOv2 and mask
  morphology. They are not official VisA defect labels and have not been manually
  renamed.
- Diffusion samples can contain semantically wrong component- or insect-like
  hallucinations even when their seams are acceptable. These failures are intentionally
  retained in unfiltered data.
- The default procedural generator uses aggregate area-ratio and aspect-ratio
  percentiles from the 10 training masks. A separate `norealstats` view removes even that
  aggregate prior.
- Downstream gains, including flat or negative outcomes, are reported in the GitHub
  README and are not a guarantee for other objects or production domains.

> **Zero real defect pixels.** The procedural-only group never sees a single real defect
> pixel. It does use *aggregate shape statistics* (area ratio and aspect ratio percentiles)
> computed from the 10 few-shot training masks — that is the entire leakage surface,
> and it is disclosed here.

## License chain

<!-- BEGIN VERIFIED LICENSE_CHAIN -->
| 資產 | License | DefectForge 義務 |
|---|---|---|
| VisA 原始 Dataset | CC BY 4.0 | 標示 VisA 與其論文；Hugging Face Dataset 不得包含原始影像 |
| `sd2-community/stable-diffusion-2-inpainting` | CreativeML Open RAIL++-M | 保留用途限制，並揭露 preservation mirror |
| `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | CreativeML Open RAIL++-M | 保留用途限制 |
| `facebook/dinov2-base` | Apache-2.0 | 標示模型與 DINOv2 論文 |
| DefectForge Synthetic Images | CC BY 4.0 | 視為 VisA 衍生內容；保留 VisA attribution，並揭露 Diffusion base model License |
| DefectForge LoRA Weights | CreativeML Open RAIL++-M | 繼承對應 base model 的限制，並附上 License 連結 |
| DefectForge Source Code | MIT | MIT 僅授權程式碼，不包含 Dataset 與 Model Weights |
<!-- END VERIFIED LICENSE_CHAIN -->

## Citation

- Zou et al., *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection
  and Segmentation*, ECCV 2022.
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.
- Zavrtanik et al., *DRAEM*, ICCV 2021.
- Ghiasi et al., *Simple Copy-Paste is a Strong Data Augmentation Method*, CVPR 2021.
- Wang et al., *AnomalyDiffusion*, AAAI 2024.

This project is an independent open-source replication and is not affiliated with or
endorsed by NVIDIA.
