# Third-party Notices

本文件說明 DefectForge 使用或參考的 Dataset、Model Weights 與研究成果。
Repository 根目錄的 [MIT License](LICENSE) **僅涵蓋 DefectForge Source Code**；
下列資產保留各自的授權條款。

| 資產 | License | 使用與揭露方式 |
|---|---|---|
| [VisA Dataset](https://registry.opendata.aws/visa/) | CC BY 4.0 | 原始影像不收錄於 GitHub 或公開 Synthetic Dataset；使用時必須保留 VisA 與論文 attribution |
| [`sd2-community/stable-diffusion-2-inpainting`](https://huggingface.co/sd2-community/stable-diffusion-2-inpainting) | CreativeML Open RAIL++-M | 遵守模型用途限制並揭露 preservation mirror |
| [`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) | CreativeML Open RAIL++-M | 遵守模型用途限制 |
| [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base) | Apache-2.0 | 保留模型與 DINOv2 attribution |
| DefectForge Synthetic Images | CC BY 4.0 | 視為 VisA 衍生內容；保留 VisA attribution，並揭露對應 Diffusion base model License |
| DefectForge LoRA Weights | CreativeML Open RAIL++-M | 繼承對應 base model 的限制 |
| DefectForge Source Code | MIT | 不包含 Dataset、Synthetic Images 或 Model Weights |

完整的授權證據鏈、Pinned Revision 與驗證狀態記錄於
[docs/license_chain.md](docs/license_chain.md) 與
[`reports/model_license_verification.json`](reports/model_license_verification.json)。

## 主要研究引用

- Zou et al., *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection
  and Segmentation*（ECCV 2022）。[arXiv:2207.14315](https://arxiv.org/abs/2207.14315)
- Hu et al., *AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model*
  （AAAI 2024）。[arXiv:2312.05767](https://arxiv.org/abs/2312.05767)
- Zavrtanik et al., *DRAEM*（ICCV 2021）。
- Ghiasi et al., *Simple Copy-Paste is a Strong Data Augmentation Method*（CVPR 2021）。
- NVIDIA GTC Taiwan 2026, *Few-shot Industrial Synthetic Data Gen with NVIDIA Defect
  Image Generation Agent*。DefectForge 是獨立 Open-source replication，與 NVIDIA
  無隸屬或背書關係。
