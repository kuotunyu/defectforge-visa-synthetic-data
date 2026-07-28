# DefectForge：VisA Synthetic Data 瑕疵生成與評估

[![持續驗證](https://github.com/kuotunyu/defectforge-visa-synthetic-data/actions/workflows/verify.yml/badge.svg)](https://github.com/kuotunyu/defectforge-visa-synthetic-data/actions/workflows/verify.yml)

> **狀態：已公開；M0–M36 全數驗證完成。**
> 下游結果表與最終結論皆由通過獨立驗證的 CSV 自動產生，不手動填寫數字。
> 驗證入口請見 `scripts/verify_readme.py` 與 GitHub Actions。

公開成果：
[Synthetic Dataset](https://huggingface.co/datasets/steven0226/defectforge-visa-synthetic) ·
[SD2／SDXL LoRA Weights](https://huggingface.co/steven0226/defectforge-visa-lora) ·
[正體中文互動 Demo](https://steven0226-defectforge-visa-demo.hf.space/) ·
[Release](https://github.com/kuotunyu/defectforge-visa-synthetic-data/releases)

**研究問題：Synthetic Data 能否改善少樣本工業瑕疵的 Classification 與 Segmentation？**

本專案在每個物件只有 **10 張真實瑕疵圖**的條件下，重建 NVIDIA GTC 2026
Cosmos AnomalyGen 的方法論，並全部改用 Open Model、Open Data 與可重現程式碼實作。
主實驗結果顯示：已篩選 Synthetic Data **沒有**優於 Real-only；本專案保留這個負面結果，
並進一步用 v2 pilot 分析 Synthetic Data 曝光比例是否造成退步。

---

## 研究問題

真實產線通常很難蒐集足夠的瑕疵影像。VisA 每個物件只有 **100 張異常影像**，
而官方 few-shot protocol 進一步縮減為 **10 張**；相較之下，正常影像容易取得。
DefectForge 要驗證的正是：Synthetic Data 能否補上這個資料不對稱。

| 項目 | 設定 |
|---|---|
| Dataset | [VisA](https://registry.opendata.aws/visa/)（CC BY 4.0）：`pcb1`、`capsules` |
| 真實瑕疵預算 | 每個物件 10 張，seed=42 |
| 下游任務 | Defect Classification＋Defect-region Segmentation |
| Synthetic Data 標註 | 全自動；放置用 Mask 就是 Segmentation Ground Truth |

### Split 很重要，而且非常容易洩漏

VisA 官方 `2cls_fewshot` 與 `2cls_highshot` CSV 是同一批影像的兩種不同切分。
在開始撰寫訓練程式前，我們先量測兩者的交集：

```text
highshot TRAIN(anomaly) ∩ fewshot TEST(anomaly) = 40   （每個物件，test 共 80 張）
fewshot TRAIN ⊂ highshot TRAIN                          True
highshot TEST ⊂ fewshot TEST                            True
```

如果用 highshot train 訓練 Full-real upper bound，卻用 fewshot test 評估，
就會把**一半的 test 瑕疵放進 training**。因此本專案統一以 `2cls_highshot`
作為唯一 base partition，所有實驗共用同一個 frozen test set。
因為 partition 具有包含關係，我們也能免費得到 **10 → 20 → 60**
張真實瑕疵圖的 scaling curve。完整決策見 [ADR-007](docs/decisions.md#adr-007)。

## 方法與系統架構

![DefectForge Synthetic Data 系統架構](docs/diagrams/readme_01_flowchart_system_architecture.png)

[查看 Mermaid 原始碼](docs/diagrams/readme_01_flowchart_system_architecture.mmd)

整體流程分成五層：

1. **資料與防洩漏**：凍結 split manifest、SHA256、pHash group 與 test blocklist。
2. **Synthetic Data 生成**：Stage A 使用 Copy-paste／Procedural；Stage B 使用
   SD2／SDXL Inpainting LoRA 與 Auto Mask Placement。
3. **品質與可追溯性**：六道 Quality Filtering、Filtered／Unfiltered view、
   per-image metadata provenance。
4. **下游評估**：以 ConvNeXt-Tiny 執行 Classification，以 SegFormer-B0 執行
   Segmentation。
5. **公開結果與證據鏈**：Validated CSV、Figures、SHA256-bound reports 與正體中文 Demo。

Cosmos AnomalyGen 各元件與 Open-source 替代方案的逐項對照，記錄在
[方法論文件](docs/methodology.md)；其中也明確標示哪些部分是本專案的工程詮釋，
而不是 NVIDIA 公開公式。

## 實驗設計

主實驗包含五組控制組；第 1–4 組使用完全相同的真實資料，Synthetic Data 僅能額外加入，
所有組別都在同一個 frozen test set 評估：

1. **Real-only**：10 張真實瑕疵圖
2. **+ Standard Augmentation**：排除一般 augmentation 就足以改善的可能
3. **+ 未篩選 Synthetic Data**
4. **+ 已篩選 Synthetic Data**：主比較組
5. **Full-real upper bound**：60 張真實瑕疵圖

其他實驗包括：

- 10／20／60 張真實瑕疵圖的 scaling curve
- `{125, 250, 500}` Synthetic Data volume sweep
- SD2 與 SDXL base model ablation
- 完全不讀取真實瑕疵 pixel 的 Segmentation 組別

Validation 與 test 僅使用真實資料。Generator、Filter 與 defect-type clusterer
不讀取 test image；split manifest 在第一張 Synthetic Image 產生前就凍結，
並公開 `splits/test_blocklist.json` 供外部檢查。完整 protocol 見
[實驗設計文件](docs/experiment_protocol.md)。

## 實驗結果

下列表格只能由 `scripts/verify_readme.py --write` 產生，而且必須先通過
Classification 與 Segmentation 的獨立 validator。所有 Figure 也從相同兩份 CSV
建立，輸入 SHA256 記錄於 `reports/phase2_figures_validation.json`。

![真實資料 Scaling Curve 與已篩選 Synthetic Data 等價量](reports/figures/real_scaling_curve.png)

### 瑕疵分類（Classification）

<!-- BEGIN VERIFIED CLASSIFICATION_MAIN -->
| 物件 | 訓練組別 | Macro-F1 | 瑕疵 F1 | AUROC | 正常樣本 FPR |
| --- | --- | --- | --- | --- | --- |
| pcb1 | Real-only（10 張） | 0.6826 | 0.4815 | 0.9086 | 0.2065 |
| pcb1 | + Standard Augmentation | 0.6826 | 0.4815 | 0.9157 | 0.2065 |
| pcb1 | + 未篩選 Synthetic Data | 0.4866 | 0.1858 | 0.5556 | 0.3134 |
| pcb1 | + 已篩選 Synthetic Data | 0.4270 | 0.0160 | 0.2229 | 0.2090 |
| pcb1 | Full-real（60 張） | 0.6826 | 0.4815 | 0.9294 | 0.2065 |
| capsules | Real-only（10 張） | 0.5728 | 0.2535 | 0.7934 | 0.0913 |
| capsules | + Standard Augmentation | 0.6031 | 0.3333 | 0.7656 | 0.1452 |
| capsules | + 未篩選 Synthetic Data | 0.3839 | 0.0331 | 0.3145 | 0.3278 |
| capsules | + 已篩選 Synthetic Data | 0.3712 | 0.0444 | 0.2844 | 0.3817 |
| capsules | Full-real（60 張） | 0.6748 | 0.4874 | 0.8583 | 0.2075 |
<!-- END VERIFIED CLASSIFICATION_MAIN -->

![五組 Classification 比較](reports/figures/main_comparison_table.png)

### 瑕疵區域分割（Segmentation）

<!-- BEGIN VERIFIED SEGMENTATION_MAIN -->
| 物件 | 訓練組別 | Dice | mIoU | Pixel AUROC | AUPRO |
| --- | --- | --- | --- | --- | --- |
| pcb1 | Real-only（10 張） | 0.3762 | 0.6156 | 0.9460 | 0.6028 |
| pcb1 | + Standard Augmentation | 0.3836 | 0.6185 | 0.9144 | 0.5740 |
| pcb1 | + 未篩選 Synthetic Data | 0.2490 | 0.5709 | 0.9324 | 0.6065 |
| pcb1 | + 已篩選 Synthetic Data | 0.0621 | 0.5156 | 0.9010 | 0.7471 |
| pcb1 | Full-real（60 張） | 0.6862 | 0.7610 | 0.9296 | 0.5999 |
| pcb1 | 僅 Procedural | 0.0316 | 0.5076 | 0.8551 | 0.5963 |
| pcb1 | 僅 Copy-paste | 0.0000 | 0.4997 | 0.9015 | 0.4386 |
| pcb1 | 僅 Diffusion | 0.0000 | 0.4997 | 0.8288 | 0.4556 |
| pcb1 | All-mixed（與已篩選 Synthetic Data 共用） | 0.0621 | 0.5156 | 0.9010 | 0.7471 |
| capsules | Real-only（10 張） | 0.5958 | 0.7119 | 0.9858 | 0.8488 |
| capsules | + Standard Augmentation | 0.0000 | 0.4996 | 0.8661 | 0.5591 |
| capsules | + 未篩選 Synthetic Data | 0.0000 | 0.4996 | 0.4919 | 0.1666 |
| capsules | + 已篩選 Synthetic Data | 0.4570 | 0.6477 | 0.9737 | 0.9137 |
| capsules | Full-real（60 張） | 0.6331 | 0.7312 | 0.9991 | 0.9591 |
| capsules | 僅 Procedural | 0.0000 | 0.4996 | 0.6127 | 0.3506 |
| capsules | 僅 Copy-paste | 0.6101 | 0.7191 | 0.9869 | 0.9440 |
| capsules | 僅 Diffusion | 0.0000 | 0.4996 | 0.5178 | 0.2556 |
| capsules | All-mixed（與已篩選 Synthetic Data 共用） | 0.4570 | 0.6477 | 0.9737 | 0.9137 |
<!-- END VERIFIED SEGMENTATION_MAIN -->

![九個 Segmentation 邏輯組別](reports/figures/segmentation_table.png)

## v2 後續實驗：退步是否來自 Sampling？

v1.0.0 凍結後，我們在完全不讀取 test set 的前提下驗證一個明確假設：
Label balancing 讓 500 張 Synthetic Anomaly 幾乎占據全部 positive-class exposure，
導致 10 張真實 anomaly 在 1,600 個 sampled positions 中只出現 14 次。

| Validation candidate | pcb1 Macro-F1 | pcb1 AUROC | capsules Macro-F1 | capsules AUROC |
|---|---:|---:|---:|---:|
| Real-only | 0.6944 | 0.9167 | 0.8133 | 0.9120 |
| v1 Class-balanced Mixing | 0.5537 | 0.3139 | 0.4545 | 0.2083 |
| Domain-balanced 50% real bad | 0.6944 | **0.9389** | 0.6571 | 0.7500 |
| Domain-balanced 75% real bad | 0.6944 | 0.8806 | 0.6571 | **0.8611** |

Domain balancing 救回 pcb1 與部分 capsules 表現，證明 exposure collapse 確實存在。
然而預先註冊的跨物件 gate 仍未通過：選出的 75% candidate 相對 Real-only
平均 Macro-F1 為 `-0.0781`，capsules 仍為 `-0.1562`。因此我們在 test evaluation
與 three-seed confirmatory run 之前正確停止。這是 **exploratory validation result**，
不取代上方 v1 結果。完整內容見 [v2 Pilot Report](reports/v2_pilot_report.md)。

## 公開互動 Demo

![DefectForge 決定性 Demo](assets/demo.gif)

[開啟正體中文 DefectForge Demo](https://steven0226-defectforge-visa-demo.hf.space/)

Demo 預設使用 CPU Basic，載入前會檢查四個 checkpoint SHA256，不保存上傳影像，
並提供五張具 attribution 的 VisA 範例。使用者可以查看 Classification Confidence、
Binary Mask、固定 0–1 色階的 Probability Heatmap 與 checkpoint provenance。
由於 SegFormer upstream License 限制，此 Demo 僅供 non-commercial
research／evaluation 使用。

上方 GIF 由 `scripts/record_demo_artifacts.py` 使用 frozen highshot test partition
中的四張固定影像產生。展示模型來自事後凍結的正式 checkpoint，會回查 raw report，
且不會改寫任何正式指標。

## 限制與誠實揭露

本專案刻意把 outcome block 放在限制段落，確保 Synthetic Data 的持平或負面結果
不會從最終敘事中被移除。

在預先註冊的 Segmentation threshold 0.5 下，四張決定性 Demo frame 的 Binary Mask
coverage 都是 0。Probability Heatmap 與 Classification Probability 仍完整保留，
且沒有重新挑選 sample 來隱藏這項限制。

<!-- BEGIN VERIFIED RESULT_OUTCOME -->
- Classification：已篩選 Synthetic Data 相對 Real-only 的平均 Macro-F1 差異為 `-0.2286`。
- Segmentation：已篩選 Synthetic Data 相對 Real-only 的平均 Dice 差異為 `-0.2264`。
- Classification 負面結果：**是——已篩選 Synthetic Data 未提升平均 Macro-F1。**
- Segmentation 負面結果：**是——已篩選 Synthetic Data 未提升平均 Dice。**
<!-- END VERIFIED RESULT_OUTCOME -->

其他限制：

- 僅研究 `pcb1` 與 `capsules`，不能直接推廣到其他工業物件。
- Synthetic Data 的視覺品質不等於下游任務有效性。
- 公開 Demo 是 research／evaluation tool，不是 production AOI、品質放行或安全系統。
- v2 pilot 是 exploratory validation；未通過 gate，因此沒有執行 confirmatory test。

## 重現方式

DefectForge 在原生 Windows 11、Python 3.12 與 RTX 4090 上開發並完成驗證。
資料根目錄只需在 `configs/paths.yaml` 設定一次；Source Code 不嵌入使用者專屬路徑。

```powershell
git clone https://github.com/kuotunyu/defectforge-visa-synthetic-data.git
Set-Location defectforge-visa-synthetic-data
uv sync --frozen --python 3.12

# 確認鎖定的 CUDA 環境。
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 下載並檢查 VisA、產生兩套官方 layout，獨立重驗 ADR-007。
uv run python scripts/download_visa.py
uv run python scripts/prepare_splits.py
uv run python scripts/verify_splits.py

# 快速 Repository 驗證；Training command 在此維持 dry-run。
uv run ruff check .
uv run pytest -q
uv run python scripts/run_classifier_matrix.py --dry-run
uv run python scripts/package_m18_colab.py --dry-run
```

完整執行順序、Immutable Checkpoint、預期 Sample Count 與每個 Milestone 的 Validator
記錄於 [PLAN.md](PLAN.md)。Configuration／CLI 契約見
[Interfaces](docs/interfaces.md)；預先註冊的下游 Protocol 見
[Experiment Protocol](docs/experiment_protocol.md)。

GPU 工作刻意拆分：

- SD2 Generation、SDXL Inference、Classification 與 SegFormer one-step smoke 在本機執行。
- SDXL LoRA Training 使用 `notebooks/01_train_inpaint_lora_sdxl.ipynb`。
- 16 個實體 SegFormer run 使用 `notebooks/02_train_segformer.ipynb`；
  `all_mixed` 引用 `filtered_syn`，不重複訓練。

兩個 Notebook 都只是 tracked Python trainer 的薄包裝，不會在 frozen manifest 之外
下載 test data，並提供 resumable checkpoint 與可獨立聚合的 raw report。
Colab 精確操作流程見 [instructions_for_me.md](instructions_for_me.md)。

## Repository 結構

| 路徑 | 內容 |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Repository 工作規則 |
| [PLAN.md](PLAN.md) | M0–M36 Milestone 與逐項驗證 |
| [docs/](docs/) | 方法論、Protocol、Spec、ADR、Worklog |
| [docs/diagrams/](docs/diagrams/) | 已驗證的 Mermaid 架構圖原始碼與 PNG |
| `src/` | Data／Synthetic／Filtering／Training／Evaluation／Inference Source Code |
| `splits/` | Frozen split manifest、Checksum、Defect Taxonomy |
| `notebooks/` | Colab Notebook |
| `reports/` | 自動產生的 Report、Evidence 與 Figure |
| `.claude/skills/` | 對應課程 Agentic Flow 的 Project-level Skill |

## 授權與引用

本 Repository 的 Source Code 採 **MIT License**（見 [LICENSE](LICENSE)）。
MIT 僅涵蓋程式碼；Dataset 與 Model Weights 各自保留原始條款：

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

主要引用：

- Zou et al., *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection
  and Segmentation*（ECCV 2022）：VisA Dataset 與官方 Split。
  [arXiv:2207.14315](https://arxiv.org/abs/2207.14315) ·
  [amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)
- Hu et al., *AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model*
  （AAAI 2024）：本 Pipeline 對照的 Spatial Anomaly Embedding 概念。
  [arXiv:2312.05767](https://arxiv.org/abs/2312.05767)
- Zavrtanik et al., *DRAEM*（ICCV 2021）：Procedural Anomaly Synthesis。
- Ghiasi et al., *Simple Copy-Paste is a Strong Data Augmentation Method*（CVPR 2021）。
- NVIDIA GTC Taiwan 2026, *Few-shot Industrial Synthetic Data Gen with NVIDIA Defect
  Image Generation Agent*：方法論靈感來源。本 Repository 是獨立 Open-source
  replication，與 NVIDIA 無隸屬或背書關係。
