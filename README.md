# DefectForge：VisA Synthetic Data 瑕疵生成與評估

[![MIT License](https://img.shields.io/badge/License-MIT-08796c.svg)](LICENSE)
[![Latest Release](https://img.shields.io/github/v/release/kuotunyu/defectforge-visa-synthetic-data?color=08796c)](https://github.com/kuotunyu/defectforge-visa-synthetic-data/releases/latest)

> **狀態：v1.2.1 已完成並公開。**
> 下游結果表與最終結論皆由通過獨立驗證的 CSV 自動產生，不手動填寫數字。
> 驗證入口請見 `scripts/verify_readme.py` 與 `scripts/verify_publish.py`。

公開成果：
[Synthetic Dataset](https://huggingface.co/datasets/steven0226/defectforge-visa-synthetic) ·
[SD2／SDXL LoRA Weights](https://huggingface.co/steven0226/defectforge-visa-lora) ·
[正體中文互動 Demo](https://steven0226-defectforge-visa-demo.hf.space/) ·
[Release](https://github.com/kuotunyu/defectforge-visa-synthetic-data/releases)

在每個物件只有 **10 張真實瑕疵圖**的條件下，本專案以 Open Model、Open Data
重建 NVIDIA GTC 2026 Cosmos AnomalyGen 方法論。主實驗結果顯示：已篩選
Synthetic Data **沒有**優於 Real-only；本專案完整保留這個負面結果，並用 v2 pilot
檢驗 Sampling 是否造成退步。

---

## 研究問題

Synthetic Data 能否改善少樣本工業瑕疵的 Classification 與 Segmentation？

| 項目 | 設定 |
|---|---|
| Dataset | [VisA](https://registry.opendata.aws/visa/)（CC BY 4.0）：`pcb1`、`capsules` |
| 真實瑕疵預算 | 每個物件 10 張，seed=42 |
| 下游任務 | Defect Classification＋Defect-region Segmentation |
| Synthetic Data 標註 | 全自動；放置用 Mask 就是 Segmentation Ground Truth |

### 防止 Split 洩漏

VisA 官方 `2cls_fewshot` 與 `2cls_highshot` CSV 是同一批影像的兩種不同切分。
在開始撰寫訓練程式前，我們先量測兩者的交集：

```text
highshot TRAIN(anomaly) ∩ fewshot TEST(anomaly) = 40   （每個物件，test 共 80 張）
fewshot TRAIN ⊂ highshot TRAIN                          True
highshot TEST ⊂ fewshot TEST                            True
```

混用兩套 Split 會把**一半的 test 瑕疵放進 training**。因此所有實驗統一以
`2cls_highshot` 為 base partition，共用同一個 frozen test set；完整決策見
[ADR-007](docs/decisions.md#adr-007)。

## 方法與系統架構

![DefectForge Synthetic Data 系統架構](docs/diagrams/readme_01_flowchart_system_architecture.png)

[查看 Mermaid 原始碼](docs/diagrams/readme_01_flowchart_system_architecture.mmd)

流程包含 frozen split 與 test blocklist、Copy-paste／Procedural／Diffusion
生成、六道 Quality Filtering、ConvNeXt-Tiny Classification、SegFormer-B0
Segmentation，以及 SHA256-bound evidence。Open-source 元件對照與工程詮釋見
[方法論文件](docs/methodology.md)。

## 實驗設計

主實驗包含五組控制組；第 1–4 組使用完全相同的真實資料，Synthetic Data 僅能額外加入，
所有組別都在同一個 frozen test set 評估：

1. **Real-only**：10 張真實瑕疵圖
2. **+ Standard Augmentation**：排除一般 augmentation 就足以改善的可能
3. **+ 未篩選 Synthetic Data**
4. **+ 已篩選 Synthetic Data**：主比較組
5. **Full-real upper bound**：60 張真實瑕疵圖

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

#### 預註冊 3-seed 複跑的 mean ± std

實驗協定要求 Real-only 與最佳 Filtered 組各複跑 3 個 seed。上表為 seed 42 的單次結果，
下表是同一批 run 的 seed 變異：

<!-- BEGIN VERIFIED CLASSIFICATION_SEED_VARIANCE -->
| 物件 | 訓練組別 | Seeds | Macro-F1（mean ± std） | AUROC（mean ± std） |
| --- | --- | --- | --- | --- |
| pcb1 | Real-only（10 張） | 3 | 0.6808 ± 0.0031 | 0.9265 ± 0.0231 |
| pcb1 | + 已篩選 Synthetic Data | 3 | 0.3175 ± 0.1066 | 0.1677 ± 0.0502 |
| capsules | Real-only（10 張） | 3 | 0.5471 ± 0.0268 | 0.8160 ± 0.0224 |
| capsules | + 已篩選 Synthetic Data | 3 | 0.3609 ± 0.0201 | 0.3243 ± 0.0426 |
<!-- END VERIFIED CLASSIFICATION_SEED_VARIANCE -->

已篩選 Synthetic 組的 seed 間標準差比 Real-only 大一個數量級以上。這個變異度本身就說明：
在這個資料規模下，單一 seed 的細部差異不足以支撐結論。其餘組別只有 seed 42，因此不在表內；
Segmentation 目前完全沒有 seed 複跑，這一點列在下方「限制與誠實揭露」。

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

#### Dice 與 AUPRO 的方向不一致

Dice 依賴固定 threshold 0.5，AUPRO 不依賴 threshold。兩者對同一批 run 給出相反的方向：

<!-- BEGIN VERIFIED SEGMENTATION_THRESHOLD -->
| 物件 | 訓練組別 | Dice（threshold 0.5） | AUPRO（不依賴 threshold） | Dice Δ vs Real-only | AUPRO Δ vs Real-only |
| --- | --- | --- | --- | --- | --- |
| pcb1 | Real-only（10 張） | 0.3762 | 0.6028 | — | — |
| pcb1 | + Standard Augmentation | 0.3836 | 0.5740 | +0.0074 | -0.0288 |
| pcb1 | + 未篩選 Synthetic Data | 0.2490 | 0.6065 | -0.1272 | +0.0037 |
| pcb1 | + 已篩選 Synthetic Data | 0.0621 | 0.7471 | -0.3140 | +0.1443 |
| pcb1 | Full-real（60 張） | 0.6862 | 0.5999 | +0.3100 | -0.0029 |
| capsules | Real-only（10 張） | 0.5958 | 0.8488 | — | — |
| capsules | + Standard Augmentation | 0.0000 | 0.5591 | -0.5958 | -0.2897 |
| capsules | + 未篩選 Synthetic Data | 0.0000 | 0.1666 | -0.5958 | -0.6821 |
| capsules | + 已篩選 Synthetic Data | 0.4570 | 0.9137 | -0.1387 | +0.0649 |
| capsules | Full-real（60 張） | 0.6331 | 0.9591 | +0.0373 | +0.1103 |
<!-- END VERIFIED SEGMENTATION_THRESHOLD -->

原因是有相當比例的 run 在 threshold 0.5 下輸出全背景 Mask（Dice 恰為 0），
但它們的 pixel AUROC 仍然很高——代表機率圖是有訊號的，是固定 threshold 把它切成全黑，
而不是模型完全失效。確切的 run 數與最高 pixel AUROC 列在下方「限制與誠實揭露」。

本專案**不因此改換主指標**：預註冊的主結論仍以 Dice 與 Macro-F1 為準，
AUPRO 與 threshold 敏感度併列揭露，讓讀者自行判斷。決策見
[ADR-027](docs/decisions.md#adr-027)。

## v2 後續實驗：退步是否來自 Sampling？

在完全不讀取 test set 的前提下，v2 pilot 檢驗 Label balancing 是否讓
Synthetic Anomaly 過度占據 positive-class exposure：

| Validation candidate | pcb1 Macro-F1 | pcb1 AUROC | capsules Macro-F1 | capsules AUROC |
|---|---:|---:|---:|---:|
| Real-only | 0.6944 | 0.9167 | 0.8133 | 0.9120 |
| v1 Class-balanced Mixing | 0.5537 | 0.3139 | 0.4545 | 0.2083 |
| Domain-balanced 50% real bad | 0.6944 | **0.9389** | 0.6571 | 0.7500 |
| Domain-balanced 75% real bad | 0.6944 | 0.8806 | 0.6571 | **0.8611** |

Domain balancing 救回 pcb1 與部分 capsules，但預先註冊的跨物件 gate 仍未通過，
因此在 test evaluation 與 three-seed confirmatory run 前停止。這是
**exploratory validation result**，不取代 v1 結果；詳見
[v2 Pilot Report](reports/v2_pilot_report.md)。

## 公開互動 Demo

![DefectForge 決定性 Demo](assets/demo.gif)

[開啟正體中文 DefectForge Demo](https://steven0226-defectforge-visa-demo.hf.space/)

Demo 使用 CPU Basic，會驗證 checkpoint SHA256、不保存上傳影像，並提供五張 VisA
範例；輸出包含 Classification Confidence、Binary Mask、Probability Heatmap 與
checkpoint provenance。受 SegFormer upstream License 限制，僅供
non-commercial research／evaluation。

## 限制與誠實揭露

<!-- BEGIN VERIFIED RESULT_OUTCOME -->
- Classification：已篩選 Synthetic Data 相對 Real-only 的平均 Macro-F1 差異為 `-0.2286`。
- Segmentation：已篩選 Synthetic Data 相對 Real-only 的平均 Dice 差異為 `-0.2264`。
- Classification 負面結果：**是——已篩選 Synthetic Data 未提升平均 Macro-F1。**
- Segmentation 負面結果：**是——已篩選 Synthetic Data 未提升平均 Dice。**
- Segmentation（threshold-free）：同一組 run 的平均 AUPRO 差異為 `+0.1046`，與 Dice **方向相反**。
- 16 個實跑的 Segmentation run 中有 6 個在固定 threshold 0.5 下 Dice = 0（整張預測為背景），其中 3 個的 pixel AUROC 仍達 0.80 以上（最高 `0.9015`）。
- 主結論仍以預註冊的 Macro-F1 與 Dice 為準；AUPRO 與 threshold 敏感度是**併列揭露**，不是事後換指標。
<!-- END VERIFIED RESULT_OUTCOME -->

- **Segmentation 完全沒有 seed 複跑**（全部 seed 42）。Classification 只有 Real-only 與
  已篩選 Synthetic 兩組達到預註冊的 3 seed。因此 Dice 與 AUPRO 的方向矛盾**無法**
  用現有證據判定孰是孰非；要解決必須補跑 Segmentation 的 3-seed 複跑。
- 合成組的正樣本曝光高度偏向 Synthetic Data：`results/classification.csv` 的
  `sampled_real_bad` / `sampled_synthetic_bad` 顯示，加入 500 張合成瑕疵後，
  真實瑕疵在同一份 sampling schedule 中的曝光量遠低於 Real-only。
  這是 v2 pilot 檢驗的假說，也代表 v1 的負面結果**同時**受合成品質與 sampling 設計影響，
  不能單獨歸因於「合成資料無效」。
- 僅研究 `pcb1` 與 `capsules`，不能直接推廣到其他工業物件。
- Synthetic Data 的視覺品質不等於下游任務有效性。
- 預註冊 threshold 0.5 下，四張決定性 Demo frame 的 Binary Mask coverage 都是 0。
- 公開 Demo 是 research／evaluation tool，不是 production AOI、品質放行或安全系統。
- v2 pilot 是 exploratory validation；未通過 gate，因此沒有執行 confirmatory test。

## 重現方式

開發與驗證環境為 Windows 11、Python 3.12、RTX 4090；資料路徑統一由
`configs/paths.yaml` 管理。

```powershell
git clone https://github.com/kuotunyu/defectforge-visa-synthetic-data.git
Set-Location defectforge-visa-synthetic-data
uv sync --frozen --python 3.12

uv run python scripts/download_visa.py
uv run python scripts/prepare_splits.py
uv run python scripts/verify_splits.py

uv run ruff check .
uv run pytest -q
uv run python scripts/verify_publish.py
```

Configuration／CLI 契約見 [Interfaces](docs/interfaces.md)，預先註冊的下游設計見
[Experiment Protocol](docs/experiment_protocol.md)，Colab 薄封裝位於
[`notebooks/`](notebooks/)。

## Repository 結構

| 路徑 | 內容 |
|---|---|
| `src/` | Data／Synthetic／Filtering／Training／Evaluation／Inference Source Code |
| `configs/`、`splits/` | 可重現設定、Frozen Manifest 與 Test Blocklist |
| `scripts/`、`tests/` | 執行入口、獨立 Validator 與測試 |
| `docs/` | 方法論、Experiment Protocol、CLI 契約與 ADR |
| `reports/`、`results/` | SHA256-bound Evidence、Figure 與凍結結果 |
| `notebooks/` | Colab Notebook |

## 授權與引用

本 Repository 的 Source Code 採 **MIT License**（見 [LICENSE](LICENSE)）。
MIT 僅涵蓋程式碼；Dataset、Synthetic Images 與 Model Weights 各自保留原始條款。
可直接引用的 Metadata 見 [CITATION.cff](CITATION.cff)，第三方資產總表見
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)：

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

如需引用 DefectForge，可在 GitHub 右側 **Cite this repository** 取得 Metadata；
若使用 VisA 或衍生 Synthetic Data，仍須引用 VisA 原始論文。完整論文與上游授權
見 [方法論](docs/methodology.md)與 [License Chain](docs/license_chain.md)。
