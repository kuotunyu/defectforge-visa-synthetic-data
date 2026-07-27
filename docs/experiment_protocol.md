# Experiment Protocol (Phase 2)

> **協定在 Phase 1 就凍結**——先定好規則再看結果，才不會事後調整標準。
> 相關：[ADR-003](decisions.md#adr-003) few-shot 原則、[ADR-007](decisions.md#adr-007) 基底 split、
> [ADR-009](decisions.md#adr-009) 下游設計、[ADR-010](decisions.md#adr-010) 合成量、
> [ADR-011](decisions.md#adr-011) 程序化口徑。

---

## 1. 資料基礎（所有實驗共用）

基底 partition = **`2cls_highshot`**，唯一 test set = **highshot test**（[ADR-007](decisions.md#adr-007)）。

| 角色 | pcb1 | capsules |
|---|---|---|
| **Test（凍結，唯一）** | 402 normal / 40 anomaly | 241 normal / 40 anomaly |
| Train pool | 602 normal / 60 anomaly | 361 normal / 60 anomaly |
| few-shot 瑕疵集（k=10） | 10 anomaly | 10 anomaly |
| 正常圖（所有組共用） | 602 | 361 |

⛔ **絕不可用 `2cls_fewshot` 的 test 評測**——會洩漏一半的測試瑕疵。

---

## 2. 五組對照（分類與分割共用）

| # | 組別 | 真實瑕疵 | 真實正常 | 合成 | 回答什麼問題 |
|---|---|---|---|---|---|
| 1 | **Real-only** | k=10 | 全部 | 無 | 基準線 |
| 2 | **+ Standard Augmentation** | k=10 | 全部 | 無（只加傳統增強） | **排除「一般 augmentation 就能達成」的質疑**——最關鍵的消融 |
| 3 | **+ Unfiltered Synthetic** | k=10 | 全部 | 未過濾（全來源混合） | 不過濾的合成資料有沒有害？ |
| 4 | **+ Filtered Synthetic** | k=10 | 全部 | 過濾後（全來源混合） | **主成果**，同時證明過濾管線的價值 |
| 5 | **Full-real 上限** | **60** | 全部 | 無 | 上限參照，**同一個 test set，零洩漏** |

第 1–4 組吃的真實影像**完全相同**，合成只作為增量。第 2 組不得使用任何額外真實資料。

### 2.1 真實資料縮放曲線（[ADR-007](decisions.md#adr-007) 的免費附贈）

因為 `k=10 ⊂ fewshot_train(20) ⊂ highshot_train(60)` 巢狀，
額外跑 **真實瑕疵 = 10 / 20 / 60** 三點（無合成），得到一條純真實資料的縮放曲線。

**這是本專案最有價值的單一數字**：把 Filtered Synthetic 組落在這條曲線上，
就能回答「**我們的合成資料相當於多少張真實瑕疵**」。README 要把這句話放在最前面。

---

## 3. 分類實驗（本機 4090）

| 項目 | 設定 |
|---|---|
| 模型 | ConvNeXt-Tiny（`timm`，ImageNet 預訓練） |
| 輸入 | 384 |
| 任務 | **每物件一個二元分類器**（good / bad）——對應 VisA 官方 2-class 設定（[ADR-009](decisions.md#adr-009)） |
| 不平衡 | pcb1 約 60:1、capsules 約 36:1 → **所有組共用同一套 class-balanced sampling** |
| 指標 | Macro-F1、AUROC、per-object F1、**正常品 false-positive rate**（AOI 實務最在意的 overkill rate） |
| 輸出 | `results/classification.csv`（long format，每列一個 run） |

### 3.1 要跑的 run

| 系列 | run 數 | 說明 |
|---|---|---|
| 五組對照 | 5 × 2 物件 = 10 | §2 的五組 |
| 真實縮放曲線 | 2 × 2 = 4 | 真實瑕疵 20 / 60（10 已在 Real-only）|
| 合成量掃描 | 3 × 2 = 6 | Filtered Syn @ **{125, 250, 500}**（[ADR-010](decisions.md#adr-010)） |
| 來源消融 | 3 × 2 = 6 | procedural-only / copy-paste-only / diffusion-only |
| 底模消融 | 1 × 2 = 2 | SD2 vs SDXL **同在 250 張**這一點比較（SD2@250 已在掃描裡） |
| 程序化口徑 | 1 × 2 = 2 | procedural `--no-real-stats` 版（[ADR-011](decisions.md#adr-011)） |
| refine 消融 | 1 × 2 = 2 | `original/` vs `searched/`（`searched` 已在主線） |
| **3-seed 複跑** | +2 × 2 × 2 = 8 | 只對 **Real-only** 與**最佳 Filtered 組**補到 3 seeds |
| **合計** | **約 40 run** | ConvNeXt-Tiny 小資料，單 run 估 5–15 分鐘 → 本機約 4–8 小時 |

### 3.2 Canonical run 與 alias（[ADR-020](decisions.md#adr-020)）

同一份資料定義不為了湊數重跑：`real_60` 引用 `full_real`、`syn_500` 引用
`filtered_syn`、`base_sd2` 引用 `bucket_searched`。seed 42 實跑 15 個 canonical
group × 2 物件，再為事前指定的 `real_only` 與主結果 `filtered_syn` 各補 seeds 43、
44，共 **38 個實跑**。

`filtered_syn` / `unfiltered_syn` / `{125,250,500}` 使用 source×defect-type 的決定性
分層抽樣；三個來源消融各用該來源原始 500 張；`bucket_original` /
`bucket_searched` 與 `base_sdxl` 固定 250 張原始 diffusion，讓 refine 與底模比較不被
不同過濾通過率混淆。任何缺少的來源都 fail closed。

---

## 4. 分割實驗（Colab T4）

| 項目 | 設定 |
|---|---|
| 模型 | SegFormer-B0 |
| 輸入 | 512 |
| Loss | Dice + BCE |
| 指標 | Dice、mIoU、pixel AUROC、**AUPRO** |
| 輸出 | `results/segmentation.csv` |

**九組**（[ADR-009](decisions.md#adr-009)）：五組鐵律 + 四組來源消融。

| 組 | 訓練資料 | 備註 |
|---|---|---|
| 1 Real-only | k=10 + 正常圖（全零 mask） | |
| 2 + Standard Aug | 同上 + 傳統增強 | |
| 3 + Unfiltered Syn | 同上 + 未過濾合成 | |
| 4 + Filtered Syn | 同上 + 過濾後合成 | 主成果 |
| 5 Full-real 上限 | 60 張真實瑕疵 + 正常圖 | |
| 6 **程序化-only** | **正常圖 + 程序化合成，零真實瑕疵影像** | **最有記憶點的一組**，口徑見 [ADR-011](decisions.md#adr-011) |
| 7 copy-paste-only | k=10 + copy-paste 合成 | |
| 8 diffusion-only | k=10 + Stage B 合成 | |
| 9 全部混合 | **= 第 4 組，直接引用不重跑** | |

→ 實際只需跑 **8 次 × 2 物件**。T4 每次估 20–40 分鐘 → 約 5–11 小時 ≈ 9–20 CU。
notebook 用 `--group` 參數切換，可平行開多本（各自唯一 `runs/` 目錄）。

正常圖要以全零 mask 加入訓練，教模型不要過度分割——這會直接反映在 false-positive 上。

---

## 5. 生成品質表

每瑕疵型 × 每 generator：FID、KID（`clean-fid`，**小樣本主報 KID**）、
`nn_score`、`mnn_score`（定義見 [ADR-006](decisions.md#adr-006) 與 [filtering_spec.md](filtering_spec.md)），
加上真假對照 grid。

**必做的討論**：把「生成品質分數」與「下游提升」畫成散點圖，看兩者是否相關。
課程 slide 8 明說影像指標的 cons 是「可能無法反映下游提升」——
若我們的資料也顯示不相關，那是一個很好的實證發現，要寫進 README 而不是藏起來。

---

## 6. 公平性規則

- **對齊訓練預算**：合成資料使訓練集變大時，用**固定 total optimizer steps** 而非固定 epochs；
  報表要註明每組的「真實樣本曝光次數」
- **Seed 策略**：所有組合先跑 1 個 seed；只對 **Real-only** 與事前指定的主
  **Filtered Synthetic 組**補到 3 seeds，
  報 mean ± std。Test 只有 40 張瑕疵／物件，**不要對 <1% 的差異下結論**
- **超參數**：所有組共用同一組超參，由 Real-only 組在 validation 上調出來後**凍結**。
  **不准為 Filtered 組單獨調參**
- **Early stopping**：以 validation（真實）指標為準，各組相同 patience
- **Validation 集合**：所有組共用同一個，從 train pool 切、只含真實資料
- **開發與 final refit**：validation 是 highshot train 每物件 × label 的固定 10% holdout，
  且排除官方 fewshot train pool；只用 Real-only 決定共用超參與 final steps。正式五組比較
  用凍結設定 refit 完整 train pool，維持 10 / 20 / 60 口徑（[ADR-013](decisions.md#adr-013)）

---

## 7. 防洩漏檢查表（每次跑實驗前逐項打勾，由 `df-guard` 執行）

- [ ] 評測用的是 **highshot test**，不是 fewshot test
- [ ] 每個訓練組 ∩ highshot test = **∅**（用 SHA256 比對，不是比路徑字串）
- [ ] Validation / Test 只含真實影像，沒有任何合成樣本混入
- [ ] 生成器（LoRA、mask placement、inpaint）從未讀過 test 路徑
- [ ] 過濾器與品質指標的參考集只用 few-shot seed 的真實瑕疵 crop
- [ ] 瑕疵分型的分群只在 few-shot seed 上做
- [ ] split manifest 的 SHA256 與 `splits/MANIFEST.sha256` 相符（沒有被動過）
- [ ] pHash 同群同 split 的斷言通過
- [ ] 每個訓練組的資料清單都能追回 `splits/split_manifest.json` 的條目
- [ ] 程序化-only 組若宣稱「零真實瑕疵」，`--no-real-stats` 版本確實沒讀過 `real_mask_stats.json`

---

## 8. Gradio Demo（本機 4090）

上傳影像 → 輸出：正常／瑕疵機率、瑕疵 mask、heatmap、單張延遲。
用最佳的分類器 + 分割模型。錄成 GIF 放 README，GIF 存 `assets/demo.gif`。
Demo 只在本機跑，**不部署到公開空間**（除非使用者另外要求）。

---

## 9. 報告規則

- 所有表格數字**一律從 `results/` 的原始 metrics 檔重新聚合計算**，不准抄 notebook 畫面上的值
- `scripts/verify_readme.py` 要能自動核對 README 每張表的數字都可從 `results/` 重算出來
- **若 synthetic 沒有提升，如實保留結果並分析失敗原因**——負面結果＋誠實分析比挑選性報告更有價值
- 每張圖表產出後**自己開起來看過**才放進 README

---

## 10. 輸出檔案

| 檔案 | 內容 |
|---|---|
| `results/classification.csv` | 所有分類 run 的原始數字（long format） |
| `results/segmentation.csv` | 所有分割 run 的原始數字 |
| `reports/figures/real_scaling_curve.png` | **真實瑕疵 10/20/60 曲線 + Filtered Syn 落點**（頭號圖表） |
| `reports/figures/synthetic_volume_curve.png` | 合成量 125/250/500 掃描，與課程曲線並排 |
| `reports/figures/main_comparison_table.png` | 五組對照主表 |
| `reports/figures/segmentation_table.png` | 九組分割對照 |
| `reports/figures/quality_vs_downstream.png` | 生成品質分數 vs 下游提升散點圖 |
| `reports/figures/sample_grid_*.png` | 真假樣本並排 |
| `assets/demo.gif` | Gradio demo 錄影 |

---

## 11. 選配 stretch goal（**先問使用者**）

Colab A100 跑 NVIDIA 官方 `nvidia/Cosmos-AnomalyGen-Metal-2B` inference（**gated model，
需要使用者先同意授權**），與我們的生成做**質性**對照，README 加一節
「與 NVIDIA 官方 pipeline 的比較」。

A100 約 10–15 CU/hr，會明顯吃額度 → **不准自作主張啟動，一定要先問**。
