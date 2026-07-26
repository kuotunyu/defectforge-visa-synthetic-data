# Experiment Protocol

> **Phase 2 才執行，但協定在 Phase 1 就凍結**——先定好規則再看結果，才不會事後調整標準。
> 相關：[ADR-003](decisions.md#adr-003) few-shot 預算、[data_protocol.md §6](data_protocol.md) 防洩漏聲明。

---

## 1. 五組對照

| # | 組別 | 真實資料 | 合成資料 | 這一組回答什麼問題 |
|---|---|---|---|---|
| 1 | **Real-only** | few-shot（k=10 瑕疵 + train pool 全部正常） | 無 | 基準線 |
| 2 | **+ Standard Augmentation** | 同上 | 無（只做傳統增強：翻轉、旋轉、色彩抖動、RandomResizedCrop 等） | **排除「一般 augmentation 就能達成」的質疑**——這是最關鍵的消融 |
| 3 | **+ Unfiltered Synthetic** | 同上 | 未過濾版 | 不過濾的合成資料有沒有害？ |
| 4 | **+ Filtered Synthetic** | 同上 | 過濾版 | **主成果**，同時證明過濾管線的價值 |
| 5 | **Full-real 上限** | `2cls_highshot` train（約 60 張瑕疵/物件） | 無 | 上限參照。**標示為不同（較大）的真實預算**，非同條件對照 |

**第 1–4 組吃的真實影像必須完全相同**，合成只作為增量。第 2 組不得使用任何額外真實資料。

### 額外消融（Phase 2 視時間執行）
- **合成量掃描**：Filtered 組另測 0.5× / 1× / 2×（相對真實正常樣本數）→ 畫出對應課程
  slide 8 的「SDG 張數 vs 下游提升」曲線
- **合成來源拆解**（分割實驗主打）：程序化-only（「零真實瑕疵」）／copy-paste-only／
  diffusion-only／全部混合
- **底模容量**：stageB_sd2 vs stageB_sdxl（[ADR-001](decisions.md#adr-001)）
- **refine 效果**：`original/` vs `searched/` 兩桶
- **（選配）嚴格 k=10/k=10** 官方協定當附錄

---

## 2. 下游任務與模型

| 任務 | 模型 | 輸入 | 主要指標 |
|---|---|---|---|
| 瑕疵分類 | ConvNeXt-Tiny（timm, ImageNet 預訓練） | 384 | Macro-F1、AUROC、per-object F1、**正常品 false-positive rate** |
| 瑕疵區域分割 | SegFormer-B0 | 512 | Dice、mIoU、pixel AUROC、AUPRO |

分類在本機 4090 跑（單組訓練短）；分割走 Colab notebook。

---

## 3. 公平性規則

- **對齊訓練預算**：各組盡量對齊 optimizer steps、batch size 與「真實樣本曝光次數」。
  合成資料使訓練集變大時，用固定 total steps 而非固定 epochs，並在報表註明每組的
  真實樣本曝光次數。
- **Seed 策略（省算力）**：所有組合先跑 **1 個 seed**；只對 **Real-only** 與
  **最佳 Filtered 組**補到 **3 seeds**，報 mean ± std。
- **超參數**：所有組共用同一組超參，由 Real-only 組在 validation 上調出來後**凍結**。
  不准為 Filtered 組單獨調參（那會讓比較失去意義）。
- **Early stopping**：以 validation（真實）指標為準，各組相同 patience。

---

## 4. 防洩漏檢查表（每次跑實驗前逐項打勾）

- [ ] Validation / Test 只含真實影像
- [ ] 生成器（LoRA、mask placement、inpaint）從未讀過 test 路徑
- [ ] 過濾器與品質指標的參考集只用 few-shot seed
- [ ] 瑕疵分型的分群只在 few-shot seed 上做
- [ ] split manifest 在第一張合成圖生成前已凍結，SHA256 與當前檔案相符
- [ ] pHash 同群同 split 的斷言通過
- [ ] 每個訓練組的資料清單都可以追回 `splits/split_manifest.json` 的條目
- [ ] `df-guard` skill 的 preflight 全綠

---

## 5. 報告規則

- 所有表格數字**一律從 `results/` 的原始 metrics 檔重新聚合計算**，不准抄 notebook 畫面上的值
- **若 synthetic 沒有提升，如實保留結果並分析失敗原因**——負面結果＋誠實分析比挑選性報告更有價值
- 報表要同時放「生成品質分數」與「下游提升」，討論兩者是否相關
  （呼應課程 slide 8 明說的 cons：影像指標可能無法反映下游提升）
- 每張圖表產出後**自己開起來看過**才放進 README

---

## 6. 輸出檔案

| 檔案 | 內容 |
|---|---|
| `results/classification.csv` | 分類五組 × 各指標的原始數字 |
| `results/segmentation.csv` | 分割各組 × 各指標 |
| `reports/figures/f1_vs_synthetic_volume.png` | 合成量掃描曲線 |
| `reports/figures/main_comparison_table.png` | 五組對照主表 |
| `reports/figures/sample_grid_*.png` | 真假樣本並排 |
| `scripts/verify_readme.py` | 自動核對 README 每張表的數字都能從 `results/` 重算出來 |
