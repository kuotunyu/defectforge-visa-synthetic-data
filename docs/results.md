# 補充結果

> 這份文件收錄從 [README](../README.md) 移出的結果細節。README 內由 `scripts/verify_readme.py` 產生並核對的表格仍留在 README；這裡的數字都抄自所註明的報告，以該報告為準。

## 結果圖表

![真實資料 Scaling Curve 與已篩選 Synthetic Data 等價量](../reports/figures/real_scaling_curve.png)

![五組 Classification 比較](../reports/figures/main_comparison_table.png)

![九個 Segmentation 邏輯組別](../reports/figures/segmentation_table.png)

其他圖：[合成量掃描](../reports/figures/synthetic_volume_curve.png) · [生成品質與下游表現](../reports/figures/quality_vs_downstream.png) · [pcb1 樣本圖](../reports/figures/sample_grid_pcb1.png) · [capsules 樣本圖](../reports/figures/sample_grid_capsules.png)

## v2 取樣檢驗

在完全不讀取 Test Set 的前提下，v2 pilot 檢驗 Label Balancing 是否讓 Synthetic Anomaly 過度占用正樣本曝光。數字來自 [`reports/v2_pilot_report.md`](../reports/v2_pilot_report.md)（validation：pcb1 正常 60／瑕疵 6 張，capsules 正常 36／瑕疵 6 張，見 [`results/v2/pilot_classification.json`](../results/v2/pilot_classification.json)）：

| Validation 候選方案 | pcb1 Macro-F1 | pcb1 AUROC | capsules Macro-F1 | capsules AUROC | 真實／合成瑕疵曝光次數 |
|---|---:|---:|---:|---:|---:|
| **Real-only** | 0.6944 | 0.9167 | 0.8133 | 0.9120 | 818 / 0 |
| **v1 Class-balanced Mixing** | 0.5537 | 0.3139 | 0.4545 | 0.2083 | 14 / 769 |
| **Domain-balanced 50% Real Bad** | 0.6944 | **0.9389** | 0.6571 | 0.7500 | 405 / 383 |
| **Domain-balanced 75% Real Bad** | 0.6944 | 0.8806 | 0.6571 | **0.8611** | 613 / 215 |

v1 的取樣器只平衡 good／bad 兩個類別，bad 類別內仍按樣本數均勻抽樣；10 張真實瑕疵對上 500 張合成瑕疵，所以 1,600 次抽樣裡真實瑕疵只出現 14 次。保留真實瑕疵曝光後 pcb1 的 Macro-F1 完全回復，capsules 仍低於 Real-only，事先寫定的三個條件都沒有通過，因此沒有進行 Test 評估（[ADR-026](decisions.md#adr-026)）。

## 曝光、外觀與面積機制檢驗

v2 之後的三次 pilot 都在執行前把判定規則 commit 進 Git：

| Pilot 階段 | 檢驗的機制假說 | 判定結果 | 預註冊依據 | 報告 |
|---|---|---|---|---|
| **v2** | 合成樣本淹沒真實瑕疵的**曝光**失衡 | 門檻未過；部分改善 | [ADR-026](decisions.md#adr-026) | [v2 pilot](../reports/v2_pilot_report.md) |
| **v3** | 效能落差來自**外觀**或放置位移 | 依物件而異；主要物件無鑑別力 | [ADR-035](decisions.md#adr-035--v3-歸因-pilot-的預註冊合成的殘餘落差來自放置還是外觀) | [v3 來源歸因](../reports/v3_source_attribution.md) |
| **v4** | 限制放置**面積**符合真實分布 | 未能檢驗（主指標無鑑別力） | [ADR-038](decisions.md#adr-038--v4-的預註冊把放置面積限回真實分布能不能改善下游) | [v4 面積檢驗](../reports/v4_placement_band.md) |
| **v5** | 同 v4，複跑至 3 seeds | **無效果**（有效陰性結論） | [ADR-040](decisions.md#adr-040--v5-的預註冊把-v4-複跑到-3-seeds判定只用沒被看過的-seed-4344) | [v5 複跑判定](../reports/v5_seed_replication.md) |

放置階段的直接量測見 [`reports/placement_geometry.md`](../reports/placement_geometry.md)。

## 程序化合成特徵洩漏檢驗

「僅 Procedural」組沒有讀取真實瑕疵像素，但它的 Mask 面積與長寬比被限制在 10 張 few-shot 訓練 Mask 的 5–95 百分位內（[ADR-011](decisions.md#adr-011)）。與 `--no-real-stats` 對照組的比較表由 `scripts/verify_readme.py` 產生，留在 [README 的實驗結果](../README.md#實驗結果)；使用統計量的版本反而表現較差，所以這個洩漏面沒有造成人為的分數抬升。分割版的對照組決定不補，理由見 [ADR-042](decisions.md#adr-042--分割版---no-real-stats-對照組決定不補改為把已量測的分類版揭露到-readme)。

## 分割的零 Dice 與增強崩潰

- [`reports/zero_dice_diagnosis.md`](../reports/zero_dice_diagnosis.md)：48 個實跑 run 逐一列出最高預測機率與越過 threshold 的像素數（[ADR-030](decisions.md#adr-030)、[ADR-034](decisions.md#adr-034--零-dice-有兩種成因adr-030-的完全由機率天花板造成在-48-個-run-上不成立)）
- [`reports/augmentation_mask_loss.md`](../reports/augmentation_mask_loss.md)：`capsules/std_aug` 的崩潰不是增強把瑕疵裁掉，而是訓練期的 Dice 項沒有啟動（[ADR-031](decisions.md#adr-031)）
- [`reports/segmentation_replication.md`](../reports/segmentation_replication.md)：3-seed 複跑的兩條事先寫定規則與判定（[ADR-032](decisions.md#adr-032)）
