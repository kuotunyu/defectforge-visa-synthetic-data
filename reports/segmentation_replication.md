# 分割 3-seed 複跑的預註冊判定

本報告只執行 [ADR-032](../docs/decisions.md#adr-032) 在**複跑開始前**就寫死的兩條規則。數字全部由 `scripts/decide_segmentation_replication.py` 從 `results/segmentation.csv` 產生，該表由 M20 聚合器從 raw `training_report.json` 重建。

Seeds：42, 43, 44

## 規則 1 — Dice／AUPRO 方向矛盾

判準：`filtered_syn − real_only` 的 Dice 差與 AUPRO 差符號相反，出現在 3 個 seed 中的 ≥ 2 個，且至少發生在一個物件上。

| 物件 | Seed | Dice Δ | AUPRO Δ | 符號相反 |
|---|---:|---:|---:|:---:|
| pcb1 | 42 | -0.3140 | +0.1443 | 是 |
| pcb1 | 43 | -0.2788 | -0.0226 | 否 |
| pcb1 | 44 | -0.2657 | +0.1081 | 是 |
| capsules | 42 | -0.1387 | +0.0649 | 是 |
| capsules | 43 | -0.4573 | -0.4884 | 否 |
| capsules | 44 | -0.5230 | -0.5410 | 否 |

| 物件 | 符號相反的 seed | 達標 | Dice Δ（mean ± std） | AUPRO Δ（mean ± std） |
|---|---|:---:|---:|---:|
| pcb1 | 42, 44 | 是 | -0.2862 ± 0.0250 | +0.0766 ± 0.0878 |
| capsules | 42 | 否 | -0.3730 ± 0.2055 | -0.3215 ± 0.3357 |

**判定：真實現象**（達標物件：pcb1）

## 規則 2 — `capsules/std_aug` 崩潰

判準：Dice 在 3 個 seed 中的 ≥ 2 個為 `0.0000`。

| Seed | Dice | Pixel AUROC |
|---:|---:|---:|
| 42 | 0.0000 | 0.8661 |
| 43 | 0.4212 | 0.9846 |
| 44 | 0.0000 | 0.8447 |

Dice 為零的 seed：42, 44

**判定：系統性**

ADR-031 的主張維持不變。
