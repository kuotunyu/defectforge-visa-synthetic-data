# v4 放置面積 pilot 的預註冊判定

本報告只執行 [ADR-038](../docs/decisions.md#adr-038) 在**執行前**寫死的規則。數字由 `scripts/decide_v4_placement_band.py` 從 `results/v4/pilot_classification.json` 產生。

主要物件：`pcb1`　對照物件：`capsules`　指標：`macro_f1`

| 物件 | real_only | db_current | db_inband | D = inband − current | 指標可分辨 |
|---|---:|---:|---:|---:|:---:|
| capsules | 0.8133 | 0.6815 | 0.6889 | +0.0074 | 是 |
| pcb1 | 0.6944 | 0.6944 | 0.6944 | +0.0000 | **否** |

**判定：**本次未能檢驗假說**（主要物件無鑑別力）**

主要物件 `pcb1` 上三個 candidate 的 `macro_f1` **完全相同**，該指標在此物件無法分辨任何差異。ADR-038 已預先規定無鑑別力的物件不得當作證據，因此本次**不是**「沒有效果」，而是**這個實驗沒有真正檢驗到自己的假說**。

這是預註冊設計的缺陷：[ADR-036](../docs/decisions.md#adr-036) 已經記錄過`pcb1` 的 `macro_f1` 對所有 candidate 相同，ADR-038 卻仍以此組合作為主要判準。

對照組 `capsules`：`D = +0.0074`，門檻 `0.05`，判定正常。

## Confirmatory gate

- gate 狀態：`stopped`
- 是否授權 confirmatory test：**否**

判定結論成立**不等於** gate 通過。gate 未過時一律不得讀 frozen test、不得跑 3 seeds（ADR-038）。
