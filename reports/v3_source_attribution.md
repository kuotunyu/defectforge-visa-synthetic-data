# v3 來源歸因 pilot 的預註冊判定

本報告只執行 [ADR-035](../docs/decisions.md#adr-035) 在**執行前**寫死的規則。數字由 `scripts/decide_v3_source_attribution.py` 從 `results/v3/pilot_classification.json` 產生。

## 兩個懲罰量（validation Macro-F1）

- 放置懲罰 `P = real_only − db_copypaste`
- 外觀懲罰 `A = db_copypaste − db_diffusion`

| 物件 | P（放置＋縫合） | A（外觀） | 該物件較大者 | 指標可分辨 |
|---|---:|---:|---|:---:|
| capsules | +0.1319 | -0.0074 | placement | 是 |
| pcb1 | +0.0000 | +0.0000 | tie | **否** |

**判定：依物件而異，無單一主因**

⚠️ **pcb1 上三個 candidate 的 macro_f1 完全相同**，代表該指標在這個物件上無法分辨，其「平手」不是兩種成因勢均力敵的證據，而是**這個物件沒有提供資訊**。判定仍照預註冊規則計入，但解讀時必須扣除。

兩物件平均：P `+0.0659`、A `-0.0037`。

## 這個歸因的界限

copy-paste 的瑕疵像素是**真實**的，但它的羽化／Poisson 縫合是合成的。因此 `P` 綁著「放置」與「縫合」兩件事，**不是**純粹的放置效應。這是兩個 bundle 的歸因，不是三個乾淨因子的分解。

## Confirmatory gate

- gate 狀態：`stopped`
- 是否授權 confirmatory test：**否**

歸因結論成立**不等於** gate 通過。gate 未過時一律不得讀 frozen test、不得跑 3 seeds（ADR-035）。
