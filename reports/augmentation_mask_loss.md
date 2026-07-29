# 標準增強是否把瑕疵切掉的診斷

> 由 `scripts/diagnose_augmentation_mask_loss.py` 產生。
> **只讀 train partition**：重放 trainer 實際會抽到的同一組 draw（同 transform、同 `_stable_seed` 序列），量測 mask 是否還在。
> 不載入模型、不重算任何指標、不做選擇。

每張瑕疵圖重放 `500` 次。

| 物件 | 瑕疵圖 | 原圖瑕疵面積中位數 | 空 mask 比率 | 曾經整個切掉的圖 | 保留面積比（平均） |
| --- | ---: | ---: | ---: | ---: | ---: |
| pcb1 | 10 | 0.2622% | 0.00% | 0/10 | 1.208 |
| capsules | 10 | 0.1630% | 1.44% | 1/10 | 1.244 |

## 每個物件最容易被切掉的瑕疵圖

### pcb1

- `real::pcb1/Data/Images/Anomaly/004.JPG`：原圖瑕疵佔 `0.1465%`，空 mask `0.0%`，保留面積比 `1.210`
- `real::pcb1/Data/Images/Anomaly/009.JPG`：原圖瑕疵佔 `0.3175%`，空 mask `0.0%`，保留面積比 `1.198`
- `real::pcb1/Data/Images/Anomaly/010.JPG`：原圖瑕疵佔 `0.3430%`，空 mask `0.0%`，保留面積比 `1.193`
- `real::pcb1/Data/Images/Anomaly/016.JPG`：原圖瑕疵佔 `0.2358%`，空 mask `0.0%`，保留面積比 `1.200`
- `real::pcb1/Data/Images/Anomaly/037.JPG`：原圖瑕疵佔 `0.1106%`，空 mask `0.0%`，保留面積比 `1.219`

### capsules

- `real::capsules/Data/Images/Anomaly/084.JPG`：原圖瑕疵佔 `1.6081%`，空 mask `14.4%`，保留面積比 `0.105`
- `real::capsules/Data/Images/Anomaly/004.JPG`：原圖瑕疵佔 `0.0791%`，空 mask `0.0%`，保留面積比 `1.344`
- `real::capsules/Data/Images/Anomaly/009.JPG`：原圖瑕疵佔 `0.1495%`，空 mask `0.0%`，保留面積比 `1.371`
- `real::capsules/Data/Images/Anomaly/010.JPG`：原圖瑕疵佔 `0.1765%`，空 mask `0.0%`，保留面積比 `1.361`
- `real::capsules/Data/Images/Anomaly/016.JPG`：原圖瑕疵佔 `0.1290%`，空 mask `0.0%`，保留面積比 `1.369`

## 訓練期的 Dice 項有沒有啟動

BCE 在每個 run 都會下降——把整張預測成背景就能拿到低 BCE。
只有 **Dice 項**能顯示模型是否真的學會與瑕疵重疊。

| 物件 | 組別 | 首窗 dice_loss | 末窗 dice_loss | 改善量 | Dice 項啟動 | 最終 Dice |
| --- | --- | ---: | ---: | ---: | :---: | ---: |
| pcb1 | real_only | 0.9925 | 0.9650 | +0.0275 | 是 | 0.3762 |
| pcb1 | std_aug | 0.9909 | 0.9652 | +0.0256 | 是 | 0.3836 |
| capsules | real_only | 0.9956 | 0.9696 | +0.0259 | 是 | 0.5958 |
| capsules | std_aug | 0.9961 | 0.9944 | +0.0016 | **否** | 0.0000 |

