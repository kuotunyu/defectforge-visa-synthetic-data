# 零 Dice 分割 run 診斷

> 由 `scripts/diagnose_zero_dice_segmentation.py` 產生。
> 這是對**已完成**結果的事後量測：重跑推論、先與已發佈數字對帳，再回報機率分布。
> 不做任何模型、threshold 或超參選擇，也不寫入 `results/`。

預註冊 threshold：`0.5`　
全部 16 個 physical run 的重算指標都與已發佈值相符（最大差異 `1.58e-03`）。

## 全部 run

| 物件 | 組別 | Dice | pixel AUROC | 最高預測機率 | ≥ threshold 的像素 | 真實瑕疵曝光佔比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pcb1 | copypaste_only | 0.0000 | 0.9015 | 0.4530 | 0 | 1.5% |
| pcb1 | diffusion_only | 0.0000 | 0.8288 | 0.2708 | 0 | 1.5% |
| pcb1 | filtered_syn | 0.0621 | 0.9010 | 0.8872 | 45,366 | 1.5% |
| pcb1 | full_real | 0.6862 | 0.9296 | 0.9839 | 42,779 | 100.0% |
| pcb1 | procedural_only | 0.0316 | 0.8551 | 0.9182 | 31,046 | 0.0% |
| pcb1 | real_only | 0.3762 | 0.9460 | 0.9642 | 18,466 | 100.0% |
| pcb1 | std_aug | 0.3836 | 0.9144 | 0.9718 | 14,600 | 100.0% |
| pcb1 | unfiltered_syn | 0.2490 | 0.9324 | 0.7360 | 8,709 | 1.5% |
| capsules | copypaste_only | 0.6101 | 0.9869 | 0.9752 | 80,965 | 1.5% |
| capsules | diffusion_only | 0.0000 | 0.5178 | 0.3169 | 0 | 1.5% |
| capsules | filtered_syn | 0.4570 | 0.9737 | 0.9014 | 65,381 | 1.5% |
| capsules | full_real | 0.6331 | 0.9991 | 0.9949 | 107,773 | 100.0% |
| capsules | procedural_only | 0.0000 | 0.6127 | 0.3235 | 0 | 0.0% |
| capsules | real_only | 0.5958 | 0.9858 | 0.9920 | 45,752 | 100.0% |
| capsules | std_aug | 0.0000 | 0.8661 | 0.3157 | 0 | 100.0% |
| capsules | unfiltered_syn | 0.0000 | 0.4919 | 0.3014 | 0 | 1.5% |

## 最高預測機率完全決定 Dice 是否退化

- 全部 6 個零 Dice run 的最高預測機率都**低於** threshold（最大 `0.4530`）
- 全部非零 Dice run 的最高預測機率都**高於** threshold（最小 `0.7360`）

也就是說：在這批 run 上，Dice 是否為 0 完全由**機率天花板**決定，與模型排序能力（pixel AUROC）無關。

## 零 Dice run 的判定

### pcb1 / copypaste_only

- 判定：`underconfident_peak_inside_ground_truth`
- 最高預測機率 `0.4530`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.4530`，區域外 `0.2950`
- pixel AUROC `0.9015`

### pcb1 / diffusion_only

- 判定：`underconfident_peak_inside_ground_truth`
- 最高預測機率 `0.2708`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.2708`，區域外 `0.2652`
- pixel AUROC `0.8288`

### capsules / diffusion_only

- 判定：`underconfident_peak_outside_ground_truth_ranking_near_random`
- 最高預測機率 `0.3169`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.2666`，區域外 `0.3169`
- pixel AUROC `0.5178`

### capsules / procedural_only

- 判定：`underconfident_peak_outside_ground_truth_ranking_near_random`
- 最高預測機率 `0.3235`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.2305`，區域外 `0.3235`
- pixel AUROC `0.6127`

### capsules / std_aug

- 判定：`underconfident_peak_outside_ground_truth_ranking_informative`
- 最高預測機率 `0.3157`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.3047`，區域外 `0.3157`
- pixel AUROC `0.8661`

### capsules / unfiltered_syn

- 判定：`underconfident_peak_outside_ground_truth_ranking_near_random`
- 最高預測機率 `0.3014`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.2587`，區域外 `0.3014`
- pixel AUROC `0.4919`

