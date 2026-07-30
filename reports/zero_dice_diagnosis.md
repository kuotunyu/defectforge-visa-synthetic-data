# 零 Dice 分割 run 診斷

> 由 `scripts/diagnose_zero_dice_segmentation.py` 產生。
> 這是對**已完成**結果的事後量測：重跑推論、先與已發佈數字對帳，再回報機率分布。
> 不做任何模型、threshold 或超參選擇，也不寫入 `results/`。

預註冊 threshold：`0.5`　
全部 48 個 physical run 的重算指標都與已發佈值相符（最大差異 `4.33e-03`）。

## 全部 run

| 物件 | 組別 | Seed | Dice | pixel AUROC | 最高預測機率 | ≥ threshold 的像素 | 真實瑕疵曝光佔比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pcb1 | copypaste_only | 42 | 0.0000 | 0.9015 | 0.4530 | 0 | 1.5% |
| pcb1 | copypaste_only | 43 | 0.0000 | 0.8913 | 0.3694 | 0 | 1.1% |
| pcb1 | copypaste_only | 44 | 0.0000 | 0.9052 | 0.3616 | 0 | 1.8% |
| pcb1 | diffusion_only | 42 | 0.0000 | 0.8288 | 0.2708 | 0 | 1.5% |
| pcb1 | diffusion_only | 43 | 0.0000 | 0.8009 | 0.4328 | 0 | 1.1% |
| pcb1 | diffusion_only | 44 | 0.0000 | 0.8524 | 0.2846 | 0 | 1.8% |
| pcb1 | filtered_syn | 42 | 0.0621 | 0.9010 | 0.8872 | 45,366 | 1.5% |
| pcb1 | filtered_syn | 43 | 0.0000 | 0.8959 | 0.8187 | 8,660 | 1.1% |
| pcb1 | filtered_syn | 44 | 0.0693 | 0.8846 | 0.7919 | 10,818 | 1.8% |
| pcb1 | full_real | 42 | 0.6862 | 0.9296 | 0.9839 | 42,779 | 100.0% |
| pcb1 | full_real | 43 | 0.6868 | 0.9733 | 0.9820 | 57,471 | 100.0% |
| pcb1 | full_real | 44 | 0.6530 | 0.9355 | 0.9899 | 36,106 | 100.0% |
| pcb1 | procedural_only | 42 | 0.0316 | 0.8551 | 0.9182 | 31,046 | 0.0% |
| pcb1 | procedural_only | 43 | 0.1620 | 0.7900 | 0.8563 | 27,010 | 0.0% |
| pcb1 | procedural_only | 44 | 0.2692 | 0.8650 | 0.9431 | 36,705 | 0.0% |
| pcb1 | real_only | 42 | 0.3762 | 0.9460 | 0.9642 | 18,466 | 100.0% |
| pcb1 | real_only | 43 | 0.2788 | 0.9017 | 0.9428 | 16,198 | 100.0% |
| pcb1 | real_only | 44 | 0.3350 | 0.9296 | 0.9681 | 13,582 | 100.0% |
| pcb1 | std_aug | 42 | 0.3836 | 0.9144 | 0.9718 | 14,600 | 100.0% |
| pcb1 | std_aug | 43 | 0.4991 | 0.9385 | 0.9733 | 28,006 | 100.0% |
| pcb1 | std_aug | 44 | 0.3482 | 0.9247 | 0.9806 | 13,651 | 100.0% |
| pcb1 | unfiltered_syn | 42 | 0.2490 | 0.9324 | 0.7360 | 8,709 | 1.5% |
| pcb1 | unfiltered_syn | 43 | 0.0000 | 0.8851 | 0.4271 | 0 | 1.1% |
| pcb1 | unfiltered_syn | 44 | 0.0000 | 0.9066 | 0.3199 | 0 | 1.8% |
| capsules | copypaste_only | 42 | 0.6101 | 0.9869 | 0.9752 | 80,965 | 1.5% |
| capsules | copypaste_only | 43 | 0.0000 | 0.9051 | 0.4385 | 0 | 1.1% |
| capsules | copypaste_only | 44 | 0.5586 | 0.9910 | 0.8927 | 53,900 | 1.8% |
| capsules | diffusion_only | 42 | 0.0000 | 0.5178 | 0.3169 | 0 | 1.5% |
| capsules | diffusion_only | 43 | 0.0000 | 0.5765 | 0.2980 | 0 | 1.1% |
| capsules | diffusion_only | 44 | 0.0000 | 0.5025 | 0.2930 | 0 | 1.8% |
| capsules | filtered_syn | 42 | 0.4570 | 0.9737 | 0.9014 | 65,381 | 1.5% |
| capsules | filtered_syn | 43 | 0.0000 | 0.5434 | 0.2402 | 0 | 1.1% |
| capsules | filtered_syn | 44 | 0.0000 | 0.4881 | 0.3244 | 0 | 1.8% |
| capsules | full_real | 42 | 0.6331 | 0.9991 | 0.9949 | 107,773 | 100.0% |
| capsules | full_real | 43 | 0.7109 | 0.9974 | 0.9788 | 84,567 | 100.0% |
| capsules | full_real | 44 | 0.6725 | 0.9968 | 0.9882 | 94,071 | 100.0% |
| capsules | procedural_only | 42 | 0.0000 | 0.6127 | 0.3235 | 0 | 0.0% |
| capsules | procedural_only | 43 | 0.0000 | 0.5850 | 0.2733 | 0 | 0.0% |
| capsules | procedural_only | 44 | 0.0000 | 0.5161 | 0.2828 | 0 | 0.0% |
| capsules | real_only | 42 | 0.5958 | 0.9858 | 0.9920 | 45,752 | 100.0% |
| capsules | real_only | 43 | 0.4573 | 0.9614 | 0.9896 | 30,506 | 100.0% |
| capsules | real_only | 44 | 0.5230 | 0.9423 | 0.9657 | 37,569 | 100.0% |
| capsules | std_aug | 42 | 0.0000 | 0.8661 | 0.3157 | 0 | 100.0% |
| capsules | std_aug | 43 | 0.4212 | 0.9846 | 0.8781 | 21,460 | 100.0% |
| capsules | std_aug | 44 | 0.0000 | 0.8447 | 0.2649 | 0 | 100.0% |
| capsules | unfiltered_syn | 42 | 0.0000 | 0.4919 | 0.3014 | 0 | 1.5% |
| capsules | unfiltered_syn | 43 | 0.0000 | 0.5736 | 0.2434 | 0 | 1.1% |
| capsules | unfiltered_syn | 44 | 0.0000 | 0.5389 | 0.2901 | 0 | 1.8% |

## 零 Dice 有兩種成因，機率天花板只解釋其中一種

- 22 / 23 個零 Dice run 連一個正像素都沒有：最高預測機率**低於** threshold（其中最高的是 `0.4530`），空 Mask 是算術上必然
- 另外 1 個 run **有**正像素，但**完全沒有落在真實瑕疵上**，因此 Dice 仍為 0。這類與機率天花板無關：
  - `pcb1 / filtered_syn`（seed 43）：最高機率 `0.8187`、8,660 個正像素、pixel AUROC `0.8959`
- 全部非零 Dice run 的最高預測機率都**高於** threshold（最小 `0.7360`）

也就是說：**機率天花板是主要但非唯一成因**。絕大多數零 Dice run 確實是整張圖沒有任何像素越過 threshold，但上列 run 證明「有正像素卻完全打偏」同樣會產生零 Dice。因此不能宣稱零 Dice 一律與模型的空間定位能力無關。

## 零 Dice run 的判定

### pcb1 / copypaste_only / seed 42

- 判定：`underconfident_peak_inside_ground_truth`
- 最高預測機率 `0.4530`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.4530`，區域外 `0.2950`
- pixel AUROC `0.9015`

### pcb1 / copypaste_only / seed 43

- 判定：`underconfident_peak_inside_ground_truth`
- 最高預測機率 `0.3694`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.3694`，區域外 `0.3357`
- pixel AUROC `0.8913`

### pcb1 / copypaste_only / seed 44

- 判定：`underconfident_peak_inside_ground_truth`
- 最高預測機率 `0.3616`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.3616`，區域外 `0.2674`
- pixel AUROC `0.9052`

### pcb1 / diffusion_only / seed 42

- 判定：`underconfident_peak_inside_ground_truth`
- 最高預測機率 `0.2708`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.2708`，區域外 `0.2652`
- pixel AUROC `0.8288`

### pcb1 / diffusion_only / seed 43

- 判定：`underconfident_peak_outside_ground_truth_ranking_informative`
- 最高預測機率 `0.4328`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.1894`，區域外 `0.4328`
- pixel AUROC `0.8009`

### pcb1 / diffusion_only / seed 44

- 判定：`underconfident_peak_outside_ground_truth_ranking_informative`
- 最高預測機率 `0.2846`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.2050`，區域外 `0.2846`
- pixel AUROC `0.8524`

### pcb1 / filtered_syn / seed 43

- 判定：`positive_pixels_never_overlap_ground_truth`
- 最高預測機率 `0.8187`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.4988`，區域外 `0.8187`
- pixel AUROC `0.8959`

### pcb1 / unfiltered_syn / seed 43

- 判定：`underconfident_peak_outside_ground_truth_ranking_informative`
- 最高預測機率 `0.4271`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.3478`，區域外 `0.4271`
- pixel AUROC `0.8851`

### pcb1 / unfiltered_syn / seed 44

- 判定：`underconfident_peak_inside_ground_truth`
- 最高預測機率 `0.3199`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.3199`，區域外 `0.2652`
- pixel AUROC `0.9066`

### capsules / copypaste_only / seed 43

- 判定：`underconfident_peak_outside_ground_truth_ranking_informative`
- 最高預測機率 `0.4385`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.4380`，區域外 `0.4385`
- pixel AUROC `0.9051`

### capsules / diffusion_only / seed 42

- 判定：`underconfident_peak_outside_ground_truth_ranking_near_random`
- 最高預測機率 `0.3169`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.2666`，區域外 `0.3169`
- pixel AUROC `0.5178`

### capsules / diffusion_only / seed 43

- 判定：`underconfident_peak_outside_ground_truth_ranking_near_random`
- 最高預測機率 `0.2980`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.1814`，區域外 `0.2980`
- pixel AUROC `0.5765`

### capsules / diffusion_only / seed 44

- 判定：`underconfident_peak_outside_ground_truth_ranking_near_random`
- 最高預測機率 `0.2930`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.2106`，區域外 `0.2930`
- pixel AUROC `0.5025`

### capsules / filtered_syn / seed 43

- 判定：`underconfident_peak_outside_ground_truth_ranking_near_random`
- 最高預測機率 `0.2402`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.1665`，區域外 `0.2402`
- pixel AUROC `0.5434`

### capsules / filtered_syn / seed 44

- 判定：`underconfident_peak_outside_ground_truth_ranking_near_random`
- 最高預測機率 `0.3244`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.2218`，區域外 `0.3244`
- pixel AUROC `0.4881`

### capsules / procedural_only / seed 42

- 判定：`underconfident_peak_outside_ground_truth_ranking_near_random`
- 最高預測機率 `0.3235`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.2305`，區域外 `0.3235`
- pixel AUROC `0.6127`

### capsules / procedural_only / seed 43

- 判定：`underconfident_peak_outside_ground_truth_ranking_near_random`
- 最高預測機率 `0.2733`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.1823`，區域外 `0.2733`
- pixel AUROC `0.5850`

### capsules / procedural_only / seed 44

- 判定：`underconfident_peak_outside_ground_truth_ranking_near_random`
- 最高預測機率 `0.2828`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.2142`，區域外 `0.2828`
- pixel AUROC `0.5161`

### capsules / std_aug / seed 42

- 判定：`underconfident_peak_outside_ground_truth_ranking_informative`
- 最高預測機率 `0.3157`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.3047`，區域外 `0.3157`
- pixel AUROC `0.8661`

### capsules / std_aug / seed 44

- 判定：`underconfident_peak_outside_ground_truth_ranking_informative`
- 最高預測機率 `0.2649`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.2502`，區域外 `0.2649`
- pixel AUROC `0.8447`

### capsules / unfiltered_syn / seed 42

- 判定：`underconfident_peak_outside_ground_truth_ranking_near_random`
- 最高預測機率 `0.3014`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.2587`，區域外 `0.3014`
- pixel AUROC `0.4919`

### capsules / unfiltered_syn / seed 43

- 判定：`underconfident_peak_outside_ground_truth_ranking_near_random`
- 最高預測機率 `0.2434`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.1911`，區域外 `0.2434`
- pixel AUROC `0.5736`

### capsules / unfiltered_syn / seed 44

- 判定：`underconfident_peak_outside_ground_truth_ranking_near_random`
- 最高預測機率 `0.2901`（threshold `0.5`）
- 真實瑕疵區域內的最高機率 `0.2049`，區域外 `0.2901`
- pixel AUROC `0.5389`

