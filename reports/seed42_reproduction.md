# Seed 42 跨機器重現檢查

ADR-032 的複跑在另一台 Colab 機器、另一個時間重新執行了 seed 42，因為 Drive 上沒有先前的 `runs/` 樹可供跳過。這個意外是免費的證據，因此加以驗證而非丟棄。

- 比較的實跑 run 數：**16**
- `model.safetensors` SHA256 相同的 run 數：**16 / 16**
- 判定：**逐 bit 相同**

| 指標 | 最大絕對差 |
|---|---:|
| dice | `0.00000000` |
| miou | `0.00000000` |
| pixel_auroc | `0.00000000` |
| aupro | `0.00000000` |

| 物件 | 組別 | model SHA256 相同 | run signature 相同 |
|---|---|:---:|:---:|
| capsules | copypaste_only | 是 | 是 |
| pcb1 | copypaste_only | 是 | 是 |
| capsules | diffusion_only | 是 | 是 |
| pcb1 | diffusion_only | 是 | 是 |
| capsules | filtered_syn | 是 | 是 |
| pcb1 | filtered_syn | 是 | 是 |
| capsules | full_real | 是 | 是 |
| pcb1 | full_real | 是 | 是 |
| capsules | procedural_only | 是 | 是 |
| pcb1 | procedural_only | 是 | 是 |
| capsules | real_only | 是 | 是 |
| pcb1 | real_only | 是 | 是 |
| capsules | std_aug | 是 | 是 |
| pcb1 | std_aug | 是 | 是 |
| capsules | unfiltered_syn | 是 | 是 |
| pcb1 | unfiltered_syn | 是 | 是 |

基準表：`reports/segmentation_seed42_baseline.csv`（SHA256 `9ab007dd39e12a16d2236bb1ce20c4bada24185e3476b4b1eee4343f20a372d4`）

目前表：`results/segmentation.csv`（SHA256 `16ab7e5efd3fd1146bfd5d13c47fde5b3dad6fae0672a4e66a743683a20c70a9`）
