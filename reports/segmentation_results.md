# M20 segmentation results

All values below were rebuilt from returned raw `training_report.json` files. Notebook output text and the per-runtime CSV files were not used.

Seeds: 42, 43, 44. Seed 42 is the pre-registered anchor; ADR-032 added the remaining seeds to all eight formal groups so that no group lacks error bars.

## capsules

### Seed 42 (pre-registered anchor)

| Group | Physical run | Dice | mIoU | pixel AUROC | AUPRO |
|---|---|---:|---:|---:|---:|
| real_only | yes | 0.5958 | 0.7119 | 0.9858 | 0.8488 |
| std_aug | yes | 0.0000 | 0.4996 | 0.8661 | 0.5591 |
| unfiltered_syn | yes | 0.0000 | 0.4996 | 0.4919 | 0.1666 |
| filtered_syn | yes | 0.4570 | 0.6477 | 0.9737 | 0.9137 |
| full_real | yes | 0.6331 | 0.7312 | 0.9991 | 0.9591 |
| procedural_only | yes | 0.0000 | 0.4996 | 0.6127 | 0.3506 |
| copypaste_only | yes | 0.6101 | 0.7191 | 0.9869 | 0.9440 |
| diffusion_only | yes | 0.0000 | 0.4996 | 0.5178 | 0.2556 |
| all_mixed | no; cites filtered_syn | 0.4570 | 0.6477 | 0.9737 | 0.9137 |

### Mean ± std across 3 seeds

| Group | Seeds | Dice | mIoU | pixel AUROC | AUPRO |
|---|---:|---:|---:|---:|---:|
| real_only | 3 | 0.5253 ± 0.0693 | 0.6788 ± 0.0320 | 0.9631 ± 0.0218 | 0.7965 ± 0.0520 |
| std_aug | 3 | 0.1404 ± 0.2432 | 0.5441 ± 0.0771 | 0.8985 ± 0.0753 | 0.6509 ± 0.1413 |
| unfiltered_syn | 3 | 0.0000 ± 0.0000 | 0.4996 ± 0.0000 | 0.5348 ± 0.0410 | 0.2405 ± 0.1169 |
| filtered_syn | 3 | 0.1523 ± 0.2639 | 0.5490 ± 0.0855 | 0.6684 ± 0.2659 | 0.4751 ± 0.3834 |
| full_real | 3 | 0.6722 ± 0.0389 | 0.7532 ± 0.0222 | 0.9978 ± 0.0012 | 0.9417 ± 0.0252 |
| procedural_only | 3 | 0.0000 ± 0.0000 | 0.4996 ± 0.0000 | 0.5713 ± 0.0498 | 0.2685 ± 0.1015 |
| copypaste_only | 3 | 0.3895 ± 0.3383 | 0.6374 ± 0.1200 | 0.9610 ± 0.0484 | 0.9124 ± 0.0463 |
| diffusion_only | 3 | 0.0000 ± 0.0000 | 0.4996 ± 0.0000 | 0.5323 ± 0.0390 | 0.2569 ± 0.0358 |
| all_mixed | 3 | 0.1523 ± 0.2639 | 0.5490 ± 0.0855 | 0.6684 ± 0.2659 | 0.4751 ± 0.3834 |

## pcb1

### Seed 42 (pre-registered anchor)

| Group | Physical run | Dice | mIoU | pixel AUROC | AUPRO |
|---|---|---:|---:|---:|---:|
| real_only | yes | 0.3762 | 0.6156 | 0.9460 | 0.6028 |
| std_aug | yes | 0.3836 | 0.6185 | 0.9144 | 0.5740 |
| unfiltered_syn | yes | 0.2490 | 0.5709 | 0.9324 | 0.6065 |
| filtered_syn | yes | 0.0621 | 0.5156 | 0.9010 | 0.7471 |
| full_real | yes | 0.6862 | 0.7610 | 0.9296 | 0.5999 |
| procedural_only | yes | 0.0316 | 0.5076 | 0.8551 | 0.5963 |
| copypaste_only | yes | 0.0000 | 0.4997 | 0.9015 | 0.4386 |
| diffusion_only | yes | 0.0000 | 0.4997 | 0.8288 | 0.4556 |
| all_mixed | no; cites filtered_syn | 0.0621 | 0.5156 | 0.9010 | 0.7471 |

### Mean ± std across 3 seeds

| Group | Seeds | Dice | mIoU | pixel AUROC | AUPRO |
|---|---:|---:|---:|---:|---:|
| real_only | 3 | 0.3300 ± 0.0489 | 0.5989 ± 0.0175 | 0.9258 ± 0.0224 | 0.5834 ± 0.0168 |
| std_aug | 3 | 0.4103 ± 0.0789 | 0.6299 ± 0.0320 | 0.9258 ± 0.0121 | 0.6067 ± 0.0308 |
| unfiltered_syn | 3 | 0.0830 ± 0.1438 | 0.5235 ± 0.0411 | 0.9080 ± 0.0237 | 0.5783 ± 0.0508 |
| filtered_syn | 3 | 0.0438 ± 0.0381 | 0.5110 ± 0.0098 | 0.8938 ± 0.0084 | 0.6600 ± 0.0993 |
| full_real | 3 | 0.6754 ± 0.0193 | 0.7549 ± 0.0109 | 0.9461 ± 0.0237 | 0.7022 ± 0.1439 |
| procedural_only | 3 | 0.1543 ± 0.1190 | 0.5430 ± 0.0349 | 0.8367 ± 0.0408 | 0.5790 ± 0.0462 |
| copypaste_only | 3 | 0.0000 ± 0.0000 | 0.4997 ± 0.0000 | 0.8993 ± 0.0072 | 0.4906 ± 0.0550 |
| diffusion_only | 3 | 0.0000 ± 0.0000 | 0.4997 ± 0.0000 | 0.8274 ± 0.0258 | 0.4711 ± 0.0829 |
| all_mixed | 3 | 0.0438 ± 0.0381 | 0.5110 ± 0.0098 | 0.8938 ± 0.0084 | 0.6600 ± 0.0993 |

## Two-object macro mean

Each cell first averages the seeds within an object, then averages the two objects; the spread is the std of the per-seed macro means.

| Group | Dice | mIoU | pixel AUROC | AUPRO |
|---|---:|---:|---:|---:|
| real_only | 0.4277 ± 0.0590 | 0.6389 ± 0.0247 | 0.9445 ± 0.0187 | 0.6900 ± 0.0337 |
| std_aug | 0.2753 ± 0.1603 | 0.5870 ± 0.0543 | 0.9121 ± 0.0429 | 0.6288 ± 0.0752 |
| unfiltered_syn | 0.0415 ± 0.0719 | 0.5115 ± 0.0205 | 0.7214 ± 0.0087 | 0.4094 ± 0.0332 |
| filtered_syn | 0.0981 ± 0.1409 | 0.5300 ± 0.0450 | 0.7811 ± 0.1363 | 0.5675 ± 0.2277 |
| full_real | 0.6738 ± 0.0218 | 0.7540 ± 0.0125 | 0.9720 ± 0.0117 | 0.8220 ± 0.0762 |
| procedural_only | 0.0771 ± 0.0595 | 0.5213 ± 0.0175 | 0.7040 ± 0.0260 | 0.4238 ± 0.0454 |
| copypaste_only | 0.1948 ± 0.1692 | 0.5686 ± 0.0600 | 0.9302 ± 0.0277 | 0.7015 ± 0.0356 |
| diffusion_only | 0.0000 ± 0.0000 | 0.4997 ± 0.0000 | 0.6798 ± 0.0080 | 0.3640 ± 0.0241 |
| all_mixed | 0.0981 ± 0.1409 | 0.5300 ± 0.0450 | 0.7811 ± 0.1363 | 0.5675 ± 0.2277 |

`procedural_only` means zero real defect **images/pixels** in training, while its procedural generator used aggregate real-mask statistics as preregistered in ADR-011.

`all_mixed` is the logical ninth group and cites the exact `filtered_syn` physical run.
