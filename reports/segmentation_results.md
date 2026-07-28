# M20 segmentation results

All values below were rebuilt from returned raw `training_report.json` files. Notebook output text and the per-runtime CSV files were not used.

## capsules

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

## pcb1

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

## Two-object macro mean

| Group | Dice | mIoU | pixel AUROC | AUPRO |
|---|---:|---:|---:|---:|
| real_only | 0.4860 | 0.6637 | 0.9659 | 0.7258 |
| std_aug | 0.1918 | 0.5590 | 0.8902 | 0.5666 |
| unfiltered_syn | 0.1245 | 0.5353 | 0.7122 | 0.3866 |
| filtered_syn | 0.2596 | 0.5816 | 0.9374 | 0.8304 |
| full_real | 0.6596 | 0.7461 | 0.9643 | 0.7795 |
| procedural_only | 0.0158 | 0.5036 | 0.7339 | 0.4734 |
| copypaste_only | 0.3050 | 0.6094 | 0.9442 | 0.6913 |
| diffusion_only | 0.0000 | 0.4997 | 0.6733 | 0.3556 |
| all_mixed | 0.2596 | 0.5816 | 0.9374 | 0.8304 |

`procedural_only` means zero real defect **images/pixels** in training, while its procedural generator used aggregate real-mask statistics as preregistered in ADR-011.

`all_mixed` is the logical ninth group and cites the exact `filtered_syn` physical run.
