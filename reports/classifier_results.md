# M16 Classification Results

**Status:** `passed`

**Formal runs:** `38`

**Test source:** frozen `2cls_highshot` test only

## Seed 42

| Run | Macro-F1 | Anomaly F1 | AUROC | Normal FPR |
|---|---:|---:|---:|---:|
| m16_base_sdxl_capsules_seed_42 | 0.4025 | 0.0198 | 0.2967 | 0.2490 |
| m16_base_sdxl_pcb1_seed_42 | 0.2917 | 0.0000 | 0.1420 | 0.5473 |
| m16_bucket_original_capsules_seed_42 | 0.3643 | 0.0000 | 0.2007 | 0.3320 |
| m16_bucket_original_pcb1_seed_42 | 0.2443 | 0.0066 | 0.1377 | 0.6517 |
| m16_bucket_searched_capsules_seed_42 | 0.3671 | 0.0000 | 0.2276 | 0.3237 |
| m16_bucket_searched_pcb1_seed_42 | 0.2866 | 0.0074 | 0.1420 | 0.5672 |
| m16_filtered_syn_capsules_seed_42 | 0.3712 | 0.0444 | 0.2844 | 0.3817 |
| m16_filtered_syn_pcb1_seed_42 | 0.4270 | 0.0160 | 0.2229 | 0.2090 |
| m16_full_real_capsules_seed_42 | 0.6748 | 0.4874 | 0.8583 | 0.2075 |
| m16_full_real_pcb1_seed_42 | 0.6826 | 0.4815 | 0.9294 | 0.2065 |
| m16_procedural_norealstats_capsules_seed_42 | 0.4848 | 0.1951 | 0.5119 | 0.2946 |
| m16_procedural_norealstats_pcb1_seed_42 | 0.5994 | 0.3457 | 0.8091 | 0.2338 |
| m16_real_20_capsules_seed_42 | 0.6390 | 0.4000 | 0.8105 | 0.1494 |
| m16_real_20_pcb1_seed_42 | 0.6826 | 0.4815 | 0.9141 | 0.2065 |
| m16_real_only_capsules_seed_42 | 0.5728 | 0.2535 | 0.7934 | 0.0913 |
| m16_real_only_pcb1_seed_42 | 0.6826 | 0.4815 | 0.9086 | 0.2065 |
| m16_src_copypaste_capsules_seed_42 | 0.4859 | 0.2206 | 0.6055 | 0.3361 |
| m16_src_copypaste_pcb1_seed_42 | 0.4213 | 0.1084 | 0.3944 | 0.3781 |
| m16_src_diffusion_capsules_seed_42 | 0.3714 | 0.0000 | 0.2123 | 0.3112 |
| m16_src_diffusion_pcb1_seed_42 | 0.2113 | 0.0061 | 0.1448 | 0.7114 |
| m16_src_procedural_capsules_seed_42 | 0.4723 | 0.1681 | 0.5000 | 0.2863 |
| m16_src_procedural_pcb1_seed_42 | 0.5417 | 0.2545 | 0.7272 | 0.2587 |
| m16_std_aug_capsules_seed_42 | 0.6031 | 0.3333 | 0.7656 | 0.1452 |
| m16_std_aug_pcb1_seed_42 | 0.6826 | 0.4815 | 0.9157 | 0.2065 |
| m16_syn_125_capsules_seed_42 | 0.5021 | 0.1053 | 0.4602 | 0.0581 |
| m16_syn_125_pcb1_seed_42 | 0.6781 | 0.4756 | 0.8779 | 0.2114 |
| m16_syn_250_capsules_seed_42 | 0.4537 | 0.1562 | 0.4324 | 0.3237 |
| m16_syn_250_pcb1_seed_42 | 0.3373 | 0.0000 | 0.1336 | 0.4403 |
| m16_unfiltered_syn_capsules_seed_42 | 0.3839 | 0.0331 | 0.3145 | 0.3278 |
| m16_unfiltered_syn_pcb1_seed_42 | 0.4866 | 0.1858 | 0.5556 | 0.3134 |

## Three-seed preregistered groups

| Group / object | Macro-F1 mean ± std | AUROC mean ± std |
|---|---:|---:|
| filtered_syn/capsules | 0.3609 ± 0.0201 | 0.3243 ± 0.0426 |
| filtered_syn/pcb1 | 0.3175 ± 0.1066 | 0.1677 ± 0.0502 |
| real_only/capsules | 0.5471 ± 0.0268 | 0.8160 ± 0.0224 |
| real_only/pcb1 | 0.6808 ± 0.0031 | 0.9265 ± 0.0231 |

All run signatures, portable data manifests, trained model hashes, frozen test inventories,
CSV rows, and test-disjoint train/validation hashes were independently verified.
