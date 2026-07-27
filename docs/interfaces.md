# Interfaces — 腳本、CLI 與設定契約

> **這份是契約，不是建議。** 無人值守執行時 agent 一律照這裡的參數名稱呼叫，
> **不准自己發明參數**。要新增或改語意，先更新這份文件並在 worklog 記一筆。
> 對應規則見 [autonomy_policy.md §4](autonomy_policy.md)。

---

## 1. 全域慣例

### 1.1 所有腳本共用的參數

| 參數 | 型別 | 預設 | 說明 |
|---|---|---|---|
| `--paths` | path | `configs/paths.yaml` | 路徑真相來源，**不得硬編絕對路徑** |
| `--config` | path | 各腳本自己的 `configs/<name>.yaml` | 該階段的參數 |
| `--object` | str（可重複） | 全部（`pcb1`, `capsules`） | 只處理指定物件 |
| `--seed` | int | `42` | 隨機種子 |
| `--device` | str | `cuda` | `cuda` / `cpu` |
| `--dry-run` | flag | off | 只印出「會做什麼」與預估輸出數量／磁碟，**不寫任何檔案** |
| `--resume` | flag | off | 掃描既有輸出，跳過已完成的項目 |
| `--log-level` | str | `INFO` | |

### 1.2 退出碼

| 碼 | 意義 | agent 應該 |
|---|---|---|
| `0` | 成功且所有內建斷言通過 | 繼續 |
| `1` | 一般錯誤 | 讀 stderr，可重試（最多 3 次） |
| `2` | **驗證／斷言失敗** | **立刻停止**，寫 handoff |
| `3` | **紅線違反**（碰到 test blocklist、split 被動過） | **立刻停止**，絕不重試 |
| `4` | 資源不足（磁碟／VRAM） | 停止，回報需求量 |

### 1.3 每支腳本都必須做的事

1. 啟動時載入 `configs/paths.yaml`，**任何路徑都從那裡來**
2. 讀任何影像前呼叫 `src/common/guard.py::assert_not_test(path)` → 命中就 `exit 3`
3. 用單一 `numpy.random.Generator(PCG64(seed))` 派生所有隨機，**不用全域 random state**
4. 結束時把「本次執行的參數 + 產出數量 + 耗時」附加到 `logs/<script>.jsonl`
5. 所有路徑用 `pathlib.Path`，**不用字串串接**（Colab 是 Linux）
6. DataLoader 預設 `num_workers=0`；要開多 worker 時進入點必須包 `if __name__ == "__main__":`

---

## 2. 共用模組（`src/common/`）

| 模組 | 提供 |
|---|---|
| `paths.py` | `load_paths(cfg="configs/paths.yaml") -> Paths`；`Paths.data_root`、`.visa_raw`、`.synthetic`、`.runs`… 已展開 `${data_root}` |
| `guard.py` | `assert_not_test(path)`；`assert_manifest_frozen()`；`assert_disk_free(gb)`；`preflight(milestone)` — 對應 [autonomy_policy.md §2](autonomy_policy.md) |
| `provenance.py` | `write_record(jsonl_path, record: dict)`；`validate_record(record) -> list[str]`（回傳缺漏欄位，空 list = 通過）。Schema 見 [synthesis_spec.md §1](synthesis_spec.md) |
| `imaging.py` | `crop_to_roi`、`blend_back`（poisson / feather）、`connected_components`、`mask_morphology_features` |
| `embed.py` | `dinov2_embed(images) -> np.ndarray`（`facebook/dinov2-base` CLS token、L2 normalized，含磁碟快取） |
| `report.py` | contact sheet / grid 產生器，統一樣式 |

---

## 3. 逐支腳本契約

### M2 `scripts/download_visa.py`
```
--paths --dry-run
--skip-download        # 只做校驗，不重新下載
```
產出：`${raw}/VisA_20220922.tar`、解壓的 `${visa_raw}/`、`splits/source_checksums.json`
斷言：tar 大小 == `1929840640`；解壓後 pcb1 = 1004/100、capsules = 602/100 → 不符 `exit 2`

### M3 `scripts/prepare_splits.py`
```
--paths --object
--split-type {2cls_highshot,2cls_fewshot,both}   # 預設 both
--spot-diff-dir PATH                             # spot-diff repo 位置（自動 clone 到 ${data_root}/tools/）
```
產出：`${visa_highshot}/`、`${visa_fewshot}/`
斷言（[ADR-007](decisions.md#adr-007)）：八格張數相符；`highshot_train ∩ highshot_test == ∅`；
`fewshot_train ⊆ highshot_train`；每張 bad 有對應 mask → 不符 `exit 2`

### M4 `scripts/freeze_manifest.py`
```
--paths --object --seed
--phash-threshold INT      # 預設 6
--force                    # 覆寫既有 manifest；無人值守時禁用
```
產出：`splits/split_manifest.json`、`splits/MANIFEST.sha256`、`splits/test_blocklist.json`、
`reports/split_report.md`
斷言：同 `group_id` 必定同 `set`；每張 test image / bad mask 的 SHA256 都在 blocklist，
且 unique 計數自洽

### M5 `scripts/sample_fewshot.py`
```
--paths --object --seed
--k INT                    # 預設 10
```
產出：`splits/fewshot_selection.json`（含 `in_fewshot_seed` / `in_val` 路徑集合與 manifest
SHA256）、`splits/FEWSHOT_SELECTION.sha256`、`reports/fewshot_stats.md`、
`reports/real_mask_stats.json`、`reports/figures/fewshot_contact_sheet_<object>.png`
斷言：重跑兩次的檔名清單雜湊相同

### M6 `scripts/cluster_defect_types.py`
```
--paths --object --seed
--k-range 1 5              # silhouette 搜尋範圍
--min-cluster-size INT     # 預設 3
--min-component-area INT   # 雜點過濾，預設 32 (px)
--auto-name                # 無人值守用：產生 <obj-typeN> 暫用 token 直接凍結，不等人
```
產出：`splits/defect_types.json`（+ SHA256）、`reports/figures/defect_type_cluster_<object>_<k>.png`
斷言：每群 ≥ `--min-cluster-size`（否則自動 fallback 合併）；所有輸入路徑不在 blocklist

### M7 `src/synthetic/copy_paste.py`
```
--paths --config configs/stage_a.yaml --object --seed --resume --dry-run
--n INT                    # 每物件張數，預設 500
--blend {poisson,feather,mixed}   # 預設 mixed
--out-name STR             # 預設 stageA_copypaste
```
產出：`${synthetic}/<out-name>/{images,masks}/`、`metadata.jsonl`

### M8 `src/synthetic/procedural.py`
```
（同 M7 的共用參數）
--n INT                    # 預設 500
--shapes perlin,crack,scratch,spot     # 預設全開
--no-real-stats            # 不讀 real_mask_stats.json，用手訂固定分布（ADR-011）
--out-name STR             # 預設 stageA_procedural；--no-real-stats 時預設 stageA_procedural_norealstats
```
斷言：`--no-real-stats` 時執行過程中**從未開啟** `real_mask_stats.json`（用檔案存取記錄驗證）

### Stage A independent validation `scripts/validate_synthetic.py`
```
--paths --config configs/stage_a.yaml
--out-name STR             # default: stageA_copypaste
--n INT                    # expected samples per object, default: 500
--report PATH              # optional JSON validation summary
```
Reopens every image/mask and checks the exact inventory, metadata schema, frozen
train-good backgrounds, frozen defect components, SHA256 test blocklist, binary
mask/size/bbox/area, then rebuilds each legal ROI from the source background and
asserts that every mask is 100% contained.

### M8 independent validation `scripts/validate_procedural.py`
```
--paths --config configs/stage_a.yaml --n INT --shapes CSV
--out-name STR --no-real-stats --report PATH
```
Rebuilds every ROI, verifies frozen train-good provenance and zero real-defect
source fields, checks exact balanced shape quotas and blocklist exclusion, then
requires area/aspect outlier rates below 10%. In `--no-real-stats` mode a Python
audit hook makes opening `real_mask_stats.json` a fatal error.

### M9 `src/synthetic/mask_placement.py`
```
--paths --config configs/placement.yaml --object --seed --resume --dry-run
--n-per-image INT          # 每張正常圖產生幾組配對，預設 3
--roi-method {otsu,dinov2,intersect}    # 預設 intersect
--max-place-tries INT      # 預設 50
--viz-n INT                # 產生幾張視覺化檢查圖，預設 24
```
產出：`${synthetic}/placements/<object>/placements.jsonl`、
`reports/figures/placement_check_<object>.png`
斷言：放置 mask 100% 在 legal ROI 內；不與其他 mask 重疊；面積在真實 5–95 百分位內

### M10 / M11 `src/training/train_inpaint_lora.py`
**本機與 Colab notebook 都呼叫這一支**（[ADR-008](decisions.md#adr-008)），notebook 不得複製訓練迴圈。
```
--paths --config configs/lora_sd2.yaml | configs/lora_sdxl.yaml --object --seed
--base-model STR           # 由 config 指定，CLI 可覆寫
--resolution INT           # SD2 512 / SDXL 1024
--rank INT --alpha INT
--max-train-steps INT
--lr FLOAT
--output-dir PATH          # 預設 ${runs}/lora_<model>/<object>/seed_<seed>
--resume-from-checkpoint {latest,PATH}
--sample-every INT         # 每 N steps 產一張 held-out 樣本圖到 output-dir/samples/
--stop-after-steps INT     # 受控中斷；只供 checkpoint/resume 驗證，不進 run signature
--smoke                    # 1 step、極小 batch，只驗證流程能跑通並存權重
--drive-sync PATH          # Colab 專用：checkpoint 定期同步目的地
--dry-run                  # 只驗 frozen inputs、模型鎖與 run signature，不載權重
```
斷言：存出的權重能被 `PeftModel` 載回；`--smoke` 模式下不覆寫正式 checkpoint；
每個 sample 同時寫 prompt／token／placement／seed／SHA256 sidecar。
`family=sdxl` 時另要求 `tokenizer_2`／`text_encoder_2`，checkpoint 必含
`text_token_adapter_2/` 與 `tokenizer_2/`；UNet forward 必帶 pooled text embedding
與 time IDs。`family=sd2` 的 bundle schema 與 pipeline version 保持不變。

正式 M10 產物重驗：
```text
scripts/validate_lora_run.py
--paths --config --run-root PATH --reload --output PATH
```
斷言：兩物件 checkpoint inventory、PEFT config、adapter hash、兩型 sample 輪替、
背景 blocklist、凍結 checksum 與 fresh `PeftModel` reload 全部通過。

M11 Colab 離線交接：
```text
scripts/validate_colab_notebook.py
scripts/package_m11_colab.py --paths --output-dir PATH
scripts/verify_colab_lora_results.py
  --paths --config --results-root PATH --archive PATH --output PATH
scripts/verify_local_sdxl_checks.py
  --paths --config --cache-root PATH --smoke-root PATH --resume-root PATH --output PATH
```
前者驗五節、resume 分支、Secrets 與「notebook 不得複製訓練迴圈」；後者只封裝
tracked source、20 張 frozen few-shot image/mask 與每型一個 held-out placement，
產兩個 zip 與 SHA256 manifest。第三支對下載 ZIP 與 `results/colab/lora_sdxl/`
做 CPU-only 重驗：CRC／安全 member、凍結 hashes、final adapter/tokenizer inventory、
sample sidecar／panel／background hashes、blocklist 與 Colab validator 證據必須一致；
不載入 SDXL base weights，也不得把它當成未實跑 resume 分支的替代證據。第四支重算
本機四份 locked base-weight SHA256，並驗兩物件 one-step smoke、step 1 → 2 resume
progression、run signature、checkpoint/final bundles、雙 tokenizer、sample 與 blocklist；
同樣不載入模型或配置 GPU。

### M12 `src/synthetic/generate_diffusion.py`
```
--paths --config configs/generate_sd2.yaml | configs/generate_sdxl.yaml --object --seed --resume --dry-run --device
--lora PATH                # 訓練好的 adapter
--defect-type STR          # 可重複；預設全部
--n INT                    # 每物件總量：SD2 500 / SDXL 250（ADR-010）
--bucket {original,searched}          # 預設 original
--guidance-scale FLOAT --num-inference-steps INT --crop-ratio FLOAT
--refine                   # 開啟 refine 搜尋，輸出到 searched/
--num-search-run INT       # 預設 4
--guidance-grid 5.0,7.5,10.0,12.5
--crop-ratio-grid 1.8,2.5,3.5
--out-name STR             # 預設 stageB_sd2 / stageB_sdxl
--contact-sheet PATH       # clean crop / binary mask / generated crop 的正式目視證據
```
產出：`${synthetic}/<out-name>/{original,searched}/{images,masks,.records}/` +
`metadata.jsonl`。每筆 atomic sidecar 鎖 config SHA、pipeline version、model revision、
placement checksum、effective parameters、候選分數與 image/mask SHA256；canonical metadata
由 sidecar 重建。

斷言：每行 metadata 通過 `provenance.validate_record`；同 seed 的獨立行程 PNG 位元相同；
完整 `--resume` 不載模型且峰值 VRAM 為 0；輸出 crop 外像素完全等於 frozen background。
`original` pipeline v0.5.0；`searched` v0.6.0 的 candidate 0 必須逐欄重現 original
的 guidance、crop、seed 與評分證據，另外三候選覆蓋剩餘搜尋網格，因此 selected score
必須大於或等於 original baseline。

正式獨立重驗：
```text
scripts/validate_diffusion.py
--paths --config --out-name --bucket {original,searched} --n --object --output
```

正式同 ID 比較圖：
```text
scripts/compare_diffusion.py
--paths --out-name --object --maximum --output
```
四欄固定為 clean / mask / original / searched，兩桶使用 union crop 的同一視野；
抽樣在 defect type 間 round-robin，且背景 SHA 與兩桶 mask SHA 不同即失敗。

### M13 `scripts/filter_synthetic.py`
```
--paths --config configs/filters.yaml
--stage-a-config configs/stage_a.yaml
--placement-config configs/placement.yaml
--real-mask-stats reports/real_mask_stats.json
--object NAME              # 可重複；預設處理兩個 formal objects
--disable RULE             # 可重複：roi|area|aspect|phash|dinov2|seam（做「哪道規則貢獻最大」分析用）
--limit-per-input INT      # smoke only
--publish                  # 未指定時只評分，不建立 hardlink views
--validation-out PATH
```
正式輸入由 `configs/filters.yaml` 的 `inputs` 鎖定。產出：
`${synthetic}/filtered/`、`${synthetic}/unfiltered/`（皆含 `metadata.jsonl`，
`unfiltered` 保留所有樣本並標 `passed` 與 `reject_reasons`）、`reports/filter_report.md`

### M14 `scripts/evaluate_generation_quality.py`
```
--paths --config configs/quality.yaml
--filter-config configs/filters.yaml
--defect-types splits/defect_types.json
--blocklist splits/test_blocklist.json
--filter-validation reports/filter_validation.json
--prepare-only             # CPU-only source audit + immutable crop cache，不載模型
--sanity-check             # 跑必做 sanity gate，通過後停止，不發布正式報表
```
模型載入前另依 `configs/quality.yaml` 檢查 existing shared VRAM；超過上限以 exit 4
停止，避免與同機其他專案爭用 GPU。
產出：`reports/generation_quality.md`、`results/generation_quality.csv`
斷言：`--sanity-check` 下真實 crop 的 `nn_score ≈ 1`、biased polynomial MMD = 0；
正式表仍用未偏 KID。不成立 `exit 2`

### M16 `src/training/train_classifier.py`
```
--paths --config configs/classifier.yaml --object --seed
--group STR                # real_only | std_aug | unfiltered_syn | filtered_syn | full_real
                           # | real_20 | real_60 | syn_125 | syn_250 | syn_500
                           # | src_procedural | src_copypaste | src_diffusion
                           # | base_sd2 | base_sdxl | procedural_norealstats
                           # | bucket_original | bucket_searched
--run-name STR             # 預設由 group+object+seed 組出
--mode {development,final} # development 只允許 real_only 且不載 test
--total-steps INT          # 固定 total steps 而非 epochs（公平性規則）
--learning-rate FLOAT --weight-decay FLOAT  # development 候選；formal 取 frozen config
--output-dir PATH
--smoke                    # 1 step 真模型 smoke，不寫 classification.csv
--dry-run                  # 完整資料展開與防洩漏檢查，不載模型
```
產出：`${runs}/cls/<run-name>/`、附加一列到 `results/classification.csv`
斷言：`df-guard` 的防洩漏檢查表全綠；development 的 test inventory 必須為空；
formal 訓練集 ∩ highshot test == ∅。每個 run 保存 portable `data_manifest.json`、
`run_config.json`、`training_report.json` 與 model safetensors；CSV 每個 run name 唯一。
alias `real_60` / `syn_500` / `base_sd2` 只引用 canonical run，不重跑。

正式 38-run 矩陣由可恢復 runner 啟動：
```text
scripts/run_classifier_matrix.py
  --paths configs/paths.yaml
  --config configs/classifier.yaml
  [--dry-run]
```
runner 在配置 GPU 前先展開並防洩漏檢查全部 38 組。每個 run 先寫入
`${runs}/cls/.<run-name>.working`，通過 report 與 `classification.csv` 雙重證據後才
原子改名；中斷的 working directory 下次移到 `${runs}/cls/_incomplete/` 保留，不刪除。
已完整驗證的 run 會跳過，alias 不會進入計畫。

兩物件 smoke 的 CPU-only 重驗：
```text
scripts/verify_classifier_smoke.py
  --paths configs/paths.yaml
  --config configs/classifier.yaml
  --output reports/classifier_smoke_validation.json
```
驗 model/revision/base hash、1 step、portable data manifest hash、validation real-only、
test inventory 為空、sample exposure、peak VRAM 與輸出 model hash；不載模型或配置 GPU。

Real-only tuning 的 CPU-only 凍結驗證：
```text
scripts/verify_classifier_tuning.py
  --paths configs/paths.yaml
  --config configs/classifier.yaml
  --output reports/classifier_tuning.json
  --report reports/classifier_tuning.md
```
驗三個 learning-rate 候選 × 兩物件的 raw report、development test inventory 皆為空、
validation 皆為真實資料、model lock 與固定搜尋預算；以兩物件 mean Macro-F1、mean AUROC、
較低 learning rate 依序選擇，並要求 config 的 frozen setting 完全相符。

### M15 `scripts/verify_phase1.py`
```text
--project-root PATH
--output reports/phase1_acceptance.json
--report reports/phase1_acceptance.md
```
CPU-only 驗收 M0–M15 的 PLAN 狀態、里程碑 commit 覆蓋、凍結 evidence、M11 五項
Colab 交接與三支獨立 validator。另逐 commit 檢查 author／committer 必須都是
`kuotunyu <61350295+kuotunyu@users.noreply.github.com>`，且歷史中不得出現
`Co-Authored-By` trailer。SegFormer 的五項具體交接依 ADR-019 由 M18 負責。

### M18 `src/training/train_segmenter.py`
```
（同 M16 的共用參數）
--group STR                # real_only | std_aug | unfiltered_syn | filtered_syn | full_real
                           # | procedural_only | copypaste_only | diffusion_only
--drive-sync PATH          # Colab 專用
--smoke
```
產出：`${runs}/seg/<run-name>/`、`results/segmentation.csv`
註：第 9 組「全部混合」**＝ `filtered_syn`，直接引用不重跑**

### M22 `src/inference/demo_gradio.py`
```
--paths --cls-ckpt PATH --seg-ckpt PATH --port INT --share   # --share 預設 off，不對外開放
```

### 驗證用腳本
| 腳本 | 用途 |
|---|---|
| `scripts/verify_filter_report.py` | 從 `metadata.jsonl` 重算漏斗表，與 `reports/filter_report.md` 逐格比對 |
| `scripts/verify_generation_quality.py` | 核對 M14 CSV／Markdown／validation、sanity gate、圖與 feature-cache SHA256 |
| `scripts/verify_readme.py` | 從 `results/*.csv` 重算 README 每張表的數字並比對 |
| `scripts/verify_splits.py` | 重跑 [ADR-007](decisions.md#adr-007) 的四項斷言 |
| `scripts/upload_hf.py` | 見 [publish_spec.md](publish_spec.md)；**預設 `--dry-run`，上傳要顯式加 `--confirm`** |

---

## 4. 設定檔

全部放 `configs/`，YAML，**參數的預設值寫在 config 而不是程式碼裡**，
CLI 只用於覆寫。每支腳本執行時把「合併後的最終 config」存進該次 run 的輸出目錄，確保可重現。

| 檔案 | 給誰 |
|---|---|
| `paths.yaml` | 全部（唯一路徑真相來源） |
| `stage_a.yaml` | copy_paste / procedural |
| `placement.yaml` | mask_placement |
| `lora_sd2.yaml` / `lora_sdxl.yaml` | train_inpaint_lora |
| `generate_sd2.yaml` / `generate_sdxl.yaml` | generate_diffusion |
| `filters.yaml` | run_filters |
| `classifier.yaml` / `segmenter.yaml` | 下游訓練 |
