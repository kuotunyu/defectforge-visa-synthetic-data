# Interfaces — 腳本、CLI 與設定契約

> **這份是契約，不是建議。** 無人值守執行時 agent 一律照這裡的參數名稱呼叫，
> **不准自己發明參數**。要新增或改語意，先更新這份文件並在 worklog 記一筆。
> 對應決策見 [ADR-012](decisions.md#adr-012)。

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
| `guard.py` | `assert_not_test(path)`；`assert_manifest_frozen()`；`assert_disk_free(gb)`；`preflight(milestone)` — 對應 [ADR-012](decisions.md#adr-012) |
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

正式矩陣的 CPU-only 獨立驗證：
```text
scripts/verify_classifier_matrix.py
  --paths configs/paths.yaml
  --config configs/classifier.yaml
  --output reports/classifier_matrix_validation.json
  --report reports/classifier_results.md
```
要求 38 個唯一 run／signature 與 CSV 精確對應，逐 run 重算 portable data manifest、
run signature、model hash、凍結 test inventory 與 train/validation 對 test 的 SHA
不相交；另驗證三個 alias 沒有被重跑，並彙整事前指定四組的三-seed mean ± sample std。

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

### M17 `scripts/analyze_quality_downstream.py`
```text
--classification results/classification.csv
--quality results/generation_quality.csv
--figure reports/figures/quality_vs_downstream.png
--report reports/quality_vs_downstream.md
--output reports/quality_vs_downstream.json
```
只連接事前指定的六個點：兩物件 × `src_copypaste`／`src_procedural`／
`src_diffusion` seed-42 source ablation；品質側固定使用 M14 `unfiltered`、
`defect_type=__all__` 對應來源的 KID／NN mean，下游提升固定減同物件 seed-42
`real_only` Macro-F1。每一點保存兩個 run name 與兩份輸入 CSV SHA256；不因相關性
方向或強度不理想而換來源、換 metric 或刪點。

### M18 `src/training/train_segmenter.py`
```
--paths configs/paths.yaml
--config configs/segmenter.yaml
--object {pcb1,capsules}
--group STR                # real_only | std_aug | unfiltered_syn | filtered_syn | full_real
                           # | procedural_only | copypaste_only | diffusion_only
--seed INT                 # default: 42
--run-name STR
--mode {development,final}
--total-steps INT --learning-rate FLOAT --weight-decay FLOAT
--output-dir PATH
--drive-sync PATH          # exact Drive mirror directory
--resume-from-checkpoint {latest,PATH}
--smoke                    # real_only + development only, exactly 1 step
--dry-run                  # full data/hash preflight; never loads CUDA
```
產出：`${runs}/seg/<run-name>/{data_manifest.json,run_config.json,training_report.json,final/}`；
formal final run 另 append `results/segmentation.csv`。模型鎖定
`nvidia/segformer-b0-finetuned-ade-512-512@489d5cd...` SafeTensors、512 input、
binary Dice+BCE、batch 4、500 optimizer steps。normal image 的 target 是動態建立的
全零 mask；所有 anomaly mask 以 `>0` binary 化。development 只允許 `real_only` 且
test list 必須為空。portable `data_manifest.json` 直接保存
`split_manifest_sha256`，並由 `run_config.json` 與 `training_report.json` 保存其
canonical SHA256，三者必須由獨立 verifier 重算相符。

Colab 交接：
```text
scripts/package_m18_colab.py
  --paths configs/paths.yaml
  --config configs/segmenter.yaml
  [--object pcb1] [--object capsules]
  [--output-dir PATH]
  [--dry-run]

scripts/validate_m18_colab_notebook.py

scripts/verify_segmenter_smoke.py
  --paths configs/paths.yaml
  --config configs/segmenter.yaml
  [--run-root PATH]
  --output reports/segmenter_smoke_validation.json

scripts/validate_segmenter_runs.py
  --paths configs/paths.yaml
  --config configs/segmenter.yaml
  --run-root PATH
  --object {pcb1,capsules}
  [--reload]
  [--seed INT]              # 可重複；預設只有 42，既有單 seed 證據不受影響
  --output PATH
```
package 產一份 `defectforge_m18_source.zip` 與每物件一份
`m18_seg_<object>.zip`；data ZIP 只包含本機已展開／雜湊的 exact formal selections、
該物件完整 real train/test 與 pooled metadata。Notebook 每次跑一個物件的八個
canonical group，並下載 `m18_seg_results_<object>.zip`。獨立 validator 從 raw run
目錄驗 run signature、model hash、exact frozen test、blocklist、零 train/test overlap、
`procedural_only` 零 real defect 與 final model fresh reload。

註：第 9 組「全部混合」**＝ `filtered_syn`，直接引用不重跑**

### M20 `scripts/aggregate_segmentation.py`
```text
--paths configs/paths.yaml
--config configs/segmenter.yaml
--results-root results/colab/segmentation/3seed
[--seed INT]              # 可重複；預設 42 43 44（ADR-032），且必須含錨點 42
--output results/segmentation.csv
--report reports/segmentation_results.md
--validation-out reports/segmentation_validation.json
```

`--results-root` 先必須含兩個未改名、未手動解壓的
`m18_seg_results_{pcb1,capsules}.zip`。聚合器會在寫任何 M20 產物前，防 Zip Slip、
symlink、Windows ADS、大小／成員數異常與 CRC 錯誤，要求每包恰有
`3 + 5 × 8 × len(seeds)` 個白名單檔案（3 seeds＝123 個），核對 notebook validator 的
seed 清單與逐 run timings，再原子匯入到 `{object}/runs/`。重跑時以
`import_manifest.json` 綁定原 ZIP SHA256；ZIP 改變就 fail closed。其後仍由
`validate_segmenter_runs.py` 從 raw manifests／reports／SafeTensors 重新驗證，
不採信 notebook 產生的 per-object CSV。
先對兩物件各自的 raw run 呼叫獨立 M18 validator，再只從
`training_report.json` + `data_manifest.json` 重建 long-format CSV 與 Markdown；
Notebook 畫面和每個 Colab runtime 自己 append 的暫存 CSV 都不當數據源。
**每個 (group, object, seed) 一列**：48 列 `physical_run=true` 加上逐 seed 的
`all_mixed` 邏輯 alias 6 列（共 54 列），並保存 48 個 raw report SHA256。
Markdown 報表同時給 seed 42 錨點表與 mean ± std 表。

> 舊的單 seed 結果包留在 `results/colab/segmentation/`，不刪除、也不再被聚合器讀取。

### M21 `scripts/build_phase2_figures.py`
```text
--classification results/classification.csv
--segmentation results/segmentation.csv
--output-dir reports/figures
--validation-out reports/phase2_figures_validation.json
```
從 verified CSV 建 `real_scaling_curve.png`、`synthetic_volume_curve.png`、
`main_comparison_table.png`、`segmentation_table.png`。Filtered Syn 的「相當於幾張
真實瑕疵」先以 Real-only 的 10／20／60 raw 點做單調 isotonic fit，再在 fitted
Macro-F1 上分段內插；低於／高於觀察範圍只報 `≤10`／`≥60`，不做無界外插。validator
保存兩份輸入 CSV 與四張圖的 SHA256，且 `visual_inspection_required=true`；M21 勾選前
仍必須逐張實際開圖。

### M21 `scripts/build_sample_grids.py`
```text
--paths configs/paths.yaml
--object pcb1 --object capsules
--count 3
--output-dir reports/figures
--validation-out reports/sample_grids_validation.json
```
以 frozen few-shot、filtered copy-paste、filtered procedural、filtered SD2 searched、
SDXL searched 各列 deterministic 樣本，疊上真實／生成 mask。validation 保存每張
image、mask 與成圖 SHA256，並標記 `visual_inspection_required=true`。

### M22 `src/inference/demo_gradio.py`
```
--paths --object {pcb1,capsules} --port INT --share
# 或顯式指定：--cls-ckpt PATH --seg-ckpt PATH
# --share 預設 off，不對外開放
```
`--object` 會從完整正式 CSV 自動選 seed-42 physical checkpoint，逐項綁回 raw
training report，並保存不含本機絕對路徑的 selection evidence。

### M22 `scripts/record_demo_artifacts.py`
```text
--paths configs/paths.yaml
--classification-results results/classification.csv
--segmentation-results results/segmentation.csv
--segmentation-runs-root results/colab/segmentation
--selection-out reports/demo_checkpoint_selection.json
--gif-out assets/demo.gif
--validation-out reports/demo_validation.json
```

不啟動 Gradio share URL；依與 UI 相同的 deterministic selector，先把兩物件的
classifier／segmenter CSV row 綁回 raw report 與 SafeTensors，再依 frozen
`2cls_highshot` test manifest 每物件固定取一張 normal、一張 anomaly。四筆真模型
輸出都必須同時產生機率、binary mask、heatmap 與 latency，原子寫入四幀 GIF。
validation 保存兩份 CSV、selection、GIF、test image 與輸出 array SHA256；M22 勾選前
仍須實際啟動本機 UI 上傳 test 圖，並目視 GIF。

### 驗證用腳本
| 腳本 | 用途 |
|---|---|
| `scripts/verify_filter_report.py` | 從 `metadata.jsonl` 重算漏斗表，與 `reports/filter_report.md` 逐格比對 |
| `scripts/verify_generation_quality.py` | 核對 M14 CSV／Markdown／validation、sanity gate、圖與 feature-cache SHA256 |
| `scripts/verify_readme.py` | 從 `results/*.csv` 與 `--reproduction reports/seed42_reproduction.json` 重算 README 每個 verified 區塊並比對；分割表以 seed 42 為錨點，另出 mean ± std、跨機器重現與 ADR-032 判定三個區塊 |
| `scripts/verify_splits.py` | 重跑 [ADR-007](decisions.md#adr-007) 的四項斷言 |
| `scripts/verify_publish.py` | 唯讀稽核：里程碑、README、raw-hash evidence、正式圖/GIF、24 小時內上游授權、HF dry-run、必備檔、公開版面邊界（2 個 skill + CI workflow 必須存在，其餘 owner-local 必須不被追蹤）、token／個人路徑、檔案大小、Git 身分與 co-author trailer。CI 版加 `--allow-stale-license-check`（[ADR-029](decisions.md#adr-029)），時效如實回報但不 gating；**Release 一律不加 flag** |
| `scripts/build_release_acceptance.py` | M24 一頁驗收報告：除自身檔案外任一 local gate 未過即拒寫；只記通過項、修正項、殘留風險，不發佈 |
| `scripts/record_phase2_visual_review.py` | M21/M22 人工目視 evidence：所有正式 PNG/GIF 可解碼且實際逐張開啟後，需明確 confirmation 與觀察 note，保存目前檔案 SHA256 |
| `scripts/package_hf_release.py` | M24 本機封裝：預設只讀 inventory；`--build` 才原子建立 D 槽 HF dataset／model bundles，不連網 |
| `scripts/upload_hf.py` | **預設 `--dry-run`，上傳要顯式加 `--confirm`** |

### 複跑判定與重現性腳本

#### `scripts/decide_segmentation_replication.py`
```
--segmentation results/segmentation.csv
--output PATH              # 預設 reports/segmentation_replication.json
--report PATH              # 預設 reports/segmentation_replication.md
```
只執行 [ADR-032](decisions.md#adr-032) 在**複跑開始前**寫死的兩條規則，看到結果後不得修改：

1. **Dice／AUPRO 方向矛盾**：逐 seed 計算 `filtered_syn − real_only` 的兩個差值；
   若**至少一個物件**上符號相反出現在 3 個 seed 中的 ≥2 個 → `real_phenomenon`，
   否則 `single_seed_artefact`。差值為 0 視為平手，**不**算符號相反
2. **`capsules/std_aug` 崩潰**：Dice 在 ≥2 個 seed 為 `0.0000` → `systematic`；
   僅 1 個為 0 → `seed_noise`，並在報告中明載 ADR-031 的主張撤回

CSV 的 seed 集合與 `SEEDS` 不符即 fail closed，避免用不完整的表下判定。

#### `scripts/verify_seed42_reproduction.py`
```
--baseline reports/segmentation_seed42_baseline.csv
--segmentation results/segmentation.csv
--output PATH              # 預設 reports/seed42_reproduction.json
--report PATH              # 預設 reports/seed42_reproduction.md
```
複跑時 Drive 上沒有既有 `runs/` 可跳過，seed 42 因此在另一台機器被完整重跑。本腳本把
目前表的 seed-42 實跑列與**複跑前已 commit** 的 `segmentation_seed42_baseline.csv`
逐 run 比對 `model_sha256`、`run_signature` 與四項指標，輸出 `bit_identical` 與各指標
最大絕對差。**不相符時如實輸出差異，不視為錯誤、也不改寫任一邊。**

### 診斷用腳本

#### `scripts/diagnose_zero_dice_segmentation.py`
```
--paths --config configs/segmenter.yaml
--runs-root PATH           # 預設 results/colab/segmentation
--object STR               # 可重複；預設兩個物件
--device STR --batch-size INT
--output PATH              # 預設 reports/zero_dice_diagnosis.json
--report PATH              # 預設 reports/zero_dice_diagnosis.md
```
解釋為何部分 M20 分割 run 在 threshold 0.5 下 Dice 恰為 0。對每個 run 重載凍結的
`data_manifest.json` test 記錄與該 run 自己的 final checkpoint，重跑推論後：

1. **先重算已發佈指標並要求相符**（容差 `5e-3`，跨機器浮點差異），
   證明診斷與released 數字量的是同一件事；不符即 `exit 2`
2. 回報預測機率分布，重點是**最高機率**——它決定空 Mask 是否在預註冊 threshold 下
   **算術上必然**

**這是對已完成結果的事後量測**：不擬合模型、不選 checkpoint／threshold／超參，
不寫入 `results/`。Evaluation 是唯一允許讀 frozen test 的階段
（見 `.claude/skills/df-guard`）。

#### `scripts/diagnose_augmentation_mask_loss.py`
```
--paths --config configs/segmenter.yaml
--runs-root PATH           # 預設 results/colab/segmentation
--run-group STR            # 預設 std_aug
--object STR               # 可重複；預設兩個物件
--draws INT                # 每張瑕疵圖重放次數，預設 500
--seed INT --loss-window INT
--output PATH --report PATH
```
診斷標準增強為何讓 `capsules/std_aug` 崩潰（[ADR-031](decisions.md#adr-031)），兩段量測：

1. **重放**：對 frozen train partition 的每張真實瑕疵圖，重放 trainer 實際會抽到的
   同一組 draw（同 transform、同 `_stable_seed(seed, sample_id, draw_index)` 序列），
   量測空 mask 比率與保留面積比
2. **loss 軌跡**：比較同物件 `real_only` 與 `std_aug` 的窗平均 dice/BCE loss，
   判定 **Dice 項是否曾經啟動**（BCE 在每個 run 都會降，不能當判準）

**只讀 train partition**，不載入模型、不重算指標、不做任何選擇。

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

### v 系列 pilot 與放置診斷（ADR-035 → ADR-042）

這些腳本都**只讀 development 結果**，不讀 frozen test，也不重新生成任何合成資料。

#### `scripts/decide_v3_source_attribution.py`
```
--pilot results/v3/pilot_classification.json
--output reports/v3_source_attribution.json
--report reports/v3_source_attribution.md
```
執行 [ADR-035](decisions.md#adr-035) 的歸因規則：
`P = real_only − db_copypaste`（放置＋縫合）、`A = db_copypaste − db_diffusion`（外觀）。
兩物件同側才判定主因，否則判 `object_dependent`。
偵測「某物件三個 candidate 完全相同」並標記為無鑑別力（ADR-036）。

#### `scripts/diagnose_placement_geometry.py`
```
--paths configs/paths.yaml
--object STR               # 可重複；預設兩個物件
--output reports/placement_geometry.json
--report reports/placement_geometry.md
```
量測 M9 放置與真實瑕疵的差距：面積分布對比真實 p5–p95、實際套用的 affine scale／rotation、
以及 mask **外**環狀帶的像素統計（不是 mask 底下——真實 mask 底下已有瑕疵，
量那裡等於比較兩種不同的東西）。**先對所有來源檔比對 test blocklist，命中即 fail closed。**
報告中的結論全部由量測導出（ADR-034）。

#### `scripts/build_v4_pilot_config.py`
```
--paths configs/paths.yaml
--geometry reports/placement_geometry.json
--base-config configs/classifier.yaml
--base-out configs/classifier_v4_base.yaml
--output configs/classifier_v4_pilot.yaml
[--seed INT]               # 預設 42；只改訓練期隨機性，不改 arm 成員
[--run-subdirectory STR]   # 預設 cls_v4_pilot
[--result STR]             # 預設 results/v4/pilot_classification.json
```
決定性產生兩臂的 sample id 清單並寫進 base config 的 `sample_ids_by_object`，
因此 pilot runner **不需要新增任何 CLI**。arm 成員由固定的 `SELECTION_SEED = 42` 決定，
`--seed` 只改訓練隨機性——這使多 seed 成為**複跑**而非重新抽樣。

#### `scripts/decide_v4_placement_band.py`
```
--pilot results/v4/pilot_classification.json
--output reports/v4_placement_band.json
--report reports/v4_placement_band.md
```
執行 [ADR-038](decisions.md#adr-038)：`D = db_inband − db_current`，主要物件 `pcb1`、
指標 Macro-F1、門檻 `±0.01`；`capsules` 為對照，`|D| ≥ 0.05` 標記異常。
若主要物件無鑑別力，判定為 `uninformative_primary_object` 而**不是**「無效果」。

#### `scripts/decide_v5_seed_replication.py`
```
--seed42 results/v4/pilot_classification.json
--seed43 results/v5/pilot_classification_seed43.json
--seed44 results/v5/pilot_classification_seed44.json
--output reports/v5_seed_replication.json
--report reports/v5_seed_replication.md
```
執行 [ADR-040](decisions.md#adr-040)：主指標 **AUROC**（Macro-F1 在 `pcb1` 已知無鑑別力），
**判定只計入 seed 43／44**——seed 42 的數值已在 ADR-039 被引用，因此照常報告但不進判定式。
門檻 `±0.02` 沿用 ADR-026 的 per-object AUROC 容差原值。
每個 pilot 結果都會被斷言 `test_data_loaded == false`，否則 fail closed。
