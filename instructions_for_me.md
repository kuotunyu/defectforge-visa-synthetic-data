# 換你做：Colab 操作手冊

> **狀態：Notebook 1 與 Notebook 2 均已完成。** M11 SDXL 的
> Colab L4 正式 fresh run、成果匯入、本機兩物件 one-step smoke 與 checkpoint
> resume 均已通過，不需再跑。M18 SegFormer 的本機 smoke、兩個 L4 正式 run、
> 回傳 ZIP 安全匯入與 M20 獨立驗證也已通過，不需再跑。

## 通用流程（每本 notebook 都一樣）

1. 把整個 `notebooks/` 資料夾複製到 Google Drive 的
   `MyDrive/sdg-portfolio/01-defectforge-visa/notebooks/`
2. 在 Colab 用「檔案 → 開啟筆記本 → Google 雲端硬碟」開啟，**不要**用上傳的方式開
3. 依下表選 Runtime，確認 Colab Secrets 已建立且對該 notebook 開啟存取權
4. 執行 → 跑完照「要下載的檔案」把產出抓回本機，放進 `results/colab/` 對應子目錄
5. 回到 Claude Code 貼「Colab 結果回報」prompt

## Colab 鐵則（notebook 內已實作，這裡列出供你檢查）

- 資料一定先解壓到 `/content/data` 再訓練，**不從掛載的 Drive 直接讀圖**
- checkpoint 定期同步回 Drive，中斷後重開同一本 notebook 會自動續跑
- 每本 notebook 用唯一的 `runs/` 子目錄，可以同時開多本平行跑
- token 只從 Colab Secrets 讀，notebook 內沒有明文金鑰

---

## 已在本機完成、不需上 Colab — M10 SD2 LoRA

ADR-008 已決定 SD2 使用本機 RTX 4090；兩物件各 400 steps 的正式訓練、checkpoint、
fresh reload 與驗證都已通過，因此沒有 `01_train_inpaint_lora_sd2.ipynb`，也不需要
建立假的 Colab 步驟。實測如下：

| 物件 | 訓練秒數 | wall-clock 秒數 | peak VRAM |
|---|---:|---:|---:|
| pcb1 | 128.66 | 134.82 | 3.20 GiB |
| capsules | 125.65 | 131.66 | 3.20 GiB |

正式產物在 `D:\sdg-data\01-defectforge\runs\lora_sd2\`，獨立驗證報告是
`reports/lora_sd2_validation.json`。

## Notebook 1 — `01_train_inpaint_lora_sdxl.ipynb`

| 項目 | 內容 |
|---|---|
| 1. 上 Colab 方式 | 把 `D:\sdg-data\01-defectforge\colab\m11\` 的兩個 zip 上傳到 `MyDrive/sdg-portfolio/01-defectforge-visa/`，再從 Drive 開 notebook |
| 2. Runtime 選型 | **L4 GPU（24 GB）**；notebook 會在總 VRAM <20 GiB 時先停止 |
| 3. 需要的 Colab Secrets | `HF_TOKEN`，只由 `google.colab.userdata` 讀取；notebook 無明文 token |
| 4. 實測時數與 compute units | pcb1 wrapper 922.2 s、capsules 855.8 s，合計 29 分 38 秒；正式 training 826.28 / 830.61 s；peak VRAM 皆 9.680 GiB。執行前未記 CU，無法誠實回推 |
| 5. 跑完要下載哪些檔案 / 放回哪個路徑 | **已完成**：validation JSON、兩物件 `training_report.json`、`final/`、`samples/` 共 49 files，已放回本機 `results/colab/lora_sdxl/`；Drive checkpoints 暫不刪 |

本機補充驗收也已完成：pcb1 / capsules one-step smoke 的 peak VRAM 皆為
9.619 GiB，pcb1 受控 resume 從 step 1 恢復至 step 2，peak 9.663 GiB；三個 final
bundle 都通過 fresh `PeftModel` 雙 encoder 重載。證據在
`reports/lora_sdxl_local_validation.json`。目前不需再操作 Notebook 1。

## Notebook 2 — `02_train_segformer.ipynb`

| 項目 | 內容 |
|---|---|
| 1. 上 Colab 方式 | 將本機 `%USERPROFILE%\Desktop\mySyntheticData\1_DefectForge\notebooks\02_train_segformer.ipynb` 上傳到 Drive 的 `MyDrive/sdg-portfolio/01-defectforge-visa/notebooks/`；再將 `D:\sdg-data\01-defectforge\colab\m18\defectforge_m18_source.zip`、`m18_seg_pcb1.zip`、`m18_seg_capsules.zip` 三個檔案上傳到 `MyDrive/sdg-portfolio/01-defectforge-visa/`。從 Drive 開 notebook，不要用 Colab 的臨時上傳方式。每次只設定一個 `OBJECT_NAME`：先 `pcb1`，完成後再用新 runtime 設 `capsules`。 |
| 2. Runtime 選型 | 正式 run 實際使用 **L4 GPU（24 GB）**；程式最低要求仍是 T4-class、至少 14 GiB VRAM。本機 RTX 4090 one-step smoke 的 peak VRAM 兩物件皆為 0.977 GiB；notebook 會在 CUDA 缺失或總 VRAM <14 GiB 時先停止。 |
| 3. Colab Secrets | **不需要任何 Secret**。鎖定的 NVIDIA SegFormer checkpoint 是公開模型；Notebook 2 不讀 `HF_TOKEN`，也沒有明文憑證。 |
| 4. 實測時數與 compute units | 固定 500 optimizer steps × 8 個實跑 group × 2 物件；`all_mixed` 只引用 `filtered_syn`，沒有第九次重跑。L4 實測：pcb1 八組合計 2,509.070 秒（41.82 分），capsules 合計 2,321.437 秒（38.69 分），總計 80.51 分；帳戶 CU 前後值未記錄，因此不回推 CU。 |
| 5. 跑完下載／回收 | **已完成**：兩個 notebook validator 均通過；原始 `m18_seg_results_pcb1.zip` 與 `m18_seg_results_capsules.zip` 已放入本機 `results/colab/segmentation/`，並由 M20 聚合器完成安全匯入與獨立驗證。Drive checkpoints 暫不影響本機重現。 |

### 上傳檔案核對

| 本機檔案 | 大小 | SHA256 |
|---|---:|---|
| `D:\sdg-data\01-defectforge\colab\m18\defectforge_m18_source.zip` | 104,900,929 bytes | `d6c9f71b25796c549eba0163f8f9911fccc2a8376ea967bd3258e1749f8a49f8` |
| `D:\sdg-data\01-defectforge\colab\m18\m18_seg_pcb1.zip` | 3,764,604,320 bytes | `50a8ab3b7eaf927089d18a5f2c86612e77cb1226dfdd8b34485efbafde4c573c` |
| `D:\sdg-data\01-defectforge\colab\m18\m18_seg_capsules.zip` | 3,948,056,619 bytes | `d2c0672f843b5daec2e7798495900faa4c26e25aca8a3f72ab8ec752a19f7688` |

封裝 manifest 是
`D:\sdg-data\01-defectforge\colab\m18\m18_colab_bundle.json`。兩個資料包的
training blocklist hits 都是 0；包內包含 frozen test 只供最後 formal evaluation，
不會用於 fitting、選超參或 early stopping。

---

## 需要你人工確認的檢查點（非 Colab）

這些是本機流程中會停下來等你看圖的地方，時間點與看什麼：

| 里程碑 | 你要看的東西 | 判斷標準 |
|---|---|---|
| M5 | `reports/figures/fewshot_contact_sheet_*.png` | 每格的瑕疵肉眼可辨、mask 輪廓有對準 |
| M6 | 各分群的 contact sheet | 同一群的瑕疵長得像；替每群取一個名字（會變成 trigger token） |
| M9 | mask placement 視覺化檢查圖 | ROI 框有框在物件上、放置的 mask 沒有跑到背景 |
| M12 | `original/` vs `searched/` 並排 grid | refine 之後是否真的更像真實瑕疵 |
| M13 | 通過 / 被拒樣本的並排 grid | 過濾有沒有把明顯好的樣本刷掉 |
