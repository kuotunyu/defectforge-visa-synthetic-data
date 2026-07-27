# 換你做：Colab 操作手冊

> **狀態：Notebook 1 已完成。** M11 SDXL 的 Colab L4 正式 fresh run、成果匯入、
> 本機兩物件 one-step smoke 與 checkpoint resume 均已通過，不需再跑。Notebook 2
>（M18 SegFormer）尚未建立；在它備妥前目前沒有 Colab 操作。

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

## Notebook 2 — M18 SegFormer（尚未建立）

這本 notebook 屬於 Phase 2 的 M18，會在分類基準 M16 與品質關聯分析 M17 完成後建立。
M18 必須先完成本機 1-step smoke，然後才會在這裡填入下列五項**可直接照做**的操作資料：

1. 上 Colab 的確切檔名、Drive 來源與開啟方式
2. Runtime GPU 選型與最低 VRAM
3. 所需 Colab Secrets（若不需要也會明寫「不需要」）
4. 依本機 smoke 推估的時數與 compute units，並要求回報實測值
5. 完整下載清單與 `results/colab/` 的本機回收路徑

在 M18 完成前不要自行建立或執行分割 notebook；目前沒有需要你操作的 Colab 工作。

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
