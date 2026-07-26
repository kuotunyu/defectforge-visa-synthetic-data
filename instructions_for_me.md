# 換你做：Colab 操作手冊

> **狀態：骨架，尚未填內容。** 兩本 notebook 在里程碑 M10 / M11 完成、通過本機
> 1-step smoke test 之後，這份文件才會被填滿（見 [PLAN.md](PLAN.md) M15）。
> 填滿之前請不要照這份文件操作。

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

## Notebook 1 — `01_train_inpaint_lora_sd2.ipynb`

| 項目 | 內容 |
|---|---|
| 1. 上 Colab 方式 | TBD (M15) |
| 2. Runtime 選型 | TBD (M15) |
| 3. 需要的 Colab Secrets | TBD (M15) |
| 4. 預估時數與 compute units | TBD (M15) |
| 5. 跑完要下載哪些檔案 / 放回哪個路徑 | TBD (M15) |

## Notebook 2 — `02_train_inpaint_lora_sdxl.ipynb`

| 項目 | 內容 |
|---|---|
| 1. 上 Colab 方式 | TBD (M15) |
| 2. Runtime 選型 | TBD (M15) |
| 3. 需要的 Colab Secrets | TBD (M15) |
| 4. 預估時數與 compute units | TBD (M15) |
| 5. 跑完要下載哪些檔案 / 放回哪個路徑 | TBD (M15) |

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
