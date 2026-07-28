# Data & Split Protocol

> 對應里程碑 M2–M6。**這份協定凍結之前，一張合成圖都不准生。**
> 相關決策：[ADR-002](decisions.md#adr-002) 瑕疵分型、[ADR-003](decisions.md#adr-003) few-shot 預算、[ADR-005](decisions.md#adr-005) 儲存佈局。

---

## 1. VisA 事實表（已查證）

| 項目 | 值 | 來源 |
|---|---|---|
| 授權 | **CC BY 4.0**（允許衍生物公開發佈，需標註來源） | [spot-diff README](https://github.com/amazon-science/spot-diff) |
| 下載 URL | `https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar` | [AWS Open Data Registry](https://registry.opendata.aws/visa/) |
| **tar 大小** | **1,929,840,640 bytes = 1.80 GB**（2026-07-27 HTTP HEAD 實測，回 200） | 本機實測 |
| 解壓後 | **1,920,559,633 bytes（12,037 files）** | M2 實測 |
| 總量 | 10,821 張（9,621 normal / 1,200 anomaly）、12 物件、3 domain | [arXiv:2207.14315](https://arxiv.org/abs/2207.14315) |
| 影像解析度 | 約 1500 × 1000 px | 同上 |
| 瑕疵尺寸 | **明顯小於 MVTec AD** | 同上 → [ADR-004](decisions.md#adr-004) |
| `pcb1` | 1,004 normal / 100 anomaly，論文記載 4 個 anomaly class | 同上 |
| `capsules` | 602 normal / 100 anomaly，論文記載 5 個 anomaly class | 同上 |
| 每型張數 | 5–20 張，**一張圖可能含多個瑕疵** | 同上 |

### 原始目錄結構

官方 tar 的 members **沒有**外層 `VisA/`；`scripts/download_visa.py` 因此直接解壓到
設定的 `${visa_raw}`，由本專案補上這層目錄：

```
VisA/
  <object>/
    Data/
      Images/
        Anomaly/   *.JPG      # 瑕疵影像
        Normal/    *.JPG      # 正常影像
      Masks/
        Anomaly/   *.png      # 只有瑕疵影像有 mask
    image_anno.csv
```

### ⚠️ 關鍵限制：沒有 per-image 瑕疵類型標註
釋出的標註只有 **anomaly / normal 二元標籤 + pixel-level mask**。
論文描述的 4/5 個 anomaly class **沒有以 per-image label 的形式釋出**。
spot-diff 的 split CSV 欄位是 `object, set, label, image_path, mask_path`，
`label` 只有 `normal` / `anomaly`。
→ 因此瑕疵型別必須自行推導，方法見 [ADR-002](decisions.md#adr-002) 與本文第 5 節。

---

## 2. 下載與校驗（M2）

1. **先跟使用者報備大小（1.80 GB）再下載**
2. 下載到 `${data_root}/raw/VisA_20220922.tar`
3. 斷言檔案大小 `== 1929840640` bytes，不符即停
4. 計算 SHA256 寫進 `splits/source_checksums.json`
5. 解壓到 `${data_root}/raw/VisA/`
6. 逐項比對第 1 節的事實表（pcb1 1004/100、capsules 602/100）；**不符就停下來報告，不要自行調整**

2026-07-27 實測 SHA256：
`2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`。

`source_checksums.json` schema：
```json
{
  "VisA_20220922.tar": { "bytes": 1929840640, "sha256": "…", "downloaded_at": "2026-…", "url": "…" }
}
```

---

## 3. 官方 split 佈局（M3）

用 spot-diff 官方工具，**不自己重寫切分邏輯**：

```
python ./utils/prepare_data.py --split-type 2cls_fewshot  --data-folder <VisA> --save-folder <out_fewshot>  --split-file ./split_csv/2cls_fewshot.csv
python ./utils/prepare_data.py --split-type 2cls_highshot --data-folder <VisA> --save-folder <out_highshot> --split-file ./split_csv/2cls_highshot.csv
```

產出的 2-class 目錄結構：
```
<out>/<object>/
  train/{good,bad}/
  test/{good,bad}/
  ground_truth/train/bad/
  ground_truth/test/bad/
```

### 3.1 官方協定與實測張數

論文的比例：`2cls_fewshot` = 20%/80%（再從 train pool 抽 k=5 或 10）；`2cls_highshot` = 60%/40%。
**下載兩份官方 CSV 實際計算後的張數**（2026-07-27，非推測）：

| CSV | 物件 | train | test |
|---|---|---|---|
| `2cls_fewshot` | pcb1 | 201 normal / **20** anomaly | 803 normal / **80** anomaly |
| `2cls_fewshot` | capsules | 120 normal / **20** anomaly | 482 normal / **80** anomaly |
| `2cls_highshot` | pcb1 | 602 normal / **60** anomaly | 402 normal / **40** anomaly |
| `2cls_highshot` | capsules | 361 normal / **60** anomaly | 241 normal / **40** anomaly |

### 3.2 ⚠️ 兩套切法**不可混用**（實測結論）

```
highshot TRAIN(anomaly) ∩ fewshot TEST(anomaly) = 40   ← 兩個物件都是 40
highshot TRAIN(normal)  ∩ fewshot TEST(normal)  = 401 (pcb1) / 241 (capsules)
```

拿 highshot 訓練、用 fewshot test 評測 → **一半的測試瑕疵在訓練集裡**。詳見 [ADR-007](decisions.md#adr-007)。

但兩套切法是**巢狀**的，這救了我們：
```
fewshot TRAIN ⊂ highshot TRAIN     (實測 True)
highshot TEST ⊂ fewshot TEST       (實測 True)
```

### 3.3 本專案採用的基底：`2cls_highshot`（[ADR-007](decisions.md#adr-007)）

```
k=10  ⊂  fewshot_train(20)  ⊂  highshot_train(60)      三者皆與 highshot_test 互斥
```

| 角色 | 定義 | pcb1 | capsules |
|---|---|---|---|
| **Test（唯一、凍結）** | highshot test | 402 normal / 40 anomaly | 241 normal / 40 anomaly |
| **Train pool** | highshot train | 602 normal / 60 anomaly | 361 normal / 60 anomaly |
| **few-shot 瑕疵集** | 從 fewshot train 的 20 張中 seed=42 抽 k=10 | 10 | 10 |
| **正常圖（所有組共用）** | train pool 全部 normal | 602 | 361 |
| **中間點** | fewshot train 全部 anomaly | 20 | 20 |
| **Full-real 上限** | train pool 全部 anomaly | 60 | 60 |

→ 免費得到一條「真實瑕疵 **10 → 20 → 60** 張」的縮放曲線。

**M3 驗證（任一不符即停）**：
1. 重算 3.1 的八格數字，與上表逐格相符
2. `highshot_train ∩ highshot_test == ∅`
3. `fewshot_train ⊆ highshot_train`
4. 每張 bad 影像都有對應 mask（數量相等、檔名可對上）

---

## 4. Split manifest 凍結（M4）

### 4.1 pHash 近似分群
VisA 同一物件的影像高度相似，且可能存在近乎重複的拍攝。
規則：**同一個 pHash 群的影像必須落在同一個 split**。

1. 對每張影像算 perceptual hash（`imagehash.phash`，hash_size=16）
2. Hamming distance ≤ `PHASH_THRESHOLD`（初值 6，M4 依實測分布調整並記錄）視為同群
3. 用 union-find 傳遞閉包成 `group_id`
4. 若某群橫跨 train 與 test → **整群移到 test 側**（保守做法：寧可縮小訓練集，也不能讓訓練看過測試的近似圖）
5. 移動後重印計數表，並在 `reports/split_report.md` 記錄「因 pHash 合併而移動了幾張」

### 4.2 `splits/split_manifest.json` schema
```json
{
  "created_at": "2026-…",
  "seed": 42,
  "phash_threshold": 6,
  "source_checksums": "splits/source_checksums.json",
  "objects": ["pcb1", "capsules"],
  "images": [
    {
      "object": "pcb1",
      "set": "train",              // train | test
      "label": "bad",              // good | bad
      "split_type": "2cls_fewshot",
      "image_path": "pcb1/Data/Images/Anomaly/000.JPG",   // relative to visa_raw
      "mask_path":  "pcb1/Data/Masks/Anomaly/000.png",    // null for good
      "sha256": "…",
      "mask_sha256": "…",          // null for good
      "phash": "…",
      "group_id": 17,
      "in_fewshot_pool": true,     // 是否屬於官方 2cls_fewshot train
      "original_set": "train",
      "moved_to_test_by_phash": false
    }
  ]
}
```

manifest 自身的 SHA256 寫進 `splits/MANIFEST.sha256`。
**凍結後不得修改**；若必須修改，新增 `split_manifest_v2.json` 並寫一筆 ADR 說明原因。
M5 的 `in_fewshot_seed` / `in_val` 選擇另存於 `splits/fewshot_selection.json`，
並反向記錄 manifest SHA256，不回寫本檔（[ADR-013](decisions.md#adr-013)）。

### 4.3 Test blocklist（防洩漏）
`splits/test_blocklist.json`：所有 test 影像與其 mask 的 SHA256 集合。

```json
{
  "image_count": 1234,
  "mask_count": 80,
  "unique_sha256_count": 1314,
  "sha256": ["…", "…"]
}
```

`sha256` 是集合，因此若兩個檔案 byte-identical，只計一個 unique hash；
驗證條件是**每一個** test image / mask 的 hash 都能在集合中找到，而不是假設檔案數必等於
unique hash 數（[ADR-013](decisions.md#adr-013)）。

**所有**生成、過濾、分型、評測腳本在讀任何影像前，都要先把該檔的 SHA256 拿去比對；
命中即 `raise` 並中止。這個檢查由 `df-guard` skill 在每個階段開始前跑一次，
腳本內部也各自再檢查一次（雙保險）。

---

## 5. few-shot 抽樣與瑕疵分型（M5–M6）

### 5.1 few-shot 預算（[ADR-003](decisions.md#adr-003) 原則 + [ADR-007](decisions.md#adr-007) 基底）

| 項目 | 值 |
|---|---|
| 真實瑕疵圖 | **k = 10 張／物件**，從 `2cls_fewshot` train 的 20 張中以 seed=42 抽 |
| 真實正常圖 | **highshot train pool 全部**（pcb1 **602**、capsules **361**） |
| 適用組別 | Real-only / +Std Aug / +Unfiltered Syn / +Filtered Syn **共用完全相同的真實影像集合** |
| Full-real 上限組 | highshot train 的全部 60 張瑕疵 + 同一批正常圖（**同一個 test set，零洩漏**） |
| 中間點（免費） | fewshot train 的全部 20 張瑕疵 → 真實資料縮放曲線 10 / 20 / 60 |
| Validation | 只從 train pool 切、只用真實資料，**所有組共用同一個 val 集合** |
| Test | highshot test，**原封不動** |
| 類別不平衡 | pcb1 約 60:1、capsules 約 36:1 → **所有組一律用相同的 class-balanced sampling**，不得只對某組調整 |

抽樣必須可重現：固定 `random.Random(42)`，對排序後的檔名清單抽樣。
**驗證方式：重跑兩次，抽出的檔名清單雜湊必須相同。**

Validation 以每物件 × label 的 highshot train pool 取 10%（向下取整、至少 1 張），
候選必須排除官方 fewshot train pool，避免侵蝕 k=10 seed；它只用於 Real-only 開發階段
凍結超參與 final training steps。正式比較再用凍結設定 refit 完整 train pool，
所以 10 / 20 / 60 的實驗口徑不變（[ADR-013](decisions.md#adr-013)）。

### 5.2 樣本統計與 contact sheet
- `reports/fewshot_stats.md`：每物件 × 每 split × 每 label 的張數、mask 面積分布
  （min/median/max/5th/95th 百分位）、瑕疵佔全圖面積比
- `reports/figures/fewshot_contact_sheet_<object>.png`：10 張 seed 並排，
  **每格疊上 mask 輪廓**，供人眼確認瑕疵位置正確

### 5.3 瑕疵分型流程（[ADR-002](decisions.md#adr-002)）

```
few-shot seed 的 GT mask
  → 連通元件分解（面積 < MIN_COMPONENT_AREA 的雜點濾掉）
  → 每個元件抽特徵
       (a) DINOv2 (facebook/dinov2-base) 對元件 ROI crop 的 CLS embedding
       (b) 形態特徵：area_ratio, aspect_ratio, solidity, extent, eccentricity,
           components_in_image, mean/contrast inside-vs-outside mask
  → 各自標準化後串接
  → Agglomerative clustering，k ∈ [1..5] 用 silhouette 選，硬性條件「每群 ≥ 3 個元件」
  → 每群產 contact sheet
  → 【停下來，等使用者確認或改名】
  → 凍結 splits/defect_types.json + SHA256
```

**Fallback**：任何群 <3 個元件 → 併入 `<obj>-defect`；可用型別 <2 → 該物件單一 trigger token。

### 5.4 `splits/defect_types.json` schema
```json
{
  "created_at": "2026-…",
  "method": "agglomerative(dinov2-base CLS + morphology), silhouette-selected k",
  "confirmed_by_user": true,
  "objects": {
    "pcb1": {
      "n_components": 23,
      "types": [
        {
          "cluster_id": 0,
          "type_name": "scratch",
          "trigger_token": "<pcb1-scratch>",
          "n_components": 9,
          "components": [
            { "image_path": "pcb1/…/000.JPG", "component_id": 0, "bbox": [x, y, w, h], "area_px": 812 }
          ]
        }
      ],
      "fallback_applied": false
    }
  }
}
```

---

## 6. 防洩漏聲明（README 會照抄）

1. Validation 與 Test **只含真實影像**，從未加入任何合成樣本
2. 生成器（LoRA 訓練、mask placement、inpaint）**只讀 few-shot seed 與 train pool 的正常圖**
3. 過濾器與品質指標的參考集 `R` **只用 few-shot seed 的真實瑕疵 crop**
4. 瑕疵分型的分群**只在 few-shot seed 上做**
5. split manifest 在生成第一張合成圖之前就已凍結並記錄 SHA256
6. 近似影像用 pHash 分群，同群必定同 split；跨界的群一律整群歸入 test
7. 以上每一條都有對應的程式斷言（test blocklist 比對），不是只寫在文件上

---

## 7. 已知風險

| 風險 | 影響 | 緩解 |
|---|---|---|
| few-shot 只有 10 張瑕疵 → 分型後每型可能只剩 2–4 個元件 | trigger token 學不起來 | [ADR-002](decisions.md#adr-002) 的 fallback；必要時退回單一 token |
| pHash 分群把太多影像合併 → train pool 被掏空 | 訓練資料不足 | M4 先印分群大小分布再決定閾值，並記錄移動張數 |
| train pool 正常圖數量兩物件差很多（602 vs 361） | 物件間不可直接比較 | per-object 指標分開報告；合成配額按物件各自計算 |
| **Test 只有 40 張瑕疵／物件**（採 highshot 基底的代價） | 指標變異大 | 所有表格附樣本數；Real-only 與最佳 Filtered 組**必須**補 3 seeds 報 mean±std；不要對 <1% 的差異下結論 |
| 極端類別不平衡（pcb1 60:1） | 準確率失去意義 | 主指標用 Macro-F1 與 AUROC，額外報正常品 false-positive rate；所有組共用同一套 balanced sampling |
| VisA 下載 URL 或內容變更 | 不可重現 | `source_checksums.json` 鎖住 SHA256；不符即停 |
