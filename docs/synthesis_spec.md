# Synthesis Spec — Stage A & Stage B

> 對應里程碑 M7–M12。方法論背景見 [methodology.md](methodology.md)。
> **前置條件：M4 的 split manifest 與 M6 的 defect_types.json 都已凍結。**

---

## 0. 共同規則

- **標註免費**：所有合成樣本的 segmentation ground truth = 生成時使用的那張 mask。
  絕不事後用模型推 mask 當 GT（那會引入偽標註誤差）。
- **輸出佈局**（`${data_root}/synthetic/` 之下）：
  ```
  stageA_copypaste/{images,masks}/  metadata.jsonl
  stageA_procedural/{images,masks}/ metadata.jsonl
  stageB_sd2/original/{images,masks}/   metadata.jsonl
  stageB_sd2/searched/{images,masks}/   metadata.jsonl
  stageB_sdxl/original/…                stageB_sdxl/searched/…
  ```
- **可重現**：同一組 (seed, 參數) 重跑必須產出位元相同的影像。所有隨機來源用單一
  `numpy.random.Generator(PCG64(seed))` 派生，不使用全域 random state。
- **檔名**：`<object>__<type>__<srcstem>__<idx:05d>.png`，mask 同名。
- **每張都要過 test blocklist 檢查**（[data_protocol.md §4.3](data_protocol.md)）。

---

## 1. `metadata.jsonl` provenance schema

**每一行一個樣本，欄位缺一即視為失敗**（M12 有 schema 驗證腳本逐行檢查）。

```json
{
  "sample_id": "pcb1__scratch__0007__00042",
  "object": "pcb1",
  "defect_type": "scratch",
  "trigger_token": "<pcb1-scratch>",
  "generator": "stageB_sd2",              // stageA_copypaste | stageA_procedural | stageB_sd2 | stageB_sdxl
  "bucket": "original",                   // original | searched | null (Stage A)
  "image_path": "images/pcb1__scratch__0007__00042.png",
  "mask_path":  "masks/pcb1__scratch__0007__00042.png",

  "source": {
    "background_image": "pcb1/Data/Images/Normal/0123.JPG",
    "background_sha256": "…",
    "defect_source_image": "pcb1/Data/Images/Anomaly/0007.JPG",   // null for procedural
    "defect_source_mask":  "pcb1/Data/Masks/Anomaly/0007.png",
    "defect_source_component_id": 1
  },

  "placement": {
    "roi_bbox": [x, y, w, h],
    "mask_bbox": [x, y, w, h],
    "affine": { "dx": 0, "dy": 0, "rotation_deg": 0.0, "scale": 1.0, "flip": false },
    "mask_area_px": 812,
    "mask_area_ratio": 0.00054
  },

  "generation": {
    "seed": 1234567,
    "base_model": "sd2-community/stable-diffusion-2-inpainting",
    "lora_path": "runs/lora_sd2/pcb1/seed_42/final",
    "prompt": "a photo of <pcb1-scratch> defect on a printed circuit board",
    "negative_prompt": "...",
    "guidance_scale": 7.5,
    "num_inference_steps": 40,
    "strength": 1.0,
    "crop_ratio": 2.5,
    "crop_bbox": [x, y, w, h],
    "model_resolution": 512,
    "blend": "poisson"                     // poisson | feather_alpha
  },

  "filter": {
    "passed": true,
    "scores": { "nn_score": 0.71, "roi_containment": 1.0, "area_zscore": 0.4,
                "aspect_zscore": -0.8, "seam_score": 0.93, "phash_min_dist": 14 },
    "reject_reasons": []
  },

  "pipeline_version": "0.1.0",
  "created_at": "2026-…"
}
```

Stage A 的樣本把 `generation` 內不適用的欄位設為 `null`，不省略欄位。
`filter` 區塊在生成時先寫 `null`，M13 過濾後回填（`unfiltered` 版保留全部樣本並標
`passed=false` + 原因；`filtered` 版只含 `passed=true`）。

---

## 2. 生成配額（對應課程 `prep-testcase`）

課程的 `prep-testcase` 依「訓練 mask 數量**按比例**」分配 `num_SDG`。照做：

```
每個瑕疵型的目標張數 = ceil(TOTAL_PER_OBJECT × n_components(type) / n_components(object))
```

Phase 1 的目標值（[ADR-010](decisions.md#adr-010)）：

| generator | 每物件總量 | 備註 |
|---|---|---|
| `stageA_copypaste` | **500** | |
| `stageA_procedural` | **500** | 另跑一份 `--no-real-stats` 版本（[ADR-011](decisions.md#adr-011)） |
| `stageB_sd2` | **500** | 主線；Phase 2 的合成量掃描點 {125, 250, 500} 從這裡取子集 |
| `stageB_sdxl` | **250** | 底模容量消融；與 SD2 在 **250 張**這一點上比較 |

配額在**物件**層級決定，再依各瑕疵型的連通元件數按比例分到型別。
若某型的配額 <50，補到 50（避免該型樣本少到無法做 per-type 指標），
並在 `reports/generation_report.md` 記錄補齊了哪幾型、補了多少。

**不要為了「更完整」在第一輪就跑 1000 張。** 生成腳本支援 `--resume`，
Phase 2 若發現曲線在 500 仍在上升，再補跑到 1000，既有的 500 張不重生。

mask 的**面積 / 位置 / 長寬比分布**由真實 few-shot mask 的統計（`reports/real_mask_stats.json`）
控制：抽樣時從真實分布的 KDE 取樣，超出 5–95 百分位就重抽。

---

## 3. Stage A — 本機保底合成（M7–M8）

不需要任何訓練，先把「合成資料→下游驗證」的整條路打通，作為 Stage B 的對照與保險。

### 3.1 `src/synthetic/copy_paste.py`（M7）
```
從 few-shot seed 的 (image, mask) 取出 defect patch（含 alpha = mask）
  → 隨機仿射：旋轉 ±180°、縮放 0.7–1.4×、可選鏡射
  → 隨機色彩擾動：亮度 / 對比 / 飽和 / 色相 小幅抖動（範圍寫進 config）
  → 隨機透明度 0.75–1.0
  → 選擇同物件的正常 train 圖當背景，在合法 ROI 內選位置
  → 邊緣處理：alpha 羽化（高斯，半徑隨機）或 Poisson (cv2.seamlessClone)
  → 輸出 (合成圖, 變換後的 mask)
```
**參考**：Ghiasi et al., *Simple Copy-Paste is a Strong Data Augmentation Method* (CVPR 2021)。

**M7 驗證**：mask 與影像尺寸相同、mask 非全零、貼上位置 100% 在 ROI 內；
隨機抽 24 張做 grid **自己開起來看**，確認沒有明顯矩形接縫、瑕疵沒貼到背景上。

### 3.2 `src/synthetic/procedural.py`（M8）
四種形狀產生器，**不使用任何真實瑕疵像素**（這支撐 Phase 2「零真實瑕疵」那組響亮標題）：

| 形狀 | 做法 |
|---|---|
| Perlin blob | Perlin/simplex noise 閾值化 → 不規則斑塊，模擬污漬、變色 |
| Crack | 隨機遊走 + 分支的細線 → 膨脹到 1–3 px 寬，模擬裂痕 |
| Scratch | 帶輕微曲率的線段，長寬比極高，模擬刮痕 |
| Spot | 橢圓／多邊形擾動，模擬異物、焊點缺陷 |

紋理填充：以背景區域統計為基礎，加上高斯雜訊、局部亮度偏移與邊緣銳化；
不是純色填滿（純色會被下游模型當成 trivial 特徵）。

**參考**：Zavrtanik et al., *DRAEM* (ICCV 2021) 的合成異常思路。

#### ⚠️「零真實瑕疵」的口徑（[ADR-011](decisions.md#adr-011)）
本產生器**不使用任何真實瑕疵像素**，但預設會用 `reports/real_mask_stats.json`
（來自 10 張 few-shot seed 的 mask 面積比與長寬比百分位）來約束形狀。
這是唯一的洩漏面，**必須主動揭露**，不得用「零真實瑕疵」含糊帶過。

必須支援 `--no-real-stats`：改用手訂的固定分布，完全不看真實統計，
另存成 `stageA_procedural_norealstats/`，在 Phase 2 的分割報表中與預設版**並列**。

**M8 驗證**：同 M7 的自動斷言；預設版的 mask 面積與長寬比落在
`reports/real_mask_stats.json` 的 5–95 百分位內、超出比例 <10%；
`--no-real-stats` 版本要斷言執行過程中**從未開啟** `real_mask_stats.json`。

---

## 4. Stage B — Diffusion inpainting（M9–M12，課程復刻主秀）

### 4.1 Auto Mask Placement（M9）— `src/synthetic/mask_placement.py`

對應課程 slide 7 的 `Text2Box ROI Generator → Auto Mask Placer`。

```
輸入：一張正常圖 + 一個真實 mask（連通元件）
  ① ROI Generator
       Otsu / 形態學前景分割  →  物件主體遮罩
       DINOv2 patch-token 異質度圖  →  「有結構的可放置區」
       兩者取交集 = legal ROI；背景 / 影像邊界 / 已放置 mask 鄰域 = illegal
  ② Auto Mask Placer
       真實 mask → 隨機仿射（dx, dy, rotation, scale, flip）
       排斥取樣直到：100% 落在 legal ROI 內 ∧ 面積與長寬比在真實分布 5–95 百分位內
       重試上限 MAX_PLACE_TRIES（超過就換一張背景，並記錄失敗次數）
  ③ 輸出
       (clean image, placed mask) 配對
       視覺化檢查圖：原圖 + ROI 邊界 + 放置 mask 疊圖
```

**M9 驗證**：自動斷言（100% 在 ROI 內、不與其他 mask 重疊、面積分布合格）＋
抽 24 張視覺化檢查圖**自己看**，ROI 明顯抓錯就換方法（例如改用 DINOv2 前景分割為主）。

### 4.2 LoRA 微調（M10 SD2 本機 / M11 SDXL Colab）

**訓練邏輯只有一份實作**：`src/training/train_inpaint_lora.py`。
本機 CLI 與 Colab notebook **都呼叫同一支腳本**，notebook 只做掛 Drive、
解壓資料到 `/content/data`、讀 Secrets、組參數、同步 checkpoint。
**不得把訓練迴圈複製進 notebook**（[ADR-008](decisions.md#adr-008)）。

| 項目 | 設定 |
|---|---|
| 訓練資料 | few-shot seed 的 (瑕疵圖, mask, defect_type) 三元組，**每物件 10 張** |
| 訓練樣本形式 | 依 [ADR-004](decisions.md#adr-004) 先 crop-to-ROI 成模型解析度的 patch，再訓練 |
| 適配器 | 每**物件**一個 LoRA（UNet attention 層），rank / alpha 寫進 config |
| 型別區分 | 每**瑕疵型**一個 trigger token，嵌在 prompt 內；token embedding 與 LoRA 一起訓 |
| prompt 模板 | `a photo of <obj-type> defect on <object description>` |
| 輸出目錄 | `runs/lora_<model>/<object>/seed_42/` |
| **SD2 平台** | **本機 4090**（M10 實測每物件約 126–129 秒、峰值 3.203 GiB）。**若未來實測超過 30 分鐘就改回 Colab 並更新 ADR-008** |
| **SDXL 平台** | **Colab L4**（2.6B UNet @1024，超過本機門檻） |
| 斷點續跑 | 兩邊都要支援；Colab 版偵測 Drive 上既有 checkpoint 自動接續 |

⚠️ **10 張訓練圖極容易 overfit**。必要防護：低 LoRA rank、早停、
固定一組 held-out prompt 每 N steps 生一張樣本圖存進 `runs/.../samples/` 供目視，
以及 M13 的 `nn_score ≥ τ_copy` 過濾（把「只是複製原圖」的樣本刷掉）。

**M10 驗證（SD2，本機完整訓練）**：訓練跑完並存出可被 `PeftModel` 載回的權重；
記錄實際耗時（超過 30 分鐘要回報並改計畫）；`runs/.../samples/` 的樣本圖**自己打開檢視**，
若每張都長得跟某張 seed 一模一樣就是 overfit，要降 rank 或減 steps 重跑。

**M11 驗證（SDXL，Colab）**：本機 `--max-train-steps 1 --smoke` 跑通薄封裝並存出權重檔；
斷點續跑分支（有／無 checkpoint）各跑一次確認行為；notebook 內無明文 token；
峰值 VRAM 記錄進對應 LoRA validation report，以佐證需要 L4。

### 4.3 批次生成（M12）— `src/synthetic/generate_diffusion.py`

本機 4090 與 Colab 都能跑（**優先本機以省 Colab 額度**）。單張流程：

```
(clean image, placed mask)
  → 依 crop_ratio 把 mask bbox 外擴成正方形 crop_bbox
  → crop → resize 到 model_resolution (SD2 512 / SDXL 1024)
  → StableDiffusion(XL)InpaintPipeline(prompt=trigger token, image=crop, mask=crop mask,
      guidance_scale, num_inference_steps, generator=torch.Generator(seed))
  → resize 回 crop 原尺寸
  → 縫合回全解析度影像（cv2.seamlessClone 或羽化 alpha）
  → 最終 GT mask = 全解析度座標下的 placed mask（不做縮放近似）
  → 寫 metadata.jsonl 一行
```

CLI 需支援：`--object`、`--defect-type`、`--n`、`--bucket {original,searched}`、
`--device`、`--resume`（掃描既有輸出跳過已完成的 sample_id）、`--dry-run`。

### 4.4 Refine 搜尋（M12，對應課程 `sdg-refine`）

```
for each sample in original/:
    for run in 1..NUM_SEARCH_RUN:
        重抽 (guidance_scale, crop_ratio)  # 從設定的網格或分布
        重新生成
        算 refine_score = w1·nn_score_in_range + w2·seam_score
    取歷來最佳的一次 → 寫入 searched/
```

`nn_score_in_range` 表示「落在 [τ_low, τ_copy) 區間內得分最高」——太低不真實、
太高是複製品，兩端都要罰（見 [ADR-006](decisions.md#adr-006)）。
**`original/` 與 `searched/` 兩桶都保留**，兩桶都能進 Phase 2 的消融。

搜尋範圍初值（M12 依實測調整並記錄）：
- `guidance_scale` ∈ {5.0, 7.5, 10.0, 12.5}
- `crop_ratio` ∈ {1.8, 2.5, 3.5}（mask bbox 的外擴倍率）

M12 smoke 後鎖定 `boundary-gradient-v2`：`original` 使用 pipeline v0.5.0；
searched v0.6.0 把 candidate 0 固定為 original 的 guidance 7.5、crop ratio 2.5 與
candidate-index-0 seed，另外三次採 deterministic stratified schedule。四次合計保證
四個 guidance 與三個 crop ratio 全覆蓋，且搜尋分數不可能低於 original。評分權重為 mask 內可見變化 0.25、
相對 frozen background 的 mask-boundary Sobel excess gradient 0.65、clipped-pixel artifact
0.10。全解析度縫合使用 2px dilation + 3px Gaussian feather。這個分數只負責四選一，
**不取代 M13 的 near-copy、語意、文字與 seam 過濾**。

**M12 驗證**：`metadata.jsonl` 每行通過 schema 驗證；同 seed 重跑位元相同；
`original` 與 `searched` 並排 grid **自己看**，確認 refine 真的變好（沒變好就檢討 refine_score 權重）。
