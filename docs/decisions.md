# Decision Log (ADR)

> **只追加，不改寫。** 若某個決策被推翻，新增一筆 ADR 說明並把舊的標為 `Superseded by ADR-XXX`，
> 不要刪除或修改舊內容——決策的演變過程本身就是這個作品集的一部分。
>
> 格式：狀態 / 日期 / 脈絡 / 決策 / 後果 / 查證來源

---

<a id="adr-001"></a>
## ADR-001 — Inpainting 底模：SD2-inpainting 主線 + SDXL-inpainting-0.1 對照

**狀態**：Accepted ｜ **日期**：2026-07-27

### 脈絡
Stage B 需要一個開源 inpainting 底模，硬性條件是**授權必須允許把生成結果公開發佈到
Hugging Face**（本專案會上傳合成資料集）。候選：SD2-inpainting、SDXL-inpainting-0.1、
FLUX.1-Fill-dev、SD1.5-inpainting。同時受限於 Colab Pro 每月 100 compute units。

### 決策
**兩個都做**：`stabilityai/stable-diffusion-2-inpainting` 為主線，
`diffusers/stable-diffusion-xl-1.0-inpainting-0.1` 為對照。
**執行順序：SD2 全流程綠燈之後才動 SDXL**，避免額度先被吃光。

| | SD2-inpainting | SDXL-inpainting-0.1 |
|---|---|---|
| 原生解析度 | 512 | 1024 |
| UNet 參數量 | 865M | 2.6B |
| 授權 | CreativeML Open RAIL++-M | CreativeML Open RAIL++-M |
| LoRA 訓練 GPU | T4 (16GB) 可 | 需 L4 (24GB) |
| 角色 | 主線，把管線跑綠 | 「底模容量 vs 下游提升」消融 |

**選 SD2 當主線的理由**：512 原生解析度正好對上 [ADR-004](#adr-004) 的 crop-to-ROI patch
尺寸；UNet 小三倍，T4 就能訓，Colab 額度最省；生成 200 張/型的批次在本機 4090 上很快。

**排除**
- `black-forest-labs/FLUX.1-Fill-dev`：非商用授權，直接擋住「把輸出公開發佈」這件事
- `stable-diffusion-v1-5/stable-diffusion-inpainting`：原始 `runwayml` repo 已下架，
  現存的是社群轉傳版，授權鏈不夠清楚，不適合要公開發佈的專案

### 後果
- 兩本 Colab notebook（M10 SD2 / M11 SDXL），SDXL 那本要標明需要 L4
- Colab 額度預算會比單一底模高，M15 要把兩本的 compute units 估算分開寫
- README 的授權表要同時列兩個底模

### 待查證（M10 前重新確認）
- 兩個 HF repo 是否仍未 gated、授權是否變更
- CreativeML Open RAIL++-M 的當期條文（使用限制條款、有無新增營收門檻）
- `diffusers` 當時版本的 `StableDiffusionInpaintPipeline` / `StableDiffusionXLInpaintPipeline` API

### 來源
- [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
- [Stable Diffusion 商用授權與輸出權利說明](https://terms.law/ai-output-rights/stable-diffusion/)（RAIL 為 use-based restriction，無營收上限、無權利金）

---

<a id="adr-002"></a>
## ADR-002 — 瑕疵分型：非監督分群 → 目視確認命名 → 凍結

**狀態**：Accepted ｜ **日期**：2026-07-27

### 脈絡
課程方法論的核心之一是 **anomaly embedding LUT：一個瑕疵型一個可學習 embedding**
（簡報範例 `"Bottle+broken_large"`）。MVTec AD 有 per-type 資料夾，但——

**VisA 公開版沒有 per-image 的瑕疵類型標註**，只有 anomaly/normal 二元標籤與 pixel mask。
論文指出 pcb1 有 4 個 anomaly class、capsules 有 5 個，**但那是論文的描述，資料裡沒有標**；
且「每型只有 5–20 張、一張圖可能含多個瑕疵」。
再加上 few-shot 只取 10 張/物件，原始 prompt 的「每種瑕疵抽 K=10 張」**不可執行**。

### 決策
用**非監督分群產生 pseudo-type，再由使用者目視確認命名並凍結**。

1. 對每個 few-shot seed 的 GT mask 取連通元件（面積過小的雜點先濾掉）
2. 每個元件抽兩組特徵：
   - `facebook/dinov2-base`（Apache-2.0）對該元件 ROI crop 的 embedding
   - 形態特徵：面積比、長寬比、solidity、extent、離心率、同圖元件數、遮罩內外的亮度／對比差
3. 兩組特徵各自標準化後串接，用 Agglomerative clustering；`k ∈ [1..5]` 以 silhouette 選，
   **硬性條件：每群至少 3 個元件**
4. 每群產 contact sheet → **使用者確認或改名** → 凍結成 `splits/defect_types.json`（附 SHA256）
5. 型別名稱即 trigger token，例：`<pcb1-scratch>`、`<capsules-discolor>`

**Fallback**
- 任何群 <3 個元件 → 併入該物件的通用 `<obj>-defect`
- 可用型別 <2 → 該物件退回**單一 trigger token**，並在報告中註明

**紅線**：分群**只在 10 張 few-shot seed 上做，絕不讀 test**。實作時用
`splits/test_blocklist.json` 反查所有輸入路徑，命中即中止。

### 後果
- 實務上每個物件大概只會得到 2–3 個可用型別，不會是論文說的 4/5 個——**這要如實寫進 README**
- README 必須標示：這是「從 few-shot seed 推導的 pseudo-type」，**不是 VisA 官方標籤**
- 生成配額按型別的元件數比例分配（對應課程 `prep-testcase`）
- 這一步本身變成方法論的一節：「在沒有型別標註的資料集上如何復刻 anomaly embedding LUT」

### 來源
- [SPot-the-Difference (ECCV 2022) / VisA](https://arxiv.org/abs/2207.14315)：每型 5–20 張、一張圖可能多瑕疵
- [amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)：split CSV 欄位僅
  `object, set, label, image_path, mask_path`，`label` 只有 normal/anomaly

---

<a id="adr-003"></a>
## ADR-003 — few-shot 預算：k=10 瑕疵圖 + train pool 全部正常圖，四組完全相同

**狀態**：Accepted ｜ **日期**：2026-07-27

### 脈絡
spot-diff 官方 `2cls_fewshot` 協定是「20%/80% 切 train/test，再從 train pool
**兩個類別各**抽 k=5 或 10」。照字面走，訓練集只有 10 張正常 + 10 張瑕疵。
但 Stage A 的 copy-paste 與 Stage B 的 inpaint 都需要**大量正常圖當背景**：
只有 10 張背景時，合成資料會退化成同一張圖的變體，整個實驗會失去意義。
同時五組對照的公平性要求「所有組別的真實資料必須完全相同」。

### 決策
**瑕疵稀缺、正常充足**：

- 基底佈局用 spot-diff `--split-type 2cls_fewshot`（官方、可引用）
- 從 train pool 以 **seed=42 抽 k=10 張瑕疵圖／物件** = few-shot 瑕疵集
- train pool 的**正常圖全數保留**（pcb1 約 200、capsules 約 120）
- **第 1–4 組（Real-only / +Std Aug / +Unfiltered Syn / +Filtered Syn）吃完全相同的真實影像集合**，
  合成只作為增量
- **Full-real 上限組**改用 `2cls_highshot` 的 train split（約 60 張瑕疵／物件），
  並明確標示它是**不同（較大）的真實預算**，不是同一條件下的對照
- Validation 只從 train pool 切、只用真實資料；Test 原封不動

**理由**：AOI 產線上正常品幾乎免費，瑕疵品才是稀缺資源——這正是課程
「Lack of real defect images for training high quality AOI model」的敘事。
把 few-shot 定義在**瑕疵**這個軸上，比對稱地砍正常圖更貼近真實問題。

**防洩漏**：建立 test 影像的 SHA256 blocklist（`splits/test_blocklist.json`）；
`df-guard` skill 在任何生成／過濾／分型動作前檢查沒有任何路徑命中 blocklist。

### 後果
- README 必須明確寫出這個定義與理由，並註明**與論文的 k=10/k=10 協定不同**
- 若時間允許，可加跑一組嚴格 k=10/k=10 當附錄消融（Phase 2 選配）
- 兩個物件的正常圖數量不同（200 vs 120），per-object 指標要分開看

### 來源
- [SPot-the-Difference 論文](https://arxiv.org/abs/2207.14315)：1-class = 90%/10%；
  2-class high-shot = 60%/40%；2-class few-shot = 20%/80% 再抽 k=5,10
- VisA 影像數：pcb1 = 1004 normal / 100 anomaly；capsules = 602 normal / 100 anomaly

---

<a id="adr-004"></a>
## ADR-004 — 生成策略：crop-to-ROI → 模型解析度 inpaint → 全解析度貼回

**狀態**：Accepted ｜ **日期**：2026-07-27

### 脈絡
VisA 影像約 **1500×1000**，而且論文明指「VisA 的瑕疵明顯小於 MVTec AD」。
若把整張圖縮到 512 再 inpaint，瑕疵區域可能只剩幾十個像素，細節會被毀掉，
生成出來的東西不會像真實瑕疵。

### 決策
```
放置 mask
  → 取 mask 的 bounding box
  → 以 crop_ratio 外擴成正方形 patch（含足夠的周邊上下文）
  → resize 到模型原生解析度（SD2 512 / SDXL 1024）
  → inpaint
  → resize 回 patch 原尺寸
  → 以 Poisson (cv2.seamlessClone) 或羽化 alpha 縫合回全解析度影像
最終 GT mask = 全解析度座標系下的放置 mask（不做任何縮放近似）
```

`crop_ratio` 與 `guidance_scale` 就是 refine 階段的兩個搜尋維度，
直接對應課程 `sdg-refine` skill 的 `(guidance, crop_ratio)` 搜尋與
`original/` vs `searched/` 雙輸出桶。

### 後果
- 生成腳本必須同時保存 patch 座標與 `crop_ratio` 進 `metadata.jsonl`，才能重現
- 縫合品質本身要成為一道過濾規則（邊界梯度不連續度），見 [filtering_spec.md](filtering_spec.md)
- SD2 用 512 patch、SDXL 用 1024 patch，兩者的 `crop_ratio` 搜尋範圍要分開設定

---

<a id="adr-005"></a>
## ADR-005 — 儲存佈局：大檔在 D:、repo 在 C:

**狀態**：Accepted ｜ **日期**：2026-07-27（使用者決定）

### 脈絡
C: 剩 202 GB、D: 剩 1728 GB。本專案預估佔用 40–60 GB（VisA 原始 1.8GB tar +
解壓約 4GB + 合成影像 + runs）。專案資料夾在 `C:\Users\3Hml\Desktop`，
已確認**未**被 OneDrive 接管（否則 40GB 影像會被同步上雲）。

### 決策
- `data_root = D:\sdg-data\01-defectforge`：VisA 原始、`VisA_pytorch` 佈局、
  合成桶、`runs/`、embedding 快取
- **Hugging Face 快取沿用 C: 預設**（`C:\Users\3Hml\.cache\huggingface`，約 8GB）
- 專案資料夾（C:）只留：程式碼、設定、文件、`splits/`、`reports/` 小圖
- **任何腳本不得硬編絕對路徑**，一律讀 [`configs/paths.yaml`](../configs/paths.yaml)

### 後果
- repo 天生乾淨，發佈前不需要大掃除
- Colab 端的路徑對應（`/content/data`）要在 notebook 內另外處理，不共用 `paths.yaml` 的值
- 若之後換機器，只需改 `paths.yaml` 一個檔

---

<a id="adr-006"></a>
## ADR-006 — 生成品質指標的定義

**狀態**：Accepted ｜ **日期**：2026-07-27

### 脈絡
課程的 `eval` skill 描述為「Computes per-type **nn_score** (key KPI), **mnn_score**, and
**fid** via DINOv2 correspondence to compare real vs. generated anomaly images」，
但**沒有公開確切公式**。我們需要一個明確、可重現、寫得出來的定義。

### 決策
所有指標**在瑕疵 crop 上計算，不是整張影像**（整張圖被正常背景主導，指標會失去鑑別力）。
Embedding 一律用 `facebook/dinov2-base` 的 CLS token，L2 normalize 後算 cosine。

設 `R` = 真實瑕疵 crop 集合（僅 few-shot seed），`G` = 生成瑕疵 crop 集合。

| 指標 | 定義 | 用途與門檻 |
|---|---|---|
| `nn_score(g)` | `max_{r∈R} cos(emb(g), emb(r))` | **過低** → 不像真實瑕疵，拒絕；**≥ τ_copy** → 幾乎是 seed 的複製品，拒絕。兩個門檻都從真實 crop 的 leave-one-out 分布訂出來，寫進 `filtering_spec.md` |
| `mnn_score` | `R` 中「與至少一個 `g∈G` 互為最近鄰」的比例 | 覆蓋度／多樣性。只追求高 `nn_score` 會導致模式塌縮，`mnn_score` 是它的制衡 |
| FID | `clean-fid` on crops | 樣本數小時不可靠，僅列出供參考 |
| KID | `clean-fid` on crops | **小樣本情境的主要影像級指標** |

分瑕疵型（trigger token）各報一份，另報全體彙總。

**健全性檢查（M14 必做）**：把真實瑕疵 crop 自己餵進去，`nn_score` 應接近 1、
KID 應接近 0。不成立表示實作有錯，不准往下走。

### 後果
- 文件與 README 必須明寫：**這是我們的詮釋，NVIDIA 未公開確切公式**
- `τ_copy` 是可調參數，調整過程與最終值要記進 `reports/filter_report.md`
- 指標的輸入 crop 來源路徑必須全部通過 test blocklist 檢查

### 來源
- [NVIDIA/skills · physical-ai-defect-image-generation](https://github.com/NVIDIA/skills/tree/main/skills/physical-ai-defect-image-generation)
- 課程簡報 slide 8：Image based quality 的 detail 列出「Feature similarity: FID, MNN, ...」
