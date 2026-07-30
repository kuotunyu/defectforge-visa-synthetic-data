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

**狀態**：Accepted，但**基底 split 的部分已被 [ADR-007](#adr-007) 取代** ｜ **日期**：2026-07-27

> ⚠️ 本 ADR 的核心原則（k=10 瑕疵、正常圖充足、第 1–4 組真實資料完全相同）**仍然有效**。
> 但「基底用 `2cls_fewshot`、Full-real 上限用 `2cls_highshot`」這一段是**錯的**，
> 會造成 50% 的測試瑕疵洩漏進 Full-real 的訓練集。修正見 [ADR-007](#adr-007)。

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
解壓約 4GB + 合成影像 + runs）。專案 git checkout 位於 C:，
已確認**未**被 OneDrive 接管（否則 40GB 影像會被同步上雲）。

### 決策
- `data_root = D:\sdg-data\01-defectforge`：VisA 原始、`VisA_pytorch` 佈局、
  合成桶、`runs/`、embedding 快取
- **Hugging Face 快取沿用 C: 預設**（`%USERPROFILE%\.cache\huggingface`，約 8GB）
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

---

<a id="adr-007"></a>
## ADR-007 — 基底 split 改用 `2cls_highshot`，修正 ADR-003 的跨 partition 洩漏

**狀態**：Accepted ｜ **日期**：2026-07-27 ｜ **取代**：[ADR-003](#adr-003) 的基底 split 部分

### 脈絡
[ADR-003](#adr-003) 讓第 1–4 組用 `2cls_fewshot` 的 test 評測，Full-real 上限組卻用
`2cls_highshot` 的 train 訓練。**這兩個 CSV 是同一批影像的不同切法**，把兩者混用等於洩漏。

實測（下載兩份官方 CSV 後計算，不是推測）：

| | train | test |
|---|---|---|
| `2cls_fewshot` pcb1 | 201 normal / **20** anomaly | 803 normal / **80** anomaly |
| `2cls_highshot` pcb1 | 602 normal / **60** anomaly | 402 normal / **40** anomaly |
| `2cls_fewshot` capsules | 120 normal / **20** anomaly | 482 normal / **80** anomaly |
| `2cls_highshot` capsules | 361 normal / **60** anomaly | 241 normal / **40** anomaly |

```
highshot TRAIN(anomaly) ∩ fewshot TEST(anomaly) = 40   (兩個物件都是 40)
highshot TRAIN(normal)  ∩ fewshot TEST(normal)  = 401 (pcb1) / 241 (capsules)
```

→ 照 ADR-003 執行的話，**Full-real 上限組的訓練集包含了一半的測試瑕疵**。
那組數字會非常漂亮，而且完全是假的。

同時實測發現兩套切法是**巢狀**的：
```
fewshot TRAIN ⊂ highshot TRAIN     (True)
highshot TEST ⊂ fewshot TEST       (True)
```

### 決策
**以 `2cls_highshot` 為唯一基底 partition。**

```
k=10  ⊂  fewshot_train(20)  ⊂  highshot_train(60)      ← 三者皆與 highshot_test 互斥
                                                          （因為 fewshot_train ⊂ highshot_train）
```

| 項目 | 定義 | pcb1 | capsules |
|---|---|---|---|
| **Test（唯一，凍結）** | `2cls_highshot` test | 402 normal / 40 anomaly | 241 normal / 40 anomaly |
| **Train pool** | `2cls_highshot` train | 602 normal / 60 anomaly | 361 normal / 60 anomaly |
| **few-shot 瑕疵集** | 從 `2cls_fewshot` train 的 20 張中以 seed=42 抽 **k=10** | 10 anomaly | 10 anomaly |
| **正常圖（所有組共用）** | train pool 全部 normal | 602 | 361 |
| **Full-real 上限** | train pool 全部 anomaly | 60 anomaly | 60 anomaly |
| **中間點（免費附贈）** | `2cls_fewshot` train 全部 anomaly | 20 anomaly | 20 anomaly |

**額外收穫**：因為巢狀，我們免費得到一條「真實瑕疵 **10 → 20 → 60** 張」的縮放曲線，
可以回答「合成資料相當於多少張真實瑕疵」——這比原本只有單一上限點強得多，
而且 k=10 仍然是從官方 `2cls_fewshot` train pool 抽的，協定可引用。

### 後果
- **所有組別（含 Full-real）共用同一個凍結 test set**，完全可比、零洩漏
- Test 變小（每物件 40 張瑕疵，兩物件合計 80），統計變異較大 → per-object 指標要附樣本數，
  Real-only 與最佳 Filtered 組補 3 seeds 報 mean±std 的規定因此更重要
- 背景正常圖從 201/120 增加到 602/361，合成的背景多樣性明顯變好
- 類別極不平衡（pcb1 約 60:1）→ **所有組一律用相同的 class-balanced sampling**，不得只對某組調整
- `df-guard` 要新增一條斷言：**任何訓練集與 `highshot_test` 的交集必須為空**

### 驗證方式（M3 必跑）
重算上表所有數字並與本 ADR 逐格比對；重算 `highshot_train ∩ highshot_test = ∅`；
重算 `fewshot_train ⊂ highshot_train`。任何一項不符即停。

---

<a id="adr-008"></a>
## ADR-008 — SD2 LoRA 在本機 4090 訓練；訓練邏輯單一實作、notebook 只是薄封裝

**狀態**：Accepted ｜ **日期**：2026-07-27（使用者決定）

### 脈絡
本機執行規範的分工原則是「>30 分鐘的 GPU 訓練一律 Colab」。
但 SD2-inpainting 的 LoRA 訓練規模很小：UNet 865M、512 patch、只有 10 張訓練圖、
只訓 LoRA 與 token embedding —— 在 4090 上估計 20–30 分鐘，**在本機門檻之內**。

若照原計畫兩個底模都上 Colab，無人值守的自動執行只能跑到 M11（notebook 備好）就卡住，
M12 之後全部要等人跑完 Colab —— 這與「睡覺時讓 AI 按圖施工」的目標直接衝突。

### 決策
- **SD2 LoRA 在本機 4090 訓練**（合規：≤30 分鐘）→ M1→M14 整條 critical path 可無人值守
- **SDXL LoRA 上 Colab**（2.6B UNet、1024，超過本機門檻）
- **訓練邏輯只寫一份**：`src/training/train_inpaint_lora.py`，本機 CLI 與 Colab notebook
  **都呼叫同一支腳本**。notebook 只負責掛 Drive、解壓資料到 `/content/data`、讀 Secrets、
  組參數、呼叫腳本、同步 checkpoint。**不得把訓練迴圈複製一份到 notebook 裡**

### 後果
- Colab 額度只花在 SDXL LoRA 與 Phase 2 的 SegFormer，估計省下一半以上
- 本機訓練必須自己記錄實際耗時；**若實測超過 30 分鐘，就要回頭改回 Colab 並更新本 ADR**
- notebook 的 smoke test 因此變成「確認薄封裝能正確呼叫腳本」，而不是驗證訓練邏輯
  （訓練邏輯由本機的完整訓練直接驗證）

---

<a id="adr-009"></a>
## ADR-009 — 下游實驗設計：分類每物件一個二元模型；分割跑五組鐵律 + 四組來源消融

**狀態**：Accepted ｜ **日期**：2026-07-27

### 脈絡
Phase 2 的 prompt 對分割只列了「合成來源」四組（程序化-only／copy-paste／diffusion／混合），
沒有 Real-only 基準。但「程序化-only ＝ 零真實瑕疵也能做分割」要成為標題，
就**必須**有 Real-only 對照，否則沒有東西可比。
分類則要決定是「每物件一個模型」還是「跨物件單一模型」。

### 決策

**分類**：每物件一個二元分類器（good / bad），ConvNeXt-Tiny @384。
理由：VisA 官方的 2-class 設定就是 per-object 二元異常分類，可直接對照文獻；
兩個物件外觀差異極大，混在一起訓練會讓「哪個物件變好」變得不可解讀。
報告 per-object F1/AUROC，再取跨物件 macro 平均。

**分割**：SegFormer-B0 @512、Dice+BCE。跑 **5 組鐵律 + 4 組來源消融 = 9 組**：

| 組 | 訓練資料 |
|---|---|
| 1 Real-only | k=10 真實瑕疵 + 正常圖 |
| 2 + Standard Aug | 同上 + 傳統增強 |
| 3 + Unfiltered Syn | 同上 + 未過濾合成（全來源混合） |
| 4 + Filtered Syn | 同上 + 過濾後合成（全來源混合） |
| 5 Full-real 上限 | 60 張真實瑕疵 + 正常圖 |
| 6 程序化-only | **正常圖 + 程序化合成**，零真實瑕疵影像 |
| 7 copy-paste-only | k=10 + copy-paste 合成 |
| 8 diffusion-only | k=10 + Stage B 合成 |
| 9 全部混合 | = 第 4 組（不重跑，直接引用） |

SegFormer-B0 很小，T4 上估每組 20–40 分鐘、9 組約 5 小時 ≈ 9 CU，Colab 額度吃得下。

### 後果
- 分割的 Colab notebook 要能用 `--group` 參數跑任一組，並支援平行開多本
- 第 6 組是本專案最有記憶點的結果，`reports/` 要專門為它做一組視覺化
- 第 9 組直接引用第 4 組，因此只需跑 8 次

---

<a id="adr-010"></a>
## ADR-010 — 合成量掃描用絕對值 {125, 250, 500}，每物件生成 500 張

**狀態**：Accepted ｜ **日期**：2026-07-27

### 脈絡
Phase 2 的 prompt 寫「合成量 0.5x/1x/2x（相對真實正樣本數）」。
但真實正樣本只有 **10 張**，算出來是 **5 / 10 / 20 張**——這個刻度掃不出任何訊號。
課程 slide 8 自己展示的曲線是 **+125 / +250 / +500 張**（mIoU +0.0519 / +0.0676 / +0.0851）。

### 決策
改用**絕對值**，直接對上課程的刻度：

| 項目 | 值 |
|---|---|
| SD2 每物件生成總量 | **500 張** |
| 掃描點 | **{125, 250, 500}** |
| SDXL 每物件生成總量 | **250 張**（它是底模容量消融，不是主線） |
| SD2 vs SDXL 的對照點 | **250 張**（同量比較才公平） |
| Stage A（copy-paste / procedural） | 各 **500 張／物件** |
| 型別配額 | 500 依各瑕疵型的連通元件數**按比例**分配（課程 `prep-testcase` 規則）；任一型不足 50 張就補到 50 |

若曲線在 500 仍在上升，之後補跑到 1000 —— 生成腳本支援 `--resume`，
既有的 500 張不用重生。**不要為了「更完整」在第一輪就跑 1000 而讓過夜跑不完。**

### 後果
- 磁碟估算：SD2 500×2 物件 ×（original + searched）= 2000 張，SDXL 250×2×2 = 1000 張，
  Stage A 500×2×2 = 2000 張，合計約 5000 張全解析度 PNG ≈ **12–15 GB**
- 本機生成時間估算（4090）：SD2 約 2 s/張 → 2000 張約 70 分鐘；
  refine 搜尋 ×4 → 再加約 3.5 小時。SDXL 約 7 s/張 → 1000 張約 2 小時
- 報表的曲線圖要與課程的三點並排，方便對照敘事

---

<a id="adr-011"></a>
## ADR-011 —「程序化-only」的口徑：零真實瑕疵**像素**，但用了真實 mask 的**統計量**

**狀態**：Accepted ｜ **日期**：2026-07-27

### 脈絡
[synthesis_spec.md](synthesis_spec.md) 規定程序化合成的 mask 面積與長寬比要落在
真實 mask 分布的 5–95 百分位內。這用到了 **10 張 few-shot seed 的真實 mask 統計量**。
因此「零真實瑕疵」這個說法在字面上不成立。

### 決策
**保留這個設計（它讓合成更合理），但把口徑講清楚**，而不是假裝沒用到。

正式措辭（README 與報表一律照抄）：

> **Zero real defect pixels.** The procedural-only group never sees a single real defect
> pixel. It does use *aggregate shape statistics* (area ratio and aspect ratio percentiles)
> computed from the 10 few-shot training masks — that is the entire leakage surface,
> and it is disclosed here.

並提供 `--no-real-stats` 旗標跑一組**完全不看真實統計**的版本（用手訂的固定分布），
在報表中並列，讓讀者自己判斷那些統計量值多少。

### 後果
- `procedural.py` 必須支援 `--no-real-stats`
- 報表要同時列「用統計量」與「不用統計量」兩版的分割結果
- 這種主動揭露洩漏面的寫法本身就是加分項，不要為了標題好看而含糊

---

<a id="adr-012"></a>
## ADR-012 — 無人值守執行模式：跑到底，任何驗證失敗即停

**狀態**：Accepted ｜ **日期**：2026-07-27（使用者決定）

### 脈絡
使用者希望「去睡覺時讓 AI 按圖施工就能幫我們跑」。這需要明確界定 agent 在無人監督時
可以自己決定什麼、必須停在哪裡、以及停下來時要留下什麼。

### 決策
採**跑到底模式**：連續執行 M1 → M14，每個里程碑跑完對應的驗證欄，
**全綠才前進**。任何一項不過就**立刻停止**，寫一份 handoff 報告，不得自行降低標準、
不得跳過、不得「先繼續之後再回來修」。

規則重點：
- **可自己決定**：門檻校準、隨機種子以外的實作細節、重試暫時性錯誤、瑕疵型別的暫用命名
- **必須停**：任何驗證失敗、需要花錢、>2GB 下載、任何 push/發佈、Colab 訓練、
  磁碟不足、生成結果目視明顯異常
- **瑕疵型別命名不阻塞**：自動產生 `<pcb1-type0>` 之類的暫用 token 繼續跑，
  使用者醒來只改**顯示名稱**（不動 token 字串，因此不需要重訓）

### 後果
- 每個里程碑要標註「無人值守可跑 / 需要你在場」
- 需要 `docs/interfaces.md` 把每支腳本的 CLI 契約寫死，agent 才不會自己發明參數
- 每晚的執行結果落在 `reports/handoff/<date>.md`，早上一看就知道跑到哪、為什麼停

---

<a id="adr-013"></a>
## ADR-013 — Manifest 真正不可變；抽樣 sidecar 與 validation 開發／refit 契約

**狀態**：Accepted ｜ **日期**：2026-07-27

### 脈絡
M4 契約說 `split_manifest.json` 寫入 checksum 後「凍結不得修改」，舊 M5 契約卻要求把
`in_fewshot_seed` / `in_val` 回寫同一份 manifest，兩者無法同時成立。blocklist 舊驗證又把
「檔案數」當成「unique SHA256 數」；byte-identical 檔案會讓這個等式失效。

此外協定同時要求 validation 從 train pool 切出、early stopping 看 validation，以及正式
Full-real 上限仍使用 60 張 anomaly。若沒有分開開發與 final refit，這三項也互相矛盾。

### 決策
1. M4 manifest 只放不可變的來源、partition、hash、pHash group 與官方 fewshot pool membership。
   M5 選擇另存 `splits/fewshot_selection.json`，內含 manifest SHA256；manifest 永不回寫。
2. blocklist 的 `sha256` 是所有 test image 與 bad mask 的 unique hash 集合，另記
   `image_count`、`mask_count`、`unique_sha256_count`。驗證逐檔確認 membership，不假設三者相加
   必等於 unique 數。
3. validation 固定為每物件 × label 的 highshot train 10%（`floor`、至少 1 張），以 seed 42
   從**不在官方 fewshot train pool**的候選抽出，確保 k=10 seed 完整可用。
4. validation 只在 Real-only 開發階段用來凍結所有組共用的超參、early-stopping patience 與
   final optimizer steps。正式比較使用這組凍結設定 refit 完整 train pool，再評估唯一 test；
   因此 10 / 20 / 60 張真實 anomaly 的主實驗口徑保持不變。

### 後果
- M5 重跑不會破壞 `MANIFEST.sha256`，任何 selection 都能明確追溯到某一版 manifest
- blocklist 即使遇到相同內容檔案仍可正確驗證
- validation 不接觸 test、不侵蝕 few-shot seed，又能避免為 Filtered 組單獨調參
- 訓練腳本必須明確區分 `development` 與 `final_refit` 模式，結果表也要記錄該欄

---

<a id="adr-014"></a>
## ADR-014 — SD2 原 repo 不可用，改採鎖定 revision 與 LFS hash 的保存 mirror

**狀態**：Accepted ｜ **日期**：2026-07-27

### 脈絡
M10 開工時重新查證 ADR-001 指定的
`stabilityai/stable-diffusion-2-inpainting`，Hugging Face API 回傳 404，且本機沒有可用 cache。
直接讓 `from_pretrained` 跟隨任意社群 repo 的 `main` 會失去供應鏈可重現性；放棄 SD2 又會
推翻已完成的 512 crop 主線與 SD2/SDXL 消融。

### 決策
採用 `sd2-community/stable-diffusion-2-inpainting`，它的 model card 明確說明是已下架原模型
的 preservation mirror。所有 M10/M12 行程固定：

- revision：`5f74973cbb64c8568780732c17f43eb269d63a0d`
- `text_encoder/model.safetensors`：
  `cce6febb0b6d876ee5eb24af35e27e764eb4f9b1d0b7c026c8c3333d4cfc916c`
- `unet/diffusion_pytorch_model.safetensors`：
  `9bcbb17f54b039f58bf78677fab8cd8a35dd686f6c9dd553e3646a8b0aaff41a`
- `vae/diffusion_pytorch_model.safetensors`：
  `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`

訓練器在載權重前用 Hugging Face API 驗證 immutable revision 與以上三個 LFS SHA256；
任一不符即 fail closed。模型授權仍為 CreativeML Open RAIL++-M。

### 後果
- ADR-001 的「SD2 作為主線」不變，只替換不可取得的託管來源
- README、publish spec、訓練 config 與 metadata 一律使用 mirror ID + exact revision
- 已下載的 Windows degraded cache 不是信任根；revision 與 LFS hash 才是
- 如果 mirror 消失，可從任何內容相同的來源恢復，但三個權重 hash 不得改

### 來源
- [Preservation mirror model card](https://huggingface.co/sd2-community/stable-diffusion-2-inpainting)
- [Diffusers Stable Diffusion inpainting API](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint)
- [Diffusers LoRA training guide](https://github.com/huggingface/diffusers/blob/main/docs/source/en/training/lora.md)

---

<a id="adr-015"></a>
## ADR-015 — Refine 搜尋必須包含 byte-reproducible original baseline，分數不得退步

**狀態**：Accepted ｜ **日期**：2026-07-27

### 脈絡
M12 最初的 deterministic stratified schedule 每筆抽四個不同的
`(guidance_scale, crop_ratio)`，雖能覆蓋四個 guidance 與三個 crop ratio，卻不保證
包含 original 的 `(7.5, 2.5)`。Formal searched 的前 111 筆稽核發現 97 筆分數提高、
5 筆相同、9 筆反而下降；平均差仍為正並不能掩蓋逐筆退步。這也表示
`original` / `searched` 配對消融沒有「搜尋至少不比基準差」的基本保證。

### 決策
- searched candidate 0 固定為 original 的 guidance 7.5、crop ratio 2.5 與
  candidate-index-0 generator seed；其評分證據必須逐欄等於 original candidate 0。
- 另外三個候選以 deterministic greedy coverage 選取；四個候選合計仍覆蓋四個
  guidance 與三個 crop ratio，且 parameter pair 不重複。
- original bucket 保留 pipeline v0.5.0；修正排程的 searched bucket 升為 v0.6.0，
  不混用或 resume 舊版 sidecar。
- 獨立 validator 要求 selected score 等於四候選最大值且不低於 original；若 selected
  candidate 是 0，輸出影像 SHA256 必須等於 original。

### 後果
- 每筆 searched 在目前 heuristic 下保證單調不退步，同時保留完整搜尋維度覆蓋。
- 每筆少一個純探索候選，換取可解釋且可驗證的 baseline safety。
- 問題排程產生的 formal prefix 移到 D: 的隔離目錄，不納入 canonical metadata；
  原始失敗數字保留在本 ADR 與 worklog，不用挑圖或改寫歷史掩蓋。

---

<a id="adr-016"></a>
## ADR-016 — M13 以 real leave-one-out 校準 DINO 門檻並發布全域 hardlink views

**狀態：Accepted**（2026-07-27）

### 背景
M13 同時過濾 copy-paste、procedural 與 searched SD2 三個來源。若每個來源各自校準，
品質分數無法橫向比較；若複製 accepted payload，則 3,000 張 1500×1000 PNG 會重複佔用
大量磁碟。正式規模測試也顯示，將所有 full-resolution masks 常駐 RAM 會與同機另外兩個
專案競爭資源。

### 決策
- 所有來源共用同一組、按 object 校準的 real few-shot reference：
  DINOv2 CLS embedding L2 normalize 後，以 real leave-one-out nearest cosine 的第 5
  百分位作 `tau_low`，以 real-to-centroid cosine distance 的第 95 百分位作
  `tau_outlier`；`tau_copy=0.98` 保持文件鎖值。
- 六道規則順序固定為 ROI → area → aspect → pHash → DINOv2 → seam；
  pHash 僅把前面所有規則與非 pHash 規則最終通過的 earlier sample 加入 accepted set。
- `${synthetic}/unfiltered` 保存全部決策，`${synthetic}/filtered` 僅保存 accepted；
  image/mask 以 NTFS hardlink 發布，metadata 以 atomic replace 發布。
- generated embeddings 以 model revision、crop ratio、輸入 ID 與檔案大小組成內容鍵；
  full RGB background LRU 限制 32 張，mask 每個 chunk 重讀後釋放。

### 後果
- 三個 generator 的門檻可直接比較，M14 可直接以同一 accepted/unfiltered membership
  建立評估組。
- hardlink 節省磁碟，但發布與驗證需在支援 hardlink 的同一 volume；跨 volume 時
  fail closed，不自動退化成 copy。
- strict real p05 與語意門檻可能讓個別 generator/type 為零；這是被保留並報告的品質
  結果，不以覆蓋率為由事後放寬。

---

<a id="adr-017"></a>
## ADR-017 — M14 以 mask-centered crop、未偏 KID 主報與 biased identity sanity

**狀態：Accepted**（2026-07-27）

### 背景
M14 必須在 35 張 real component crops 對 3,000 張 generated crops 的小樣本條件下，
同時衡量單張相似度、流形覆蓋與集合分布。clean-fid 0.1.35 的直接 FID 實作又與
SciPy 1.17 的 `sqrtm` API 不相容。第一次正式 sanity run 也證明：未偏 KID
U-statistic 用同一有限集合對自身比較時，會因排除 kernel 對角線而得到
`-0.0399` 到 `-0.1270`，不是應為 0 的 identity check。

### 決策
- 所有指標只看 ratio 2.5 的 mask-centered defect crop，不看 whole image。
- DINOv2 CLS 固定既有 revision；`nn_score` 取 generated-to-real 最大 cosine，
  `mnn_score` 報 real references 參與 mutual-NN pair 的比例。
- 正式 generated-vs-real 表以 clean-fid clean-mode Inception feature 計算
  deterministic unbiased degree-3 polynomial KID；KID 為主，FID 僅參考。
- real-self / noise sanity 使用包含 kernel 對角線的 biased polynomial MMD；
  同一有限集合對自身精確為 0。正式 KID estimator 不因 sanity 修正而變更。
- FID 用 centered low-rank factors 的 nuclear-norm 恆等式精確計算；單元測試與
  標準 covariance sqrtm 公式相符，不呼叫已失效的 clean-fid FID wrapper。
- copy-paste / SD2 的 `type0/type1` 對應 frozen real type；procedural 的
  `crack/perlin/scratch/spot` 沒有等價 real taxonomy，只能標
  `real_scope=object_all`。空群組照實列出。

### 後果
- sanity 4/4 通過：real-self NN/mNN 為 1、biased KID 為 0、FID 約 0；
  deterministic noise 的 NN 均低於 `tau_low` 且 KID/FID 大幅增加。
- 44 列正式結果可比較 filtered/unfiltered，但不能把 FID 當小樣本主結論。
- `pcb1/copy-paste/type1` filtered 為空仍保留 `status=empty`，不補樣本、不改 taxonomy。
- 每次 run 先雜湊全部 provenance，且在模型載入前檢查同機 existing VRAM；
  test blocklist 命中或資源上限超標皆 fail closed。

---

<a id="adr-018"></a>
## ADR-018 — M11 延伸同一 trainer 支援 SDXL 雙文字編碼器，Colab 使用最小資料 bundle

**狀態：Accepted**（2026-07-27）

### 背景
既有 M10 trainer 固定為 SD2 的單 tokenizer／單 CLIP text encoder。SDXL 若只換
model ID 與解析度，會缺少第二組 token conditioning、pooled projection 與 time IDs；
這種 smoke 即使勉強啟動也不是正確的 SDXL 訓練。另一方面完整 VisA tar 為 1.8 GB，
M11 實際只需要 20 張 frozen few-shot image/mask 與四個 held-out placement。

### 決策
- `train_inpaint_lora.py` 保持單一訓練迴圈，以 `model.family` 分支 conditioning：
  SD2 schema/version 保持 0.2.0；SDXL 使用 0.3.0。
- SDXL 同時載入兩個 tokenizer/text encoder，兩者各訓一份 TrainableTokens adapter；
  UNet 收到兩個 penultimate hidden states 的 concatenation、第二 encoder 的 projected
  pooled embedding，以及 `[1024,1024,0,0,1024,1024]` time IDs。
- SDXL 鎖定 public model revision `115134f...e41e` 與兩個 text encoder、UNet、VAE
  四個 LFS SHA256；dry-run 必須先通過遠端 metadata 與 frozen data guards。
- notebook 只負責 L4 preflight、Drive、Secrets、`uv sync`、fresh/resume 路由與
  validator，不複製任何 loss／optimizer／scheduler 程式。
- Colab bundle 只帶實際會讀到的來源，另寫 archive SHA256 manifest；不包含 `.git`、
  `.env`、test data 或整份 VisA。

### 後果
- M10 已完成的 SD2 checkpoints、validator 與 pipeline version 不需重訓或改寫。
- M11 在使用者操作前可完成 model-lock dry-run、notebook 靜態驗證與資料交接；
  超過 2 GB 的 SDXL 權重下載與真正 smoke 仍需使用者明確同意。
- M11 不得因 structural tests 通過就勾選；必須等 real-model smoke、resume、reload
  與 peak VRAM 證據全齊。

---

<a id="adr-019"></a>
## ADR-019 — M15 驗收已完成的 Phase 1；M18 才建立分割 Notebook 交接

**狀態：Accepted**（2026-07-27）

### 背景
原始 M15 驗收文字要求 SDXL 與 SegFormer 兩本 notebook 都先具備五項操作資料，但
SegFormer notebook 的建立、`--group` 實作與本機 smoke 明確屬於 Phase 2 的 M18。
同一份 PLAN 又規定 Phase 1 全綠前不准開始 Phase 2，因此形成
「M15 等 M18、M18 等 M15」的循環依賴，無法在不違反規格的情況下前進。

### 決策
- M15 驗收 M0–M14 的凍結證據、M11 SDXL notebook 的五項完整交接、Phase 2 協定已
  事先凍結，以及 M0–M15 的 commit／Contributor 完整性。
- M18 負責建立 SegFormer notebook、跑本機 1-step smoke，並備妥五項
  **具體**操作資料；M19 才要求使用者上 Colab。
- M11 未在執行前記錄 Colab CU，因此只報實測 wall time、training time 與 peak VRAM，
  CU 明列 unavailable，不以第三方費率倒推成假精確數字。
- `scripts/verify_phase1.py` 是 M15 的 CPU-only 獨立驗收入口；它同時逐 commit 驗證
  author／committer 只有 `kuotunyu`，且禁止 `Co-Authored-By` trailer。

### 後果
- Phase 1 可以在不偷跑 Phase 2、也不捏造不存在 notebook 的前提下誠實關閉。
- 使用者目前不需做 Colab 操作；等 M18 的 notebook 與 smoke 全綠後，才會收到確切
  檔名、Runtime、Secrets、成本估計與下載清單。
- 未來若 M18 的實作改變資源需求，交接內容會依實測更新，不受 M15 的猜測綁死。

---

<a id="adr-020"></a>
## ADR-020 — M16 預註冊 38 個 canonical run，Real-only 開發階段完全不載入 test

**狀態：Accepted**（2026-07-27）

### 背景
Phase 2 原協定寫「約 40 run」，但 `real_60` 與五組中的 `full_real`、`syn_500` 與
主線 `filtered_syn`、`base_sd2` 與 refine 的 `bucket_searched` 實際是相同資料。
重跑相同 alias 只會浪費 GPU，還可能因隨機性產生看似獨立的偽重複。另一方面，
「最佳 Filtered 組補 3 seeds」若等看完 test 才選，會把 test 變成模型選擇集。

### 決策
- ConvNeXt-Tiny 固定 `timm/convnext_tiny.fb_in1k` revision
  `b43a6303c9fcf176d2d707478a128c2c91e93528`，`model.safetensors` SHA256
  `08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276`，
  384×384、每物件二元分類。模型載入後才把 1000-class head 換成 2-class head。
- `--mode development` 只允許 `real_only`，只載 frozen real validation，資料結構中
  test 清單就是空的；由兩物件共同 validation 凍結一套 learning rate、weight decay、
  total steps 與 patience，之後所有合成組不得單獨調參。
- `filtered_syn` 事前指定為三 seed 的主結果，不依 test 排名更換。`real_only` 與
  `filtered_syn` 各補 seeds 43、44。
- alias 不重跑：`real_60 → full_real`、`syn_500 → filtered_syn`、
  `base_sd2 → bucket_searched`。seed 42 跑 15 個 canonical group × 2 物件，再加
  8 個補 seed run，合計 **38 個實跑**，仍符合原定「約 40」。
- 主線 filtered/unfiltered 與量掃描各用 deterministic source×type 分層抽樣；
  source ablation 使用各來源原始 500 張，避免 pcb1 copy-paste 過濾後只剩 84 張造成
  數量混淆。refine 與 SD2/SDXL 底模消融都固定原始 diffusion 250 張，隔離生成器差異。
- 所有組固定 optimizer steps 並使用相同 label-balanced sampler；報告實際抽到的
  real-good、real-bad、synthetic-bad exposure，不能只報資料集大小。

### 後果
- `results/classification.csv` 只收 formal test 結果；development 與 smoke 留在 D 槽
  run 目錄，不會在調參時意外看到 test。
- `base_sdxl` 在 `stageB_sdxl/searched` 尚未產生時 fail closed，不得拿 SD2 或 Colab
  sample panel 代替；這是正式 M16 前必須補齊的 GPU 前置工作。
- Standard Aug 組只改訓練 transform，不增加真實影像：random resized crop、水平翻轉、
  小角度 affine 與輕量 color jitter；其他組用 deterministic resize，差異可被獨立解釋。

### 來源
- [timm Quickstart](https://huggingface.co/docs/timm/main/quickstart)
- [timm model factory reference](https://huggingface.co/docs/timm/reference/models)
- [locked ConvNeXt-Tiny model card](https://huggingface.co/timm/convnext_tiny.fb_in1k)

---

<a id="adr-021"></a>
## ADR-021 — M18 鎖定 SafeTensors SegFormer-B0、固定步數與每物件最小 Colab bundle

**狀態：Accepted**（2026-07-27）

### 背景
M18 已凍結為 SegFormer-B0 @512、Dice+BCE、九組分割比較，但尚未決定可重現的公開
checkpoint、binary head 口徑、optimizer-step 預算與 Colab 交接方式。若每一組在 Colab
重新從完整 D 槽資料選樣，不只要上傳所有合成池，也無法證明 Notebook 讀到的 500 張
就是本機預註冊的同一批。

### 決策
- 基底鎖定 `nvidia/segformer-b0-finetuned-ade-512-512` immutable revision
  `489d5cd81a0b59fab9b7ea758d3548ebe99677da`；只載入 SafeTensors，
  `model.safetensors` SHA256
  `6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9`。
- 保留 pretrained encoder 與 MLP decode head，只把 ADE20K 的 150-class 最後
  `1×1` classifier 重建為一個 defect logit。loss 在 shared trainer 中明確計算
  `Dice + BCEWithLogits`，不依賴 Transformers 內建多類別 CrossEntropy。
- 輸入 512、batch 4、learning rate `6e-5`、weight decay `0.01`、500 個
  optimizer steps、50-step warmup；`6e-5` 來自官方 SegFormer semantic
  segmentation recipe。這套設定在讀 test 前即凍結，八個 formal group 完全共用，
  不依 synthetic group 的結果再調參。
- normal image 在載入時建立同尺寸全零 mask；VisA 的 instance-valued anomaly mask
  與所有 synthetic mask 都用 `>0` 轉成 binary target。paired crop／flip／affine
  同步作用於 image 與 mask，color jitter 只作用於 image。
- 正式實跑八組 × 兩物件；`all_mixed → filtered_syn` 是第九組的引用，不建立另一個
  run。`procedural_only` 的 train set 只含真實 normal + 500 張程序化 synthetic，
  validator 必須證明 real-defect train image 數為 0。
- Colab 使用一份 source ZIP + 每物件一份 data ZIP。package script 先在本機展開並
  雜湊正式 group，再把精確 sample ID 寫入 bundle；Notebook 只把這些 ID 注入
  `sample_ids_by_object`，不在 Colab 重新抽樣。訓練資料先解壓到 `/content`，
  checkpoint 才同步到 Drive。
- 每個 run 保存 portable `data_manifest.json`、`run_config.json`、raw
  `training_report.json` 與 final SafeTensors。M20 必須從這些 raw report 重建
  `results/segmentation.csv`，不把 notebook 輸出文字當資料來源。

### 後果
- M18 model lock 只有約 15 MB，且是安全的 SafeTensors；不需要 Colab Secret。
- packaged selection 可以比完整 synthetic pool 小很多，但仍保留原影像位元、
  mask、來源 view、train-side provenance 與凍結 test inventory。
- 本機 one-step smoke 完成前不得把 Notebook 交給使用者；smoke 後再記錄實測 peak
  VRAM、時間與最後五項交接。

### 來源
- [Transformers image segmentation recipe](https://huggingface.co/docs/transformers/main/tasks/semantic_segmentation)
- [locked SegFormer-B0 checkpoint](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/tree/489d5cd81a0b59fab9b7ea758d3548ebe99677da)
- [locked SafeTensors file and SHA256](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/blob/489d5cd81a0b59fab9b7ea758d3548ebe99677da/model.safetensors)

---

<a id="adr-022"></a>
## ADR-022 — M22 展示模型使用固定事後排序，不能改寫正式結果

**狀態：Accepted**（2026-07-28）

### 背景
M22 要求用「最佳」分類器與分割模型做本機 Demo，但 M16 的主線比較已事前指定
`filtered_syn`，不能因看到 test 後把較高分組改稱主結果。另一方面，要求使用者手動從
數十個 D 槽／Colab run 猜 checkpoint 路徑，容易選到 smoke、alias 或不同物件模型。

### 決策
- Demo 選模是正式評估完成後的展示行為，**不改變** M16/M20 表格、主線組或結論。
- 每物件只納入 seed 42 的 formal physical runs。分類依
  Macro-F1 → AUROC → run name；分割依 Dice → AUPRO → run name 排序。
- 分割 `all_mixed → filtered_syn` 是 logical alias，不作第二個 physical candidate。
- 自動選模前要求完整 38-row classification CSV 與 18-row segmentation CSV，
  並把入選 row 綁回 raw `training_report.json` 的 object、seed、canonical group、
  run signature、metrics 與 SafeTensors SHA256。
- `--object` 模式保存 CSV／report／model hashes，但不把本機絕對 checkpoint 路徑寫入
  tracked evidence。顯式 `--cls-ckpt`／`--seg-ckpt` 仍保留給診斷用途。

### 後果
- M22 的 UI 可用一個物件參數重現，不需要人工作業挑路徑。
- 若 M20 缺檔、CSV 不完整、alias 被重跑、row 與 raw report 不一致或模型 hash 改變，
  Demo 在配置 GPU 前 fail closed。
- 排名使用 test 指標只服務展示模型，因此必須標記
  `selection_is_post_evaluation_demo_only=true`，不得拿選模後分數另立新實驗結論。

---

<a id="adr-023"></a>
## ADR-023 — 凍結 JSON 保存原始位元，Colab source bundle 使用固定白名單

**狀態：Accepted**（2026-07-28）

### 背景
M24 的全新環境驗證發現 Windows 工作樹中的 frozen JSON 是 CRLF 位元，但既有
`.gitattributes` 會在 Git blob／GitHub Source ZIP 中轉為 LF。正式 run report、
checksum sidecar 與 blocklist 都已引用原始 CRLF SHA256；因此公開 archive 的
`verify_splits.py` 會 fail closed。另有一個獨立問題：M18 packager 依賴
`git ls-files`，導致不含 `.git` 的 GitHub Source ZIP 無法再生 Colab bundle。

### 決策
- `split_manifest.json`、`defect_types.json`、`fewshot_selection.json`、
  `test_blocklist.json`、`source_checksums.json` 與 `splits/*.sha256` 一律標成
  `binary`，禁止 Git 的 text conversion、diff 與 merge；Git 必須保存已簽發證據
  的原始位元，不得隱式轉換換行。
- 不重新正規化 frozen JSON，也不改寫既有正式 report 的 hash；修正的是 Git 儲存
  行為，而不是重新定義已發行的證據。
- M18 source bundle 不再依賴 Git metadata，改用固定 root files 與
  `configs/`、`notebooks/`、`scripts/`、`splits/`、`src/` 白名單。
- 白名單內仍 fail closed：拒絕 symlink、秘密／credential 命名、Git internals
  與大於 10 MiB 的單檔；忽略 `.gitkeep`、bytecode 與 `__pycache__`。

### 後果
- `git clone`、GitHub Source ZIP 與普通解壓目錄都能重現相同的最小 M18 source
  bundle，不會夾帶 reports、results、`.venv` 或模型權重。
- staged-tree archive 的 `split_manifest.json` SHA256 恢復為
  `3d3c385cf0ff78479ecf90b4faf25fc07c88830e043616fb15aefb1282983e8c`；
  在全新 Python 3.12 `.venv` 中，166 tests、`verify_splits.py` 與兩物件 M18
  package dry-run 全部通過。
- source bundle 為 97 檔、2,330,613 uncompressed bytes；兩個 data bundle 的
  selection SHA 與 training blocklist 零命中契約保持不變。

---

<a id="adr-024"></a>
## ADR-024 — M24 必須驗完整 evidence graph，M22 GIF 由 frozen test 真模型輸出產生

**狀態：Accepted**（2026-07-28）

### 背景
原 `verify_publish.py` 只驗必備路徑、安全掃描與 Git 身分；只要放入任意
`results/segmentation.csv` 與 `assets/demo.gif` 就可能通過，沒有證明 M19–M23、
README 數字、raw report、圖表、授權或 HF bundle 彼此仍一致。M22 也只有互動 UI，
缺少可重跑且能證明 GIF 來自正式模型／frozen test 的產生流程。

### 決策
- M24 gate 必須要求 M0–M23 canonical PLAN rows 全勾、README 無 TBD／pending、
  12 份 final evidence 全部 `status=passed`，並重算 CSV、README、license documents、
  figures、demo selection／GIF 與 visual review 的 SHA256。
- 上游模型 visibility／gated／license 報告必須在 24 小時內；HF upload evidence
  只接受 `mode=dry_run`、不改 visibility、不建立或更新 private repositories。
- 正式 PNG 必須可完整解碼且至少 320×180；GIF 必須是至少兩幀的有效動畫，
  且全 repo 10 MiB tracked-file 上限仍適用。
- `record_demo_artifacts.py` 先選完並綁定兩物件 formal checkpoints，才檢查／配置 CUDA；
  每物件從 frozen highshot test 固定取一張 normal、一張 anomaly，產生四幀
  input／probability／mask／heatmap／latency GIF 與 array-level hash evidence。
- 目視 review 只有在逐張打開全部 final media 後，顯式傳入 confirmation 與觀察 note
  才能保存；報告綁定當下每個媒體 SHA。
- 一頁 acceptance report 只能在除自身外所有 local gate 通過後產生。任何缺檔或
  false check 都拒寫；external GitHub／HF 寫入與 visibility 仍等待使用者明確同意。

### 後果
- M24 incomplete 變成可定位清單，不會把「未發現錯誤」誤當成完成證據。
- M19 尚未回收時，gate 應明確停在 M19–M23、六份 downstream evidence、四張正式圖、
  README、GIF 與 acceptance；不得用 dummy segmentation 或單幀占位 GIF 解鎖。

---

<a id="adr-025"></a>
## ADR-025 — 公開 Demo 採 CPU-first、hash-bound Space；v2 與 v1 結果分流

**狀態：Accepted**（2026-07-28）

### 背景
v1.0.0 已公開 dataset、LoRA 與可重現的本機 Demo，但作品集訪客仍必須自行 clone、
準備四個本機 checkpoint 才能互動。使用者也同意在不改寫 v1 負面結果的前提下，
追加一輪「為什麼合成資料失敗、能否低成本修正」的 v2 實驗。

M16 raw evidence 顯示，500 張合成瑕疵加入後，單一 1,600-sample schedule 中真實瑕疵
曝光量從 real-only 的 818 次降到 9–14 次，合成瑕疵則達 769–798 次。這是明確且可
在 validation 上測試的 domain-imbalance 假說，不需要重新生成影像或先租 A100。

### 決策
- 公開 Demo 使用 Hugging Face Gradio Space；部署包不進 GitHub，而由白名單 packager
  從四個 formal checkpoint、raw report 與固定授權檔組成。
- 每個模型載入前驗大小與 SHA256；任何缺件、tamper、物件／group/report 不一致即拒絕服務。
- runtime 預設自動選 CPU/GPU，但正式公開先用 CPU Basic。實測本機 CPU 的單張四模型
  lazy-load 後推論約 0.3–0.4 秒，因此不為展示常駐付費 GPU；只有遠端實測證明不足才升級。
- Space 不保存上傳影像、private API 關閉、單併發、20 MB 上限、25 MP 解碼上限。
- `nvidia/segformer-b0-finetuned-ade-512-512` 的 upstream license 僅允許非商業研究／
  評估，因此 Space、README、完整 license copy 都要明示「非 production AOI」。
- v2 先做 domain-balanced／curriculum pilot，只以 development validation 選策略。
  pilot 未在兩物件均不退步且至少一項主要指標改善，就停止而不開正式 A100 run。
- v1 的 CSV、圖表、release 與結論凍結；所有 v2 產物另存 `results/v2/`、`reports/v2/`，
  並標明 exploratory pilot 或 confirmatory run。

### 後果
- 訪客可直接操作正式模型，同時看到 checkpoint provenance 與誠實的研究用途界線。
- GitHub 不追蹤約 243 MiB 權重，Space bundle 也不會夾帶 D 槽路徑、token 或原始 VisA。
- Colab 的付費 GPU 額度保留給 M27；M26 可先用短程本機 4090 或 L4，並以 fail-fast
  validation gate 控制成本。

---

<a id="adr-026"></a>
## ADR-026 — M26 只修正 anomaly domain exposure，先提交預註冊再執行

**狀態：Accepted**（2026-07-28）

### 背景
M16 的 `WeightedRandomSampler` 只平衡 good／bad 兩個 label；bad 類別內仍按樣本數均勻。
`filtered_syn` 有 10 張 real bad 與 500 張 synthetic bad，因此 1,600 次抽樣中，
真實瑕疵實際只曝光 9–14 次，遠低於 real-only 的約 800 次。這個機制足以解釋
synthetic run 的 domain collapse，先換大模型或重新生成會同時改太多變因。

### 決策
- v1 的 `class_balanced` sampler 保持預設且位元行為不變。
- 新增 `domain_balanced`：good 固定占 50%；bad 的 50% 再依
  `real_bad_share` 分成 real／synthetic。pilot 比較 0.50 與 0.75。
- 使用同一 ConvNeXt、filtered 500、seed 42、100 steps、learning rate、augmentation
  與 frozen real-only validation；總共 4 candidates × 2 objects = 8 development runs。
- 原 v1 安全閘仍禁止 synthetic development；M26 必須顯式傳入
  `experimental_synthetic_development`，此欄位寫進 run signature 與 report。
  development integrity guard 仍要求 test list 為空、validation 全為真實 train-side holdout。
- 候選只在 `domain_balanced_50/75` 中依兩物件 mean Macro-F1、mean AUROC 排序。
- 進 M27 的 gate 在執行前固定為：
  - 每物件 Macro-F1 不得比 real-only 低超過 0.01；
  - 每物件 AUROC 不得比 real-only 低超過 0.02；
  - 兩物件 mean Macro-F1 至少提升 0.01。
- 任一條失敗就記錄 `stopped`，不得用 test 或追加 candidate 救結果；若要新假說，
  必須另開新的、先提交的實驗版本。

### 後果
- M26 只回答「保留真實瑕疵曝光能否修正 v1 mixing」，因果解讀比換模型清楚。
- 8 個 run 預估各約 30–45 秒、峰值約 3.2 GiB；不需 A100。
- `configs/classifier_v2_pilot.yaml`、runner、gate tests 與本 ADR 必須先形成乾淨 commit，
  才可執行 `--execute`，避免看到 validation 後改門檻。
- GIF 可由同一套正式 checkpoint selector 重現，且不需開 share URL 或人工剪貼
  notebook／UI 數字。

---

<a id="adr-027"></a>
## ADR-027 — M39 併列揭露 seed 變異與 Dice／AUPRO 方向矛盾，但不更換預註冊主指標

**狀態：Accepted**（2026-07-29）

### 背景
公開 README 只呈現 seed 42 的單次結果，也只用 Dice 宣告 Segmentation 的負面結論。
重新稽核 `results/*.csv` 的原始欄位後，發現兩個已存在但未被呈現的事實：

1. `results/classification.csv` 已含 `real_only` 與 `filtered_syn` 各 3 個 seed
   （ADR-020 預註冊），但 README 從未呈現 mean ± std。實際變異很大：
   `filtered_syn` 的 seed 間標準差比 `real_only` 高一個數量級以上，
   單一 seed 的細部差異不足以支撐結論。
2. 16 個實跑的 Segmentation run 中有 6 個在固定 threshold 0.5 下 Dice 恰為 0
   （整張預測為背景），其中 3 個的 pixel AUROC 仍達 0.80 以上（最高 `0.9015`）。
   代表機率圖有訊號，是固定 threshold 把 Mask 切成全黑。
   同一批 run 的 threshold-free AUPRO 給出與 Dice **相反**的方向：
   `filtered_syn` 相對 `real_only` 的平均 Dice 為 `-0.2264`，平均 AUPRO 為 `+0.1046`。

只報 Dice 會讓讀者以為結論是單一方向；改報 AUPRO 則會踩到「因為結果不好看就換指標」
這條誠實性紅線。兩者都不可接受。

### 決策
**兩個指標併列揭露，主結論不變。**

- `RESULT_OUTCOME` 的 `classification_negative` / `segmentation_negative` 仍由
  預註冊的 Macro-F1 與 Dice 決定，措辭與數值一字不改；
  `negative_results_preserved` 保持 `true`。
- 新增兩個 VERIFIED block，數值同樣只能由 `scripts/verify_readme.py --write` 產生：
  - `CLASSIFICATION_SEED_VARIANCE`：自動找出所有達到
    `MIN_REPLICATED_SEEDS`（3）的 (物件, 組別)，輸出 Macro-F1 與 AUROC 的
    mean ± std。未達 3 seed 的組別不得出現在表內。
  - `SEGMENTATION_THRESHOLD`：五個主組的 Dice（threshold 0.5）與 AUPRO
    （threshold-free）及各自相對 Real-only 的差值。
- `RESULT_OUTCOME` 追加三行：AUPRO 差值與方向是否一致、零 Dice run 的數量與其中
  pixel AUROC ≥ `INFORMATIVE_PIXEL_AUROC`（0.80）的數量與最大值，以及一句明確聲明
  「AUPRO 與 threshold 敏感度是**併列揭露**，不是事後換指標」。
- 計數一律以 **physical run** 為準：`all_mixed` 是 `filtered_syn` 的 alias
  （ADR-009／ADR-021），不得被算成第二個 run。
- README 說明文字**不得複述**任何由 block 產生的數值，避免 CSV 更新後 prose 走鐘。

### 後果
- README 同時揭露「單 seed 不可靠」與「兩個分割指標互相矛盾」，讀者可自行判斷；
  這正好呼應課程 slide 8 自陳的 cons：影像／閾值指標未必反映真實下游效果。
- 這**不是**新的實驗結果，只是把既有 CSV 中已存在、先前未呈現的欄位攤開；
  沒有重跑任何訓練，`results/*.csv` 位元不變。
- 遺留缺口照實列入 README 限制：Segmentation 目前完全沒有 seed 複跑，
  因此 Dice／AUPRO 的矛盾**無法**用單 seed 判定孰是孰非；要解決必須補跑
  Segmentation 的 3-seed 複跑。
- `verify_readme.py` 的 block 數由 3 增為 5，`reports/readme_validation.json`
  的 `block_sha256` 隨之擴充；`verify_publish.py` 的既有綁定不需改動。

---

<a id="adr-028"></a>
## ADR-028 — 公開 orchestrator 與 guard 兩個 agent skill，其餘 11 個維持 owner-local

**狀態：Accepted**（2026-07-29）｜**部分取代**：M38 對 `.claude/` 的一律不追蹤

### 背景
本專案的設計目標之一是復刻課程 Chapter 5 的 agentic flow：把 SDG 管線拆成
13 個可被自然語言驅動、各自帶驗證與護欄的 skill。M38 為了精簡公開版面，
把整個 `.claude/` 停止追蹤，這一層在公開 repository 上因此完全消失，
README 也沒有任何說明。

但全部 13 個公開也不成立。實際稽核發現：

- `defectforge/SKILL.md` 有多處過期內容（`M0–M15` 的勾選範圍、
  「尚未建立的 skill」、`df-release` 從未實作、`Test-Path D:\sdg-data\...` 硬編路徑），
  且連結指向多份 owner-local 文件，公開後會是 404
- `df-guard/SKILL.md` 寫著「Keep the repository remote-free until the user creates
  the GitHub repository」——repository 早已公開；另有一處提及 `school email`，
  是歷史洩漏事件的殘留指涉
- 11 個階段 skill 含大量 `D:\sdg-data\01-defectforge` 的本機指令，
  且會隨 CLI 演進持續過期

### 決策
**公開兩個、其餘維持 owner-local。**

| Skill | 公開 | 理由 |
|---|---|---|
| `defectforge` | ✅ | orchestrator，證明「脈絡恢復 → 階段路由 → 里程碑收尾」的結構真實存在 |
| `df-guard` | ✅ | 防洩漏護欄，是本專案可信度的執行層，內容為可驗證的具體斷言 |
| 其餘 11 個 `df-*` | ❌ | 含本機絕對路徑、與 CLI 高度耦合、維護成本高於展示價值 |

公開前必須完成的清理（本 ADR 已一併執行）：
1. 移除所有硬編絕對路徑，改由 `configs/paths.yaml` 解析（ADR-005）
2. 指向 owner-local 檔案的 markdown 連結改為程式碼樣式純文字，避免 GitHub 404
3. 修正過期敘述：里程碑範圍、`df-release` 未實作、`df-guard` 已建立、repository 已公開
4. 移除 `school email` 字眼，改為一般性的 personal email addresses

### 邊界如何強制
- `.gitignore`（**版控內**）以 `/.claude/*` 排除全部，再負向放行兩個 skill 目錄。
  原本這條規則放在 `.git/info/exclude`（本機專屬、不進版控），已移出並加註：
  **父目錄一旦被排除，`.gitignore` 的負向規則就救不回來**，因此不得再放回 blanket `/.claude/`
- `scripts/verify_publish.py` 新增 `PUBLIC_SKILL_PREFIXES`，把兩個 skill 從
  `OWNER_LOCAL_PATHS` 的 `.claude/` 前綴中豁免；同時把兩份 `SKILL.md` 加入
  `REQUIRED_PATHS`，避免日後又被悄悄移除
- 單元測試同時驗證「兩個公開 skill 不被判為 owner-local」與「其他 `df-*` 仍被判為 owner-local」

### 後果
- 公開 repository 首次呈現 agentic flow 這一層，README 新增對應段落
- 維護面積只有 2/13；其餘 skill 演進不影響公開版面
- 若日後要再公開更多 skill，必須先做同一套清理，並追加 ADR，不得直接放行

### 未採用的選項
- **全部 13 個公開**：清理與長期維護成本高，且過期內容比不公開更扣分
- **只在文件描述、不放檔案**：任何人都能宣稱自己有 13 個 skill，缺少可驗證的實體

---

<a id="adr-029"></a>
## ADR-029 — 恢復 CI workflow，並把 24 小時授權時效改為僅發佈時強制

**狀態：Accepted**（2026-07-29）｜**部分取代**：M38 對 `.github/` 的一律不追蹤

### 背景
M35 建立 GitHub Actions（Windows runner、完整 Git 歷史、唯讀 token、鎖定 action SHA、
frozen lock），在每次 `main` push／PR 重跑 Ruff、pytest 與 publication audit。
M38 為精簡公開版面把整個 `.github/` 停止追蹤，**workflow 因此從遠端消失**。

2026-07-29 實查遠端 `actions/workflows`：`total_count = 1`，且那唯一一個是 GitHub
自動的 `Dependency Graph`。也就是說 M35 之後的所有 push 都沒有跑過任何驗證，
而 PLAN M35 仍寫著「每次 main push／PR 自動重跑相同 gate」。

直接把 workflow 加回來還有第二個問題：`verify_publish.py` 的
`model_license_verification_fresh` 要求 `reports/model_license_verification.json`
在 **24 小時內**（ADR-024）。這個規則的目的是「**發佈當下**證明上游授權剛查過」，
但套用到 CI 會讓每次 push 都因**報告年齡**（與被測 commit 無關）而變紅。

### 決策

**1. 只公開 workflow，不公開 template。**
`.github/workflows/verify.yml` 重新追蹤；`PULL_REQUEST_TEMPLATE.md` 與
`ISSUE_TEMPLATE/` 維持 owner-local（它們是個人協作習慣，公開沒有意義）。
`.gitignore` 沿用 ADR-028 的模式：`/.github/*` 排除，再負向放行
`!/.github/workflows/verify.yml`。

**2. 把「發佈閘門」與「持續驗證」分開，而不是放寬閘門。**

| | 發佈 | CI |
|---|---|---|
| 指令 | `verify_publish.py` | `verify_publish.py --allow-stale-license-check` |
| `model_license_verification_fresh` | **強制** | 回報但不 gating |
| 上游授權是否真的還有效 | 由 24 小時內的報告證明 | **CI 每次重新連 Hub 實查**（獨立步驟） |

CI 的實查步驟用 `--output` 寫到 `runner.temp`，**不覆寫已提交的報告**，
因此 `license_chain → model_license_verification` 的 hash 綁定不受影響。
這比原本更強：原本 CI 只驗報告年齡，現在是真的重新查一次上游。

**3. 豁免必須誠實。**
`build_audit(waived_checks=...)` 不竄改 `checks` 的值——
`model_license_verification_fresh` 過期時仍如實回報 `false`，只是不列入 `failed_checks`。
輸出新增 `waived_checks` 與 `failed_checks` 兩個欄位，CI 模式會在 stdout 印出
`NOT a release gate: waived ...`。未知的豁免名稱直接 `raise`，避免打錯字變成靜默放行。

### 後果
- 持續驗證恢復；`.github/workflows/verify.yml` 進入 `REQUIRED_PATHS`，
  日後被移除會讓發佈稽核 fail closed
- CI 不再因報告年齡變紅，但仍會因上游授權真的變更／變 gated 而變紅
- 發佈流程不變：release 前一定要跑無 flag 的完整閘門
- 單元測試涵蓋「豁免不竄改回報值」與「未知豁免名稱被拒絕」

### 待觀察
`verify.yml` 從 M35 建立後就沒有實際在遠端執行過。**本次 push 後必須確認 CI 真的跑起來
且為綠**；若 runner 上有本機沒有的失敗（例如 CRLF、路徑大小寫、缺 owner-local 測試檔），
要在下一個 commit 修掉而不是關掉 CI。
→ 已確認：commit `5d1027f` 的 Lint／Test／Re-verify upstream model licences／Audit
四個步驟全部 `success`。

---

<a id="adr-030"></a>
## ADR-030 — 零 Dice 完全由機率天花板造成；主指標不因此更換

**狀態：Accepted**（2026-07-29）

### 背景
16 個 physical 分割 run 中有 6 個在預註冊 threshold 0.5 下 Dice 恰為 0。
[ADR-027](#adr-027) 當時只能從 `pixel_auroc` 間接推論：3 個「機率圖有訊號、
被 threshold 切成全黑」，另 3 個「接近隨機」。那是**推論**，不是量測。

`scripts/diagnose_zero_dice_segmentation.py` 重跑全部 16 個 run 的推論，
先重算已發佈指標並要求相符（最大差異 `1.58e-03`，跨 Colab L4 與本機 4090 的浮點差），
再回報預測機率分布。

### 量測結果

**天花板完全決定 Dice 是否退化：**

| | 最高預測機率 |
|---|---|
| 6 個零 Dice run | 全部 **< 0.5**（最大 `0.4530`） |
| 10 個非零 Dice run | 全部 **> 0.5**（最小 `0.7360`） |

沒有任何一個 run 落在中間。也就是說在這批結果上，Dice 是否為 0 **與模型排序能力無關**
——零 Dice 那 6 個的 `pixel_auroc` 從 `0.4919` 一路分布到 `0.9015`。
空 Mask 是**算術上必然**，不是分割失敗。

**峰值位置才是有意義的第二層判準：**

| 峰值落在 | run | 意義 |
|---|---|---|
| 真實瑕疵**內** | pcb1 `copypaste_only`（`0.4530` vs 區域外 `0.2950`）、pcb1 `diffusion_only` | 排序正確但信心不足 |
| 真實瑕疵**外** | capsules 的 `std_aug`／`procedural_only`／`diffusion_only`／`unfiltered_syn` | 最有信心的像素是偽陽性 |

**最乾淨的對照是 capsules `std_aug`**：它的真實瑕疵曝光是 **100%**（完全沒有合成資料），
與 `real_only` 唯一差別是開了標準增強。機率天花板從 `0.9920` 掉到 `0.3157`，
`pixel_auroc` 只從 `0.9858` 降到 `0.8661`。**至少一個零 Dice 案例與合成資料完全無關**，
是增強造成的信心校準崩潰。

### 決策
1. **主指標不變。** 預註冊的 Dice 與 Macro-F1 結論、[ADR-027](#adr-027) 的措辭與
   `negative_results_preserved` 全部維持。
2. **不做 threshold 調校，也不回報調校後的 Dice。** 在 test 上挑一個讓 Dice 變好看的
   threshold，就是用 test 做模型選擇，違反 [autonomy_policy 的誠實性紅線](autonomy_policy.md)。
   本 ADR 只新增**量測**，不新增結果。
3. **修正 ADR-027 的敘述強度**：正確說法不是「3 個仍有訊號」，而是
   「**6 個全部**的機率天花板低於 threshold」。這個說法更強、更簡單，也更容易驗證。
4. 診斷腳本、`reports/zero_dice_diagnosis.{json,md}` 與其單元測試納入公開版面，
   README 的 threshold 段落連結該報告但**不複述任何數字**。

### 後果
- 「固定 threshold 的 Dice 在本設定下不是穩健主指標」從觀點變成有量測支撐的結論
- 未來若要讓 Dice 有意義，正確做法是在**不看 test** 的前提下處理校準
  （例如以 validation 訂 threshold、調整 loss 的正負權重），而不是事後挑 threshold
- `capsules/std_aug` 的崩潰顯示標準增強在此資料規模下可能有害；這是獨立於合成資料的
  新問題，尚未診斷

### 未做
未重跑訓練、未改動 `results/*.csv`、未改動任何已發佈結論。

---

<a id="adr-031"></a>
## ADR-031 — `capsules/std_aug` 是訓練期 Dice 項從未啟動；「裁切掉瑕疵」假設已被量測推翻

**狀態：Accepted**（2026-07-29）

### 背景
[ADR-030](#adr-030) 發現 `capsules/std_aug` 是最乾淨也最違反直覺的零 Dice 案例：
真實瑕疵曝光 **100%**（零合成資料），與 `real_only` 唯一差別是開了標準增強，
機率天花板卻從 `0.9920` 掉到 `0.3157`。`pcb1/std_aug` 則完全不受影響
（Dice `0.3836` vs `real_only` 的 `0.3762`）。

最直覺的解釋是：增強管線先做 random resized crop 再做仿射，**小瑕疵可能被推出畫面**，
留下一張標為「瑕疵」但 mask 已空的訓練對。

### 量測一：這個假設是錯的
`scripts/diagnose_augmentation_mask_loss.py` 重放 trainer 實際會抽到的同一組 draw
（同 transform、同 `_stable_seed(seed, sample_id, draw_index)` 序列），每張瑕疵圖 500 次：

| 物件 | 空 mask 比率 | 曾整個被切掉的圖 |
|---|---|---|
| pcb1 | **0.00%** | 0 / 10 |
| capsules | **1.44%** | **1 / 10** |

1.44% 遠不足以解釋完全崩潰。而且唯一會被切掉的 `capsules/…/084.JPG` 是**瑕疵最大**的
一張（佔全圖 `1.6081%`），與「小瑕疵被切掉」的方向相反——它的保留面積比只有 `0.105`，
比較像瑕疵沿邊界延伸而被裁掉大半。**假設推翻，不硬拗。**

### 量測二：崩潰發生在訓練期，不是測試期
比較訓練 loss 的窗平均（每 50 步）：

| 物件 | 組別 | 首窗 dice_loss | 末窗 dice_loss | Dice 項啟動 | 最終 Dice |
|---|---|---|---|---|---|
| pcb1 | `real_only` | `0.9925` | `0.9650` | 是 | `0.3762` |
| pcb1 | `std_aug` | `0.9909` | `0.9652` | 是 | `0.3836` |
| capsules | `real_only` | `0.9956` | `0.9696` | 是 | `0.5958` |
| capsules | `std_aug` | `0.9961` | `0.9944` | **否** | `0.0000` |

**BCE 在四個 run 都正常下降**（把整張預測成背景就能拿到低 BCE），
只有 `capsules/std_aug` 的 **Dice 項從頭到尾平的**。
pcb1 的兩條曲線幾乎逐窗重疊——增強在該物件近乎中性。

所以這**不是**測試期的校準假象：模型在訓練集上就沒學會與瑕疵重疊。
`capsules` 的瑕疵本來就比較小（原圖面積中位數 `0.1630%` vs pcb1 `0.2622%`），
`real_only` 的 dice_loss 也降得比 pcb1 慢——它本來就更靠近邊緣，
增強再吃掉一些樣本效率，在**凍結的 500 步預算**內就再也啟動不了。

### 決策
1. **如實記錄「假設被推翻」**，不改寫成事後看起來正確的敘事。
2. **不修增強設定、不加步數、不重跑**。那會動到 [ADR-021](#adr-021) 凍結的訓練預算，
   而且是在看過 test 之後才調——等同於用 test 做選擇。
3. **明確標示無法判定的部分**：單一 seed、每物件 10 張瑕疵圖，
   **無法區分「系統性」與「seed 雜訊」**。要判定必須先補分割的 3-seed 複跑
   （PLAN 已知未完成項目第 2 項）。此診斷因此**卡在**該項上。
4. 診斷腳本、`reports/augmentation_mask_loss.{json,md}` 與單元測試納入公開版面。

### 後果
- 「標準增強在此資料規模下可能有害」有了訓練期證據，但**尚未證明是系統性的**
- 未來若要處理，正確順序是：先補 seeds → 確認可重現 → 才在**不看 test** 的前提下
  調整增強強度或步數預算
- 這也說明第 2 組（`+ Standard Augmentation`）作為「排除一般增強就能達成」的對照組，
  在 `capsules` 上其實沒有發揮預期功能——它退化成了一個失敗的訓練，而非有效對照

### 未做
未重跑訓練、未改動 `results/*.csv`、未改動任何已發佈結論、未調整任何超參。

---

<a id="adr-032"></a>
## ADR-032 — 分割 3-seed 複跑的**預註冊**：全部 8 組，判定規則先寫死

**狀態：Pre-registered**（2026-07-29）｜**執行前必須先 commit 本 ADR**

### 背景
分割目前 18 列**全部是 seed 42**。這同時卡住兩個已量測、但無法判定的問題：

1. [ADR-027](#adr-027)：`filtered_syn` 相對 `real_only` 的 Dice 為 `-0.2264`、
   AUPRO 為 `+0.1046`，**方向相反**。單 seed 無法判斷哪個方向是真的。
2. [ADR-031](#adr-031)：`capsules/std_aug` 的 Dice 項在訓練期從未啟動。
   單 seed 無法區分系統性與雜訊。

原協定（[experiment_protocol §6](experiment_protocol.md)）只要求 `real_only` 與最佳
Filtered 組補到 3 seeds。那是為了省算力訂的；但實測 M19 單 run 僅約 5 分鐘
（pcb1 八組 2,509 秒、capsules 八組 2,321 秒），全部補完的成本並不高。

### 決策：**全部 8 個 formal group 都補**

| 項目 | 值 |
|---|---|
| 組別 | 8 個 formal group（`all_mixed` 是 alias，**不重跑**） |
| Seeds | `42`（已有）、`43`、`44` |
| 物件 | `pcb1`、`capsules` |
| **新增 run** | 8 × 2 × 2 = **32** |
| 估時 | 約 5 分鐘/run → 每物件約 80 分鐘 |

選「全部」而非只補兩組的理由：**不挑哪些組有誤差棒**。只補兩組會讓表格一半有
mean±std、一半沒有，而且會招致「看到結果才選複跑對象」的合理質疑。
全補是這三個選項中**最不具選擇性**的一個。

### 執行前就固定的判定規則（看到結果後不得更改）

**Dice／AUPRO 方向矛盾**
對每個物件，計算 `filtered_syn − real_only` 的**逐 seed** Dice 差與 AUPRO 差。
若在**至少一個物件**上，兩者符號相反的情況出現在 **3 個 seed 中的 ≥2 個**，
則判定「方向矛盾為真實現象」；否則判定為單 seed 假象。

**`capsules/std_aug` 崩潰**
若 `capsules/std_aug` 的 Dice 在 **3 個 seed 中的 ≥2 個**為 `0.0000`，
判定為**系統性**；若僅 1 個為 0，判定為 **seed 雜訊**，
並據此撤回 ADR-031 對「標準增強可能有害」的推測強度。

### 不得發生的事
- **不改任何超參、步數或增強設定**（[ADR-021](#adr-021) 的預算維持凍結）
- **不重新抽樣資料**：沿用已打包的 `m18_colab_selection.json` sample ID，
  seed 只影響訓練期隨機性（sampler draw、augmentation draw、head 初始化）
- **不丟棄任何 seed**，即使某個 seed 的結果不好看
- **不調 threshold**（[ADR-030](#adr-030) 已明令）
- seed 42 的既有數字與已發佈結論**在複跑完成前完全不動**

### 呈現方式
主表仍以預註冊協定為錨，另加 mean ± std——與 [ADR-027](#adr-027) 對分類的處理方式一致。
`results/segmentation.csv` 由 M20 聚合器從 raw report 重建，
`verify_readme.py` 的分割驗證需同步放寬到「每組 3 個 seed」。

### 後果（數字為實測，非估計）
- 需要使用者再上一次 Colab。只需重傳**兩個小檔**：
  - `defectforge_m18_source.zip`：**618,669 bytes（0.6 MB）**、107 檔，
    SHA256 `1a8c90e5a4cd2495e503642b96229369115a46ddc51bdeb830f89f48abab0473`
  - `notebooks/02_train_segformer.ipynb`
- **兩個 data bundle 完全不必重傳**：`m18_seg_pcb1.zip` **3.59 GB**、
  `m18_seg_capsules.zip` **3.77 GB**，sample ID 已凍在裡面，seed 不影響資料選擇
- 為此在 `package_m18_colab.py` 新增 `--source-only`：只重建 source archive，
  並沿用既有 manifest 的 data archive 事實，**不會改變兩個 data zip 的 SHA256**
- 舊的 100 MB source zip 是 [ADR-023](#adr-023) 收緊白名單**之前**的產物；
  重建後只剩 0.6 MB（不再夾帶 `reports/figures/`），舊檔保留為 `*.prev-2026-07-28.zip`
- 結果 ZIP 會從約 108 MB 增為約 325 MB／物件（24 個 run 的 checkpoint）
- `validate_segmenter_runs.py` 支援 seed 清單；預設維持 `[42]`，
  既有 seed-42 證據實測仍 `status=passed`

### 資料回來之後才做（不在本次範圍）
`aggregate_segmentation.py`、`verify_publish.py` 的
`physical_runs == 16` / `logical_rows == 18` 硬編值，以及 `verify_readme.py` 的
「18 列且 seed 必須是 42」驗證，**都必須同步放寬到 3 seeds**。
現在不改，因為改了會讓目前的證據鏈立刻失效。

---

## ADR-033 — 分割 3-seed 複跑的判定結果：兩條規則都執行，AUPRO 的提升未通過複跑

**狀態：Accepted**（2026-07-30）｜承接 [ADR-032](#adr-032) 的預註冊，**規則未作任何修改**

### 執行事實

| 項目 | 值 |
|---|---|
| 新增 run | 每物件 24 個（8 組 × 3 seeds），兩物件共 48 個實跑 |
| Runtime | Colab L4 22.0 GiB |
| 耗時 | pcb1 125.0 分、capsules 115.8 分；平均 312.6／289.4 秒每 run |
| Compute units | 執行前 363.70 → 執行後 356.94，**共 6.76 CU** |
| `results/segmentation.csv` | 由 M20 聚合器從 raw `training_report.json` 重建，**54 列**（48 實跑 ＋ 6 列逐 seed `all_mixed` alias） |

Drive 上沒有既有 `runs/` 可跳過，因此 seed 42 也被重跑，而非 ADR-032 預期的 16 個新 run。

### 判定結果（規則見 ADR-032，此處只記錄輸出）

**規則 1 — Dice／AUPRO 方向矛盾：判定為真實現象。**
`pcb1` 在 3 個 seed 中有 2 個（42、44）符號相反，達到預註冊門檻；`capsules` 只有 1 個。

**規則 2 — `capsules/std_aug` 崩潰：判定為系統性。**
Dice 在 seed 42 與 44 為 `0.0000`，達到 ≥2 個的門檻。因此 [ADR-031](#adr-031) 的主張
**維持不變，不予撤回**。

判定由 `scripts/decide_segmentation_replication.py` 產生，過程見
`reports/segmentation_replication.md`。

### 複跑推翻的一件事：seed 42 的 AUPRO 提升

[ADR-027](#adr-027) 依 seed 42 記載「`filtered_syn` 相對 `real_only` 的兩物件平均 AUPRO
為正、與 Dice 方向相反」。補上 seed 43、44 之後，**這個正向差異沒有重現**：

| 兩物件 macro Δ（`filtered_syn − real_only`） | seed 42 | 3-seed mean ± std |
|---|---:|---:|
| Dice | `-0.2264` | `-0.3296 ± 0.0903` |
| AUPRO | `+0.1046` | `-0.1224 ± 0.1976` |

`capsules` 在 seed 43、44 上 Dice 與 AUPRO **同時**大幅退步。因此在 macro 層級，兩個指標
的 3-seed 平均**方向一致、都是負的**；預註冊規則判定為真實的方向矛盾只存在於 `pcb1`。

**這使負面結論變強而非變弱。** 本 ADR 明確記載這一點，以免日後只引用 ADR-027 的
seed-42 敘述而得到過度樂觀的印象。ADR-027 依「只追加不改寫」的規則保留原文。

### 意外取得的重現性證據

seed 42 的 16 個實跑在不同機器、不同時間重跑後，**每個 `model.safetensors` 的 SHA256
與已發佈值逐一相同**，四項指標最大絕對差皆為 `0.00000000`。比對基準
`reports/segmentation_seed42_baseline.csv` 是複跑**之前**就已 commit 的表格，
由 `scripts/verify_seed42_reproduction.py` 驗證，不是事後挑選的數字。

### 隨判定一起放寬的硬編值（ADR-032「資料回來之後才做」的項目）

| 位置 | 由 | 改為 |
|---|---|---|
| `aggregate_segmentation.py` | 18 列／16 實跑、seed 42 硬編 | `LOGICAL_GROUPS × OBJECTS × SEEDS`，新增 `--seed` |
| `verify_publish.py` | `physical_runs == 16`、`logical_rows == 18` | `48` / `54`，並要求 `seeds == [42, 43, 44]` |
| `verify_readme.py` | 「18 列且 seed 必須是 42」 | 每組每物件必須恰有 3 個 seed；主表改以錨點取列 |
| `build_phase2_figures.py` | 隱含單列 | 明確取 seed 42，圖標題加註錨點 |
| `demo_gradio.py` | 要求全表 seed 42 | 改為在 54 列中取 seed 42 子集；出貨 checkpoint 不變 |

### 仍然沒做的事

- seed 43／44 新增的零 Dice run **沒有**重跑推論，因此
  [ADR-030](#adr-030) 的機率天花板機制在那些 run 上是尚未驗證的推論。
  README 的限制段落已明載
- Classification 仍只有兩組達到 3 seed，不在本次範圍

---

## ADR-034 — 零 Dice 有兩種成因；ADR-030 的「完全由機率天花板造成」在 48 個 run 上不成立

**狀態：Accepted**（2026-07-30）｜修正 [ADR-030](#adr-030) 的適用範圍，ADR-030 原文保留

### 觸發

[ADR-033](#adr-033) 的 3-seed 複跑把分割從 16 個實跑擴大到 48 個。零 Dice 的 run
從 6 個增為 23 個，因此把 `diagnose_zero_dice_segmentation.py` 重跑在全部 48 個 run 上。

### 量測結果

重算指標與已發佈值全數相符（最大差異 `4.33e-03`，容差 `5e-3`）後：

| 成因 | run 數 |
|---|---:|
| 最高預測機率低於 threshold，**連一個正像素都沒有** | 22 |
| **有**正像素，但完全沒有落在真實瑕疵上 | 1 |

例外是 `pcb1 / filtered_syn / seed 43`：最高機率 `0.8187`、8,660 個正像素、
pixel AUROC `0.8959`，判定 `positive_pixels_never_overlap_ground_truth`。

### 決策

1. **ADR-030 的核心機制維持**：機率天花板仍是零 Dice 的主要成因（22/23）
2. **但撤回其較強的推論**：「Dice 是否為 0 完全由機率天花板決定、與模型排序能力無關」
   在 48 個 run 上**不成立**。零 Dice 可以來自空間定位失敗，而不只是信心不足
3. ADR-030 依「只追加不改寫」保留原文，由本 ADR 界定其適用範圍

### 同時修正的工具缺陷（這才是重點）

原腳本把結論寫成**固定字串**，只把 `max()` 與 `min()` 的數值填進去，
**從未驗證該主張是否成立**。在 seed 42 的 16 個 run 上它碰巧全部正確，因此無人察覺；
擴大樣本後它輸出了「全部零 Dice run 的最高機率都低於 threshold `0.5`（最大 `0.8187`）」
這種自相矛盾的句子。

修正：

- 摘要改由資料推導——先分類，再依分類選標題與結論句；
  只有在完全沒有例外時才輸出「完全由機率天花板決定」
- 天花板上限只在**真正受天花板限制**的 run 上取 max（因此從 `0.8187` 更正為 `0.4530`）
- 表格與細節標題補上 seed 欄（原本三列同名 run 無法分辨）
- 新增 3 項回歸測試，其中一項直接以本次的反例為輸入

**教訓**：任何「全部 X 都滿足 Y」的敘述都必須由程式檢查後產生，不得預先寫成字串。
這與 [CLAUDE.md](../CLAUDE.md) 既有的「數字只能由腳本產生」是同一條原則的延伸——
**結論本身也算數字**。

---

## ADR-035 — v3 歸因 pilot 的**預註冊**：合成的殘餘落差來自放置還是外觀？

**狀態：Pre-registered**（2026-07-30）｜**執行前必須先 commit 本 ADR**

### 為什麼不是「繼續 M27」

[ADR-026](#adr-026) 的 gate 在執行前寫死，M26 pilot **三條全部未過**
（pcb1 AUROC `-0.0361`、capsules Macro-F1 `-0.1562`、AUROC `-0.0509`、
兩物件 mean Macro-F1 `-0.0781`），`confirmatory_run_authorized_by_gate=false`。
ADR-026 明文：**不得用 test 或追加 candidate 救結果；若要新假說，必須另開新的、
先提交的實驗版本**。因此 M27 維持 `stopped`，本 ADR 是那個「新的實驗版本」。

### 本次要問的問題

v2 證明曝光假說**部分正確**：`domain_balanced` 讓 pcb1 完全恢復，capsules 也從 v1 的
崩潰救回大半，但仍明顯低於 real-only。v2 沒有被設計來分辨殘餘落差的機制。

**同時發現一個汙染**：現有的三個來源消融 `src_copypaste` / `src_procedural` /
`src_diffusion` 全部是用 v1 壞掉的 sampler 跑的——`sampled_real_bad=14`、
`sampled_synthetic_bad=769`，與 `filtered_syn` 完全相同。因此「哪個合成來源比較好」
**從未在修正後的取樣下被問過**，現有排序不可採信。

### 設計：用既有資料做二因子分解

| Candidate | 瑕疵像素來源 | 放置與縫合 |
|---|---|---|
| `real_only` | 真實 | 真實 |
| `db_copypaste` | **真實**（貼上真實瑕疵 crop） | 合成 |
| `db_diffusion` | 生成 | 合成 |
| `db_procedural` | 生成（零真實瑕疵像素） | 合成 |
| `db_filtered` | 混合 | 合成 |

除 `real_only` 外全部使用 `domain_balanced`、`real_bad_share = 0.75`。
`0.75` **不是重新調參**：它是 v2 自己的預註冊排序選出的值，在 v3 以凍結常數進入。
其餘（ConvNeXt-Tiny、seed 42、100 steps、optimizer、transform、frozen real-only
validation）全部沿用，5 candidates × 2 objects = **10 個 development run**。

### 執行前就固定的判定規則（看到結果後不得更改）

以**每物件的 validation Macro-F1** 計算兩個懲罰量：

- **放置懲罰** `P = real_only − db_copypaste`
- **外觀懲罰** `A = db_copypaste − db_diffusion`

判定：

- 若 **兩個物件上都** `P > A` → 判定**放置／縫合為主因**
- 若 **兩個物件上都** `A > P` → 判定**外觀為主因**
- 其餘情況（含任一物件相等）→ 判定**依物件而異，無單一主因**，兩個數值都照實報告

「相等」以 `math.isclose(rel_tol=1e-9, abs_tol=1e-12)` 認定。這是**浮點表示的護欄，
不是科學門檻**：`0.8−0.6` 與 `0.6−0.4` 是同一個數卻不會 bit-equal，若用嚴格 `==`
比較，平手分支永遠走不到，等於讓 1e-17 的表示誤差決定判定方向。

**必須寫進報告的界限**：copy-paste 的瑕疵像素是真實的，但它的羽化／Poisson 縫合是合成的，
因此 `P` 實際上綁著「放置 + 縫合」兩件事，不是純粹的放置效應。這是**兩個 bundle 的歸因**，
不是三個乾淨因子的分解，不得在文件中被寫成後者。

### Confirmatory gate（與 ADR-026 逐字相同，不因 v2 失敗而放寬）

- 每物件 Macro-F1 不得比 real-only 低超過 `0.01`
- 每物件 AUROC 不得比 real-only 低超過 `0.02`
- 兩物件 mean Macro-F1 至少提升 `0.01`

任一條失敗即記錄 `stopped`。**歸因結論成立不等於 gate 通過**——即使我們查出主因，
只要 gate 未過就一樣不得讀 test、不得跑 3 seeds。

### 內建的有效性檢查

`real_only` 的設定與 v2 完全相同（`class_balanced`、`real_bad_share=0.50`、seed 42、
100 steps）。因此它的 validation 指標**必須重現 v2 pilot 的數值**。若不相符，代表環境
或資料發生漂移，**立即停止並回報**，不得繼續解讀其餘 candidate。

### 不得發生的事

- **不讀 frozen test**。development integrity guard 仍要求 test list 為空、
  validation 全為真實 train-side holdout
- 不重新生成任何合成資料、不改超參、不改 100-step 預算、不改 filter 門檻
- 不因結果不好而追加 candidate 或改 `real_bad_share`
- 不修改 [ADR-026](#adr-026) 的 gate 數值
- v1 的 `class_balanced` 位元行為維持不變

### 成本

10 個 development run，各約 30–45 秒、峰值約 3.2 GiB，本機 RTX 4090 即可，
**不需 Colab、不花錢**。

### 實作註記

- **不需要修改任何程式碼**：`run_v2_classifier_pilot.py` 的 candidate／gate 邏輯本來就
  泛用，v3 只是換一份 config。因此 [interfaces.md](interfaces.md) 的 CLI 契約不變
- runner 產生的 run name 仍帶 `m26_` 前綴（寫死在共用的 `_run_name`）。
  **刻意不改**：改它會動到 v2 的 resume 比對邏輯，破壞既有證據的可重現性。
  v3 的 run 以輸出目錄 `cls_v3_pilot` 區隔

---

## ADR-036 — v3 歸因結果：gate 再次未過；唯一有鑑別力的物件指向「放置／縫合」

**狀態：Accepted**（2026-07-30）｜承接 [ADR-035](#adr-035) 的預註冊，**規則未作任何修改**

### 有效性檢查（先於一切解讀）

ADR-035 要求 `real_only` 必須重現 v2 的數值，否則立即停止。實測**四個指標全部完全相同**：

| 物件 | Macro-F1 | Δ vs v2 | AUROC | Δ vs v2 |
|---|---:|---:|---:|---:|
| pcb1 | `0.6944` | `+0.0000` | `0.9167` | `-0.0000` |
| capsules | `0.8133` | `+0.0000` | `0.9120` | `+0.0000` |

無環境漂移，可以解讀其餘 candidate。

### Gate：再次 `stopped`

| 檢查 | 門檻 | 實測 | |
|---|---|---:|:---:|
| pcb1 Macro-F1 | ≥ −0.01 | `+0.0000` | ✅ |
| pcb1 AUROC | ≥ −0.02 | `-0.0194` | ✅ |
| capsules Macro-F1 | ≥ −0.01 | `-0.1244` | ❌ |
| capsules AUROC | ≥ −0.02 | `-0.1343` | ❌ |
| 兩物件 mean Macro-F1 gain | ≥ +0.01 | `-0.0622` | ❌ |

`confirmatory_run_authorized_by_gate = false`。**沒有讀取 frozen test**
（`test_data_loaded = false`），沒有跑 3 seeds，沒有花錢。M27 維持 `stopped`。

### 歸因判定：依物件而異（但只有一個物件有鑑別力）

| 物件 | P（放置＋縫合） | A（外觀） | 較大者 | 指標可分辨 |
|---|---:|---:|---|:---:|
| capsules | `+0.1319` | `-0.0074` | placement | 是 |
| pcb1 | `+0.0000` | `+0.0000` | tie | **否** |

依預註冊規則，pcb1 平手 → 整體判定 **`object_dependent`**，**不得**宣稱放置為主因。

但 pcb1 的平手不是「兩種成因勢均力敵」：該物件上 `real_only`、`db_copypaste`、
`db_diffusion` 的 Macro-F1 **完全相同（`0.6944`）**，指標在此物件沒有鑑別力。
判定腳本會偵測並在報告中標記這一點，不讓它被讀成證據。

因此**唯一有鑑別力的物件（capsules）給出強烈訊號**：把真實瑕疵貼到合成位置要付
`+0.1319` 的代價，而把瑕疵材質從真實換成生成只要 `-0.0074`——**是負的**，
生成外觀略優於真實外觀。這是**假說生成級**的證據，不是已確認的歸因。

### 順帶推翻的一個既有排序

現有的來源消融（`src_copypaste` / `src_procedural` / `src_diffusion`）全部是用 v1 壞掉的
sampler 跑的（`real_bad=14`、`synthetic_bad=769`），排序為
**copypaste > procedural > diffusion**，看起來像是「真實外觀比較好」。

在修正後的取樣（`real_bad=613`、`synthetic_bad=215`）下，排序**反轉**為
**diffusion > copypaste > filtered > procedural**。原排序是曝光崩潰的產物，不可採信；
本 ADR 明確作廢它。

### 修正的實作缺陷

1. `decide_v3_source_attribution.py` 初版假設的結果 schema（`candidates` 巢狀 dict）
   與 runner 實際輸出（`runs` 扁平 list）不符。**只改讀取層，判定規則一字未動。**
2. 測試先抓到嚴格 `==` 讓「平手」分支在浮點下永遠走不到（`0.8−0.6` ≠ `0.6−0.4`），
   已於 ADR-035 commit 前改用 `math.isclose` 並註明那是浮點護欄。
3. 新增 `metric_discriminates` 旗標與兩項測試，讓「指標無鑑別力」不會被誤讀成平手。

### 執行事實

10 個 development run，本機 RTX 4090。實測約 **75 秒／run**，比 ADR-035 估的 30–45 秒慢，
因為 `src_*` 群組要額外雜湊驗證合成檔案。中途一次 process 中斷留下 1 個半成品目錄，
runner 的 fail-closed 檢查正確擋下並要求人工檢視；該目錄無指標、模型也無綁定報告，
刪除後由 seed 42 決定性重跑，其餘 5 個完整 run 原封沿用。

### 下一步的候選（皆尚未預註冊）

若要繼續，必須再開一個先提交的版本。目前看來最有價值的方向是**檢驗放置管線**
（M9 的 mask placement 與 M12 的縫合），而不是再換生成模型——但這只是傾向，
不是本 ADR 的結論。
