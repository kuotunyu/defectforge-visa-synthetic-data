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
[CLAUDE.md](../CLAUDE.md) 的分工原則是「>30 分鐘的 GPU 訓練一律 Colab」。
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
- 第 9 組直接引用第 4 組，`instructions_for_me.md` 要寫清楚只需跑 8 次

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
採**跑到底模式**：連續執行 M1 → M14，每個里程碑跑完 [PLAN.md](../PLAN.md) 的驗證欄，
**全綠才前進**。任何一項不過就**立刻停止**，寫一份 handoff 報告，不得自行降低標準、
不得跳過、不得「先繼續之後再回來修」。

完整規則書寫在 [autonomy_policy.md](autonomy_policy.md)，重點：
- **可自己決定**：門檻校準、隨機種子以外的實作細節、重試暫時性錯誤、瑕疵型別的暫用命名
- **必須停**：任何驗證失敗、需要花錢、>2GB 下載、任何 push/發佈、Colab 訓練、
  磁碟不足、生成結果目視明顯異常
- **瑕疵型別命名不阻塞**：自動產生 `<pcb1-type0>` 之類的暫用 token 繼續跑，
  使用者醒來只改**顯示名稱**（不動 token 字串，因此不需要重訓）

### 後果
- [PLAN.md](../PLAN.md) 每個里程碑要標註「無人值守可跑 / 需要你在場」
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
