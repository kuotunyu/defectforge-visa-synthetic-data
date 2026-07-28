# Methodology — Cosmos AnomalyGen → open-source replication

> 本專案的靈魂文件。實作任何 Stage B 元件前先讀這份。
> 來源：NVIDIA GTC Taiwan 2026 課程《Few-shot Industrial Synthetic Data Gen with NVIDIA
> Defect Image Generation Agent》簡報（`gtc_tw_lab.pptx`，17 張），以及公開的
> [NVIDIA/skills · physical-ai-defect-image-generation](https://github.com/NVIDIA/skills/tree/main/skills/physical-ai-defect-image-generation)。
>
> **本專案與 NVIDIA 無關聯、未經其背書。** 我們復刻的是**方法論**，不是程式碼或權重。
> 表格中凡標記 🔶 的欄位是**我們的詮釋**——NVIDIA 未公開該部分的確切公式或實作。

---

## 1. 課程的完整管線（簡報 Step 1–3）

```
Step 1  Few-shot fine-tuning
        Training dataset (Real anomaly & mask) → Cosmos AnomalyGen post-train

Step 2  High quality Synthetic data generation
  2.1   User input data augmentation
        (Clean image, given mask) → Auto Mask placement → (Clean image, augmented mask)
          · AI / CV based ROI Generator
          · Placement (Legal / Illegal ROI)
          · Augmentation (Pos, size, angle...)
  2.2   Synthetic dataset generation (SDG)
        (Clean image, augmented mask) → Inference (post-trained AnomalyGen)
                                      → (Clean, mask, synthetic image)
  2.3   Pseudo labeling & dataset curation
        · Label curation      · KPI filtering
        · Mask refinement     · Pseudo labeling (BBOX, caption...)
        · Downstream format conversion
                                      → High-quality downstream dataset

Step 3  Downstream model training → Finetune → Deployment
```

課程的規模參考（簡報 slide 4 / 13）：訓練約 1 小時 @ 1×H100 跑 10k steps；推論約 5 秒/張
@ 1×H100（2B 模型）；硬體門檻最低 48GB VRAM（L40），建議 80GB（A100/H100）。
**我們用 24GB 的 4090 與 Colab T4/L4 做到同一套方法論**，這本身就是復刻的重點之一。

---

## 2. 元件對照表

### 2.1 生成模型骨幹（簡報 slide 5–6）

| 課程元件 | 課程做法 | 我們的開源替代 | 說明 |
|---|---|---|---|
| 擴散骨幹 | Cosmos Predict2 T2W（2B，video diffusion），輸出經 Cosmos Tokenizer Decoder | **SD2-inpainting UNet（865M）**，SDXL-inpainting-0.1（2.6B）作對照 | 兩者皆 CreativeML Open RAIL++-M，允許發佈輸出。理由見 [ADR-001](decisions.md#adr-001) |
| 文字編碼 | T5 Encoder → Text Embedding | SD 內建 CLIP text encoder | 我們不需要 T5 級的長文本條件；瑕疵描述極短 |
| Anomaly Embedding | 可學習的 **look-up table**，一個 anomaly type 一個 embedding（例：`"Bottle+broken_large"`），輸出 (B, 256, 1024) | **每個瑕疵型一個 trigger token**（textual-inversion 風格）＋ 每物件一個 LoRA | 功能等價：把「瑕疵外觀類型」壓成一個可學習的條件向量。VisA 沒有原生 type label，型別怎麼來見 [ADR-002](decisions.md#adr-002) |
| Mask Encoder | Preprocess → **預訓練 NVDINOV2** → Spatial Embedding (B, 25, 1024) | inpainting pipeline **原生的 mask 通道**（UNet 輸入 9 通道：4 latent + 1 mask + 4 masked-latent） | SD-inpainting 的架構本來就把 mask 當空間條件吃進去，不需要外掛 encoder。DINOv2 在我們這邊改用於 **ROI 偵測**與**過濾/評測** |
| Text Inversion | Caption → T5 → 視為 2×81 tokens | trigger token 直接嵌在 prompt 裡：`a photo of <pcb1-scratch> defect on a printed circuit board` | 同一個概念的輕量版 |
| Encode + Blend | Clean image → Adapter → Encode + Blend（for inpainting） | **crop-to-ROI → inpaint → 全解析度縫合回去**（Poisson / 羽化 alpha） | VisA 約 1500×1000 且瑕疵極小，直接縮圖會毀掉細節。見 [ADR-004](decisions.md#adr-004) |
| 條件融合 | Condition (B, 512, 1024) = spatial ⊕ anomaly ⊕ text | prompt（trigger token）＋ mask 通道 ＋ 原圖 latent | 三路條件的角色一一對應 |

課程明講「Idea adopted from AnomalyDiffusion」。我們同樣以
[AnomalyDiffusion (AAAI 2024)](https://arxiv.org/abs/2312.05767) 的 **spatial anomaly
embedding** 為理論基礎：把瑕疵資訊拆成「外觀（anomaly embedding）」與「位置形狀
（mask 條件）」兩路，兩路都可學。

### 2.2 Auto Mask Placement（簡報 slide 7、17）

課程的分支：

```
Has CAD ?
  ├── Yes → CAD to ROI Generator ──┐
  └── No  → Text2Box ROI Generator ─┴→ ROI BBox → Auto Mask Placer → placement result
```

| 課程元件 | 我們的實作（`src/synthetic/mask_placement.py`） |
|---|---|
| CAD to ROI Generator | **不做**。VisA 沒有 CAD 檔，這條分支在本專案無對應輸入 |
| Text2Box ROI Generator | 🔶 兩段式：(1) Otsu / 形態學前景分割取物件主體；(2) DINOv2 patch-token 的異質度圖抓「有結構的可放置區」。兩者取交集當合法 ROI |
| Legal / Illegal ROI | 合法 = 物件前景 ∩ 非影像邊界；非法 = 背景、影像外框、已放置 mask 的鄰域 |
| Auto Mask Placer | 真實 mask → 隨機仿射（平移／旋轉／縮放，可選鏡射）→ 排斥取樣直到完全落在合法 ROI 內；面積與長寬比須落在真實分布的 5–95 百分位 |
| Augmentation (Pos, size, angle) | 同上；參數與範圍寫在 `configs/` 並記進 `metadata.jsonl` |

課程的賣點是「**1 張正常圖 + 1 張初始 mask → 增強出很多 mask**」，讓 SDG 能規模化。
我們的驗收標準完全相同：每張正常圖都能產出多組合法 (clean, mask) 配對，且有視覺化檢查圖。

### 2.3 配額分配（`prep-testcase` skill）

課程的 `prep-testcase` 會「依訓練 mask 數量**按比例**分配 `num_SDG`」。
我們照做：每個瑕疵型要生幾張，正比於該型在 few-shot seed 中的連通元件數量，
使合成資料的型別分布不偏離真實分布。細節寫在 [synthesis_spec.md](synthesis_spec.md)。

### 2.4 Refine 搜尋（`sdg-refine` skill）

課程：對已存在的 `original/` 桶做 `num_search_run` 輪 per-sample 搜尋，搜尋維度是
**`(guidance, crop_ratio)`**，把每個樣本「歷來最佳的一次嘗試」組裝成 `searched/` 桶。

我們一比一照抄這個設計，包括**雙輸出桶**：
`stageB_<model>/original/` 與 `stageB_<model>/searched/`，兩桶都保留、都可進消融。
「最佳」的判準用 [ADR-006](decisions.md#adr-006) 的 `nn_score` 加上邊界融合分數。

### 2.5 Quality Evaluation（簡報 slide 8）

課程列了三種解法，我們三種都做：

| 課程解法 | Pros / Cons（課程原話重點） | 我們的實作 |
|---|---|---|
| 1. Visual inspection | 有 domain knowhow，但難量化、無法自動化（HIL） | 每個生成階段都產 contact sheet / grid，先完成內部目視，再交使用者確認 |
| 2. Image based quality | 可量化、可整合進 SDG 流程，但指標可能有偏差 | `nn_score`、`mnn_score`、FID、KID —— 定義見 [ADR-006](decisions.md#adr-006)。**在瑕疵 crop 上算**，不是整張圖 |
| 3. Downstream model improvement | 直接反映使用者目標，但每次評估等於一次下游微調，很耗時 | Phase 2 的五組對照實驗，含合成量 0.5×/1×/2× 掃描 |

課程展示的下游成效形狀（mIoU 相對無 SDG 基準）：
`+0.0519 @ +125 張`、`+0.0676 @ +250 張`、`+0.0851 @ +500 張`——
**遞增但邊際遞減**。我們 Phase 2 的合成量掃描就是為了畫出同一條曲線；
若我們的曲線是平的或往下，仍須如實報告並分析。

### 2.6 Agentic Flow（簡報 slide 10–11）

課程用 9 個 skill 把管線拆成可被 agent 呼叫的模組（`anomalygen`、`anomalygen-guard`、
`anomalygen-release`、`eval`、`finetune`、`prep-testcase`、`sdg-inference`、
`sdg-refine`、`setup`）。這是課程的 Chapter 5，也是我們最容易做出差異化的地方：
**我們用 Claude Code 的專案級 skills 復刻同一套 agentic flow**。
開發期間使用的 Project-level Skills 屬 Owner 本機工作流程，不列入公開 Repository。

---

## 3. 我們刻意**不**復刻的部分

誠實劃界，README 也會照抄這一節：

| 不做 | 原因 |
|---|---|
| Cosmos Predict2 / AnomalyGen 權重本身 | `nvidia/Cosmos-AnomalyGen-Metal-2B` 是 gated model 且需要 48GB+ VRAM；本專案的定位是**用開源工具達成同一目標** |
| CAD-to-ROI 分支 | VisA 沒有 CAD 輸入 |
| IsaacSim 結構性瑕疵（tombstone / shift / sideflip）與 USD 渲染 | 需要 3D 資產與模擬環境，超出 2D 資料集的範圍 |
| Blackbox VLM 品質評分（GPT / Gemini） | 要花錢且引入外部相依；我們用 DINOv2 指標取代。（Phase 2 若使用者同意再加） |
| OSMO / DIG 雲端工作流與 Docker release 流程 | 那是 NVIDIA 的內部基礎設施 |

---

## 4. 為什麼這樣復刻仍然成立

課程 slide 6 自陳的 main idea 是三句話：

1. 利用預訓練擴散模型的先驗知識
2. 以瑕疵為條件——訓練額外元件從額外輸入抽取資訊
3. 想法取自 AnomalyDiffusion

這三點在 SD-inpainting + LoRA + trigger token + mask 條件的組合下**全部成立**。
差異只在骨幹容量（2B video diffusion vs 865M/2.6B image diffusion）與硬體規模。
因此「底模容量對下游提升的影響」本身就值得做成一組消融——這正是我們同時做
SD2 與 SDXL 的理由（[ADR-001](decisions.md#adr-001)）。
