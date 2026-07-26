# PLAN.md — Phase 1 里程碑

> 規則見 [CLAUDE.md](CLAUDE.md)｜決策見 [docs/decisions.md](docs/decisions.md)｜每次工作紀錄見 [docs/worklog.md](docs/worklog.md)
> **Phase 1 目標**：資料與 split 凍結 → Stage A 合成 → Stage B 合成（課程復刻主秀）→ 品質過濾 → smoke test 與交接文件。
> **Phase 1 明確不做**：分類器訓練、分割訓練、Gradio demo、發佈（全部是 Phase 2）。

狀態圖例：`[ ]` 未開始｜`[~]` 進行中｜`[x]` 完成

---

## M0 — 文件與規則凍結

- [x] **M0** 建立 `CLAUDE.md`、`PLAN.md`、`docs/` 九份規格、ADR-001~006、`defectforge` skill、目錄骨架
  - **驗證**：`Get-ChildItem -Recurse` 對照 `docs/skills_roadmap.md` 與本檔清單；grep 全 md 無 `/mnt/c`、`~/sdg-portfolio`、無 `gho_`/`hf_`/`sk-`/`AIza` 開頭字串；`git log --stat` 有一筆 commit 且 `git status` 乾淨

---

## M1–M2 — 環境與資料落地

- [ ] **M1** 建立 uv 虛擬環境並鎖版（Python 3.12、torch cu128、diffusers/peft/transformers/opencv/imagehash/scikit-learn/clean-fid）
  - **驗證**：`uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` 印出 CUDA 可用且裝置是 RTX 4090；`uv.lock` 存在並進 git；`uv run python -c "import diffusers, peft, cv2, imagehash, sklearn, cleanfid"` 全部匯入成功

- [ ] **M2** 下載 VisA 並校驗（**下載前先跟使用者報備大小 1.80 GB**），解壓到 `D:\sdg-data\01-defectforge\raw\VisA`
  - **驗證**：`VisA_20220922.tar` 檔案大小 = `1929840640` bytes；記錄其 SHA256 到 `splits/source_checksums.json`；解壓後 `pcb1` 影像數 = 1004 normal + 100 anomaly、`capsules` = 602 normal + 100 anomaly（與 [docs/data_protocol.md](docs/data_protocol.md) 的事實表逐項比對，不符就停下來報告）

---

## M3–M6 — Split 與分型凍結（**這四項全綠之前，一張合成圖都不准生**）

- [ ] **M3** 用 spot-diff 官方 `prepare_data.py --split-type 2cls_fewshot` 與 `2cls_highshot` 產生兩套佈局
  - **驗證**：兩套 `VisA_pytorch/` 目錄結構符合 `train/{good,bad}`、`test/{good,bad}`、`ground_truth/{train,test}/bad`；每個 bad 影像都有對應 mask（數量相等、檔名可對上）；印出 train/test × good/bad 的四格計數表，與 20%/80%、60%/40% 協定相符

- [ ] **M4** pHash 近似分群 + 凍結 split manifest + 建立 test blocklist
  - **驗證**：`splits/split_manifest.json` 含每張圖的 `{object, set, label, path, sha256, phash, group_id}`；斷言「同 `group_id` 的影像 `set` 必定相同」通過（不通過就自動把整群搬到 test 側並重印）；`splits/test_blocklist.json` 的 SHA256 集合與 test 影像數相等；manifest 自身的 SHA256 寫進 `splits/MANIFEST.sha256`

- [ ] **M5** few-shot 抽樣（seed=42、k=10 瑕疵/物件）＋ 統計表 ＋ contact sheet
  - **驗證**：重跑兩次抽樣結果完全相同（雜湊比對）；`reports/fewshot_stats.md` 的樣本數表與 manifest 重新聚合的數字一致；`reports/figures/fewshot_contact_sheet_{pcb1,capsules}.png` 產出後**自己開啟檢視**，確認每格都疊了 mask 輪廓且瑕疵肉眼可辨 → 交給使用者過目

- [ ] **M6** 瑕疵分型（DINOv2 + 形態特徵分群 → contact sheet → **使用者確認命名** → 凍結）
  - **驗證**：`splits/defect_types.json` 含每個連通元件的 `{object, image, component_id, cluster_id, type_name, trigger_token}`；每群 ≥3 個元件（否則已依 [ADR-002](docs/decisions.md#adr-002) fallback 合併）；分群輸入路徑全部落在 few-shot seed 內（用 test blocklist 反查，命中即失敗）；每群 contact sheet 自己看過且使用者已確認名稱

---

## M7–M8 — Stage A 合成（本機，保底）

- [ ] **M7** `src/synthetic/copy_paste.py`：從 few-shot 真實 mask 擷取 defect patch，貼到同物件正常圖
  - **驗證**：產 200 張/物件到 `data_root/synthetic/stageA_copypaste/`，每張都有配對 mask 與 `metadata.jsonl` 一行；斷言 mask 與影像尺寸相同、mask 非全零、貼上位置在 ROI 內；隨機抽 24 張做 grid **自己檢視**，確認邊緣沒有明顯矩形接縫、瑕疵沒有貼在背景上

- [ ] **M8** `src/synthetic/procedural.py`：Perlin/裂痕/刮痕/斑點 mask ＋ 紋理填充
  - **驗證**：同 M7 的自動斷言；額外驗證產生的 mask 面積與長寬比分布落在 `reports/real_mask_stats.json`（由 few-shot 真實 mask 算出）的 5–95 百分位內，超出比例 <10%；grid 圖自己檢視

---

## M9–M12 — Stage B 合成（課程復刻主秀）

- [ ] **M9** `src/synthetic/mask_placement.py`：ROI 偵測（Otsu / DINOv2）→ 真實 mask 隨機仿射 → 放進 ROI
  - **驗證**：對每張正常圖輸出 `(good 圖, 放置後 mask)` 配對與**視覺化檢查圖**（ROI 框 + 放置 mask 疊圖）；斷言放置 mask 100% 落在 ROI 內、與其他已放置 mask 不重疊、面積在真實分布範圍內；抽 24 張視覺化圖自己檢視，ROI 明顯抓錯就換方法

- [ ] **M10** `notebooks/01_train_inpaint_lora_sd2.ipynb`（Colab）：diffusers + peft 微調 SD2-inpainting，每物件一個 LoRA、每瑕疵型一個 trigger token
  - **驗證**：本機以 `--max_train_steps 1 --smoke` 跑通並存出 LoRA 權重檔（檔案存在且能被 `PeftModel` 載回）；notebook 有斷點續跑分支（刪 checkpoint / 保留 checkpoint 各跑一次驗證行為）；輸出目錄符合 `runs/lora_sd2/<object>/seed_42/`；notebook 內**沒有任何明文 token**

- [ ] **M11** `notebooks/02_train_inpaint_lora_sdxl.ipynb`（Colab）：同上，改用 SDXL-inpainting-0.1（[ADR-001](docs/decisions.md#adr-001)：SD2 全流程綠燈後才動）
  - **驗證**：同 M10；額外記錄本機 smoke test 的峰值 VRAM，並在 `instructions_for_me.md` 標明需要 L4 而非 T4

- [ ] **M12** `src/synthetic/generate_diffusion.py`：批次生成（本機 4090 / Colab 通用）＋ `refine` 的 `(guidance_scale, crop_ratio)` 搜尋
  - **驗證**：每瑕疵型生成 200 張到 `data_root/synthetic/stageB_<model>/original/`，refine 版進 `.../searched/`；`metadata.jsonl` 每行含 [docs/synthesis_spec.md](docs/synthesis_spec.md) 規定的所有 provenance 欄位（用 schema 驗證腳本逐行檢查，缺欄即失敗）；同一 seed 重跑產出位元相同；grid 圖自己檢視，`original` 與 `searched` 並排比對確認 refine 真的有變好

---

## M13–M15 — 過濾、指標與交接

- [ ] **M13** `src/filtering/`：六道過濾 → 輸出 `filtered/` 與 `unfiltered/` 兩版
  - **驗證**：`reports/filter_report.md` 的漏斗表（生成數 → 每道拒絕數 → 最終數）能從 `metadata.jsonl` 重新聚合出完全相同的數字；每筆被拒樣本都有 `reject_reason` 且值在 enum 內；抽查 12 張被拒與 12 張通過的樣本做並排 grid **自己檢視**，若過濾把明顯好的樣本刷掉就調門檻並記錄

- [ ] **M14** `src/evaluation/`：`nn_score` / `mnn_score` / FID / KID（[ADR-006](docs/decisions.md#adr-006)）
  - **驗證**：`reports/generation_quality.md` 分瑕疵型列出四項指標；健全性檢查——把**真實瑕疵 crop 自己餵進去**，`nn_score` 應接近 1、KID 應接近 0（不成立表示實作有錯）；確認所有輸入 crop 的來源路徑都不在 test blocklist 內

- [ ] **M15** `instructions_for_me.md` 填滿 ＋ Phase 1 驗收
  - **驗證**：兩本 notebook 各自具備五節（上 Colab 方式／Runtime 選型／Colab Secrets 名稱／預估時數與 compute units／跑完下載哪些檔放回 `results/colab/` 的哪個路徑）；照著文件從零走一遍紙上流程沒有缺口；`PLAN.md` M1–M15 全勾；`git log` 每個里程碑至少一筆 commit

---

## 交接與追蹤

每完成一個里程碑：勾選本檔 → 追加 [docs/worklog.md](docs/worklog.md) 一筆 → `git commit` → 給使用者「換你做」清單。
