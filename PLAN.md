# PLAN.md — 里程碑

> 規則見 [CLAUDE.md](CLAUDE.md)｜決策見 [docs/decisions.md](docs/decisions.md)｜工作紀錄見 [docs/worklog.md](docs/worklog.md)
> 無人值守規則見 [docs/autonomy_policy.md](docs/autonomy_policy.md)｜腳本契約見 [docs/interfaces.md](docs/interfaces.md)

狀態：`[ ]` 未開始 ｜ `[~]` 進行中 ｜ `[x]` 完成

值守標記：
- **🤖** 無人值守可跑到底
- **👀** 會產生要看的圖，但**不阻塞**（agent 自己先看過即可繼續，使用者事後複核）
- **🙋** **必須停下來等人**（花錢／大下載／Colab／發佈／需要人眼決策）

---

# Phase 1 — 資料凍結與兩階段合成管線

## M0 — 文件與規則凍結 🤖

- [x] **M0** `CLAUDE.md`、`PLAN.md`、`docs/` 十二份規格、ADR-001~012、`defectforge` skill、目錄骨架
  - **驗證**：檔案清單對照本檔；全 md 的相對連結 0 broken；ADR anchor 全部對得上；
    grep 無 `/mnt/c`、`~/sdg-portfolio`、無金鑰前綴；`pyproject.toml`/`paths.yaml` 可被解析；
    `git log` 有 commit 且 `git status` 乾淨

---

## M1–M2 — 環境與資料落地

- [x] **M1** 🤖 建立 uv 虛擬環境並鎖版（Python 3.12、torch **2.13.0+cu130**）
  - **驗證**：`uv run python -c "import torch; ..."` 印出 `2.13.0+cu130`、CUDA 可用、裝置為 RTX 4090
    （**版本字串必須含 `+cu130` 後綴**——沒有就是裝到 PyPI 的 CPU-only 輪子，見 `docs/environment.md`）；
    `uv run python -c "import diffusers, peft, timm, cv2, imagehash, sklearn, cleanfid"` 全數匯入成功；
    **確認 `diffusers` 0.39 與 `transformers` v5 的相容性**（`uv lock` 的解析結果會直接顯現；
    若不相容，取捨要記成一則 ADR）；
    `uv.lock` 存在並進 git；開工前已重新查證各套件當時最新版並記進 `docs/environment.md`

- [x] **M2** 🙋 下載 VisA（**1.80 GB，下載前必須先問使用者**），解壓到 `${data_root}/raw/VisA`
  - **驗證**：tar 大小 == `1929840640` bytes；SHA256 寫進 `splits/source_checksums.json`；
    解壓後 pcb1 = 1004 normal + 100 anomaly、capsules = 602 normal + 100 anomaly
    （對照 [data_protocol.md §1](docs/data_protocol.md)，**不符即停，不要自行調整**）

---

## M3–M6 — Split 與分型凍結（**這四項全綠之前，一張合成圖都不准生**）

- [x] **M3** 🤖 跑 spot-diff `prepare_data.py` 產生 `2cls_highshot`（基底）與 `2cls_fewshot`（取 k=10 用）
  - **驗證**（[ADR-007](docs/decisions.md#adr-007)，任一不符即停）：
    ① 八格張數與 [data_protocol.md §3.1](docs/data_protocol.md) 逐格相符
    ② `highshot_train ∩ highshot_test == ∅`
    ③ `fewshot_train ⊆ highshot_train`
    ④ 每張 bad 影像都有對應 mask（數量相等、檔名可對上）

- [x] **M4** 🤖 pHash 近似分群 → 凍結 `splits/split_manifest.json` → 建 `splits/test_blocklist.json`
  - **驗證**：manifest 含每張圖的 `{object,set,label,path,sha256,phash,group_id,...}`；
    斷言「同 `group_id` 必定同 `set`」通過（不通過就整群移到 test 側並重印）；
    blocklist 涵蓋每張 test 影像與其 bad mask，且 unique SHA256 計數自洽；
    `splits/MANIFEST.sha256` 寫入；
    `reports/split_report.md` 記錄因 pHash 而移動的張數

- [x] **M5** 👀 few-shot 抽樣（seed=42、k=10／物件）＋ 統計表 ＋ contact sheet
  - **驗證**：重跑兩次抽樣的檔名清單雜湊完全相同；`reports/fewshot_stats.md` 的數字能從 manifest 重新聚合；
    `reports/figures/fewshot_contact_sheet_{pcb1,capsules}.png` **自己用 Read 打開檢視**，
    確認每格都疊了 mask 輪廓、瑕疵肉眼可辨；`reports/real_mask_stats.json` 產出

- [x] **M6** 👀 瑕疵分型（DINOv2 + 形態特徵分群）→ 暫用命名 → 凍結 `splits/defect_types.json`
  - **無人值守做法**（[ADR-012](docs/decisions.md#adr-012)）：自動產生暫用 token `<pcb1-type0>`…並繼續往下跑，
    **不要停在這裡等命名**。使用者事後只改**顯示名稱**（不動 token 字串，因此不需重訓）
  - **驗證**：每群 ≥3 個元件（否則已依 [ADR-002](docs/decisions.md#adr-002) fallback 合併）；
    分群輸入路徑全數不在 test blocklist 內（命中即失敗）；每群 contact sheet 自己看過

---

## M7–M8 — Stage A 合成（本機保底）

- [x] **M7** 👀 `src/synthetic/copy_paste.py` — **每物件 500 張**
  - **驗證**：每張都有配對 mask 與 `metadata.jsonl` 一行且通過 schema；mask 與影像同尺寸、非全零、
    貼上位置 100% 在 ROI 內；抽 24 張 grid **自己看**，確認無矩形接縫、瑕疵沒貼到背景

- [x] **M8** 👀 `src/synthetic/procedural.py` — **每物件 500 張**，另跑 `--no-real-stats` 版
  - **驗證**：同 M7 的自動斷言；預設版 mask 面積／長寬比落在真實 5–95 百分位內、超出比例 <10%；
    `--no-real-stats` 版斷言**從未開啟** `real_mask_stats.json`（[ADR-011](docs/decisions.md#adr-011)）；grid 自己看

---

## M9–M12 — Stage B 合成（課程復刻主秀）

- [x] **M9** 👀 `src/synthetic/mask_placement.py` — ROI 偵測 → 仿射 → 放置 + 視覺化檢查圖
  - **驗證**：放置 mask 100% 在合法 ROI 內、不與其他 mask 重疊、面積在真實分布範圍內；
    抽 24 張視覺化圖 **自己看**，ROI 明顯抓錯就換方法（例如改以 DINOv2 前景分割為主）
  - **完成證據**：963 個 train-good 背景 × 每張 3 個互斥 placement＝2,889 筆；
    DINOv2/Otsu intersection ROI 全數重建，0 越界、0 重疊、0 分布 outlier、
    0 test-blocklist hit；兩次獨立小跑 8/8 PNG SHA256 相同

- [x] **M10** 👀 **SD2 LoRA 本機 4090 訓練**（[ADR-008](docs/decisions.md#adr-008)）
  - **驗證**：訓練完成並存出可被 `PeftModel` 載回的權重；**記錄實際耗時**
    （>30 分鐘就回報並改回 Colab、更新 ADR-008）；`runs/.../samples/` 的樣本圖**自己看**，
    若每張都跟某張 seed 一模一樣＝ overfit，降 rank 或減 steps 重跑
  - **完成證據**：pcb1 / capsules 各 400 steps、4 checkpoints、type0/type1 輪替 sample；
    訓練 128.66 / 125.65 秒、峰值皆 3.203 GiB；受控中斷／resume 與兩物件
    `PeftModel` 獨立重載通過，詳見 `reports/lora_sd2_report.md`

- [x] **M11** 🙋 `notebooks/01_train_inpaint_lora_sdxl.ipynb`（Colab L4）＋ 本機 smoke test
  - **無人值守**：notebook 與 `--smoke` 本機驗證可自動做完；**實際 Colab 訓練必須等人**
  - **驗證**：`--max-train-steps 1 --smoke` 跑通薄封裝並存出權重檔；斷點續跑分支（有／無 checkpoint）
    各跑一次；notebook 內無明文 token；峰值 VRAM 記進 `instructions_for_me.md`
  - **完成證據**：Colab L4 正式 fresh run 已完成兩物件各 400 steps、四個 checkpoints、
    三份 adapters、8 張 sample 與 fresh `PeftModel` reload；峰值 9.680 GiB。本機兩物件
    one-step smoke（峰值皆 9.619 GiB）與 pcb1 step 1 → 2 跨 process resume（9.663 GiB）
    均通過，四份 base weights、雙 tokenizer、三份 adapters、sample 與 blocklist 的
    CPU verifier 全綠，詳見 `reports/lora_sdxl_report.md`。

- [x] **M12** 👀 `src/synthetic/generate_diffusion.py` — SD2 生成 **500 張／物件** ＋ refine 搜尋
  - **無人值守**：SD2 部分可跑到底；SDXL 權重已由 M11 回收，本 milestone 的正式
    generation／refine 證據仍為 SD2
  - **驗證**：`metadata.jsonl` 每行通過 schema（缺欄即失敗）；同 seed 重跑位元相同；
    `original/` 與 `searched/` 並排 grid **自己看**，確認 refine 真的變好（沒變好就檢討 `refine_score` 權重）
  - **完成證據**：original / searched 各 1,000 筆；searched 逐筆保留 original baseline，
    778 改善 / 222 相同 / 0 退步；兩桶皆 1,000 unique image/mask SHA、0 blocklist hit，
    完整 validator、37 tests、Ruff 與 type-balanced 四欄目視通過，詳見
    `reports/generation_report.md`

---

## M13–M15 — 過濾、指標與交接

- [x] **M13** 👀 `src/filtering/` — 六道過濾 → `filtered/` 與 `unfiltered/` 兩版
  - **驗證**：`reports/filter_report.md` 的漏斗表能從 `metadata.jsonl` 重新聚合出**完全相同**的數字
    （`scripts/verify_filter_report.py`）；每筆被拒樣本的 `reject_reason` 都在 enum 內；
    抽 12 張被拒 + 12 張通過做並排 grid **自己看**，若把明顯好的樣本刷掉就調門檻**並記錄調整過程**

- [x] **M14** 🤖 `src/evaluation/` — `nn_score` / `mnn_score` / FID / KID
  - **驗證（健全性檢查不過就不准往下走）**：真實 crop 自餵 → `nn_score ≈ 1`、KID ≈ 0；
    純雜訊 crop → `nn_score` 明顯低於 τ_low、KID 很大；所有輸入 crop 路徑不在 test blocklist 內；
    `reports/generation_quality.md` 分瑕疵型列出四項指標

- [x] **M15** 🙋 `instructions_for_me.md` 的 M11 交接填滿 ＋ Phase 1 驗收
  - **驗證**（[ADR-019](docs/decisions.md#adr-019)）：SDXL notebook 具備五節（上 Colab
    方式／Runtime／Secrets／實測時數與 compute units／跑完抓哪些檔放回
    `results/colab/` 的哪個路徑）；SegFormer notebook 的建立、smoke 與五節具體交接由
    M18 負責，避免 M15 ↔ M18 循環依賴；`scripts/verify_phase1.py` 紙上走一遍無缺口，
    M0–M15 全勾、每個里程碑至少一筆 commit、所有歷史身份僅 `kuotunyu`

---

# Phase 2 — 五組對照實驗、Demo 與發佈

> 協定見 [docs/experiment_protocol.md](docs/experiment_protocol.md)，發佈規格見 [docs/publish_spec.md](docs/publish_spec.md)。
> **Phase 1 全綠之前不准開始 Phase 2。**

- [x] **M16** 🤖 分類實驗（本機 4090，約 40 run）
  - **驗證**：`results/classification.csv` 每列可追回 `splits/split_manifest.json` 的資料清單；
    `df-guard` 的防洩漏檢查表（[experiment_protocol.md §7](docs/experiment_protocol.md)）全綠；
    所有組的超參相同且來自 Real-only 在 validation 上的調參；每組的真實樣本曝光次數已記錄

- [x] **M17** 🤖 生成品質表 ＋「品質分數 vs 下游提升」散點圖
  - **驗證**：散點圖的每個點都能對回 `results/classification.csv` 與 `reports/generation_quality.md` 的原始值；
    若兩者不相關，**如實寫進結論**而不是換指標重畫

- [x] **M18** 🤖 分割 notebook（SegFormer-B0）＋ 本機 smoke test ＋ 寫進 `instructions_for_me.md`
  - **驗證**：`--group` 參數能切換九組中的任一組；本機 1-step smoke 跑通；
    notebook 與 `instructions_for_me.md` 具備五節具體交接（上 Colab 方式／Runtime／
    Secrets／預估時數與 compute units／跑完抓哪些檔放回哪個路徑）；第 9 組在文件中
    明確標示「引用第 4 組，不重跑」（實跑 8 組 × 2 物件）

- [x] **M19** 🙋 **使用者在 Colab 跑分割**，產出放回 `results/colab/`
  - **驗證**：對照 `instructions_for_me.md` 的預期清單盤點產出，缺件就停下來列清單問使用者，不要硬做

- [x] **M20** 🤖 分割分析
  - **驗證**：`results/segmentation.csv` **從 raw metrics 檔重新聚合**，不抄 notebook 畫面上的值；
    第 6 組（程序化-only）的口徑照 [ADR-011](docs/decisions.md#adr-011) 的正式措辭陳述

- [x] **M21** 👀 圖表全套 → `reports/figures/`
  - **驗證**：[experiment_protocol.md §10](docs/experiment_protocol.md) 列的每張圖都存在；
    **每張都自己打開檢視**；頭號圖 `real_scaling_curve.png` 能直接回答「合成資料相當於幾張真實瑕疵」

- [x] **M22** 👀 Gradio demo（本機）＋ `assets/demo.gif`
  - **驗證**：上傳真實 test 影像能同時輸出機率、mask、heatmap 與延遲；GIF **自己看過**確認畫面正確
    `scripts/record_demo_artifacts.py` 必須從 frozen highshot test 與 formal raw-bound
    checkpoints 原子產出四幀 GIF／`reports/demo_validation.json`，不以假資料或截圖替代

- [x] **M23** 🤖 README 完成 ＋ `scripts/verify_readme.py`
  - **驗證**：`verify_readme.py` 跑過，README 每張表的數字都能從 `results/` 重算出來且完全相符；
    負面結果（若有）已寫進 Limitations；防洩漏聲明與授權表齊全

- [x] **M24** 🙋 發佈（**轉 public 前一定要使用者過目**）
  - **驗證**：照 [docs/publish_spec.md](docs/publish_spec.md) 的檢查表逐項打勾；
    全 repo 掃過無 API key／絕對路徑／個資；commit 無 `Co-Authored-By`；
    HF dataset card 與 model card 完整；使用者說 OK 才轉 public

---

# Phase 3 — 公開互動展示與 v2 改良實驗

> v1.0.0 的凍結表格、負面結果與結論保持不變。Phase 3 是明確標示的後續延伸，
> 不得用新結果覆寫原實驗，也不得用 test set 選擇 v2 超參。

- [x] **M25** 🤖 公開 Hugging Face Gradio Space
  - **驗證**：四個展示 checkpoint 的大小與 SHA256 在載入前 fail closed；匿名訪客可完成
    pcb1／capsules 上傳、分類、mask、heatmap 與 provenance 檢視；預設 CPU Basic，
    只有實測不敷使用才升級 GPU；頁面與 bundle 完整附帶研究用途及第三方授權界線

- [x] **M26** 🤖 v2 低成本 pilot：修正合成樣本淹沒真實瑕疵的取樣問題
  - **驗證**：先凍結候選策略與 validation-only 選模規則；以 real-only、v1 mixing 與
    domain-balanced／curriculum mixing 做小規模對照，記錄每域曝光量、walltime、VRAM、
    validation Macro-F1／AUROC；pilot 未過門檻就停止，不消耗 A100 額度做正式重跑

- [x] **M27** 🙋 v2 confirmatory run 與公開 addendum（**gate 未通過，依預註冊正確停止而未執行**）
  - **驗證**：只有 M26 在兩物件 validation 都未退步且至少一項主要指標改善才啟動；
    以凍結設定跑 3 seeds、最後一次性評估 test，輸出獨立 `results/v2/` 與報告；
    README 清楚並列 v1 與 v2，不能把探索性 pilot 當成預註冊主結果

- [x] **M28** 🤖 公開 Demo 正體中文 UI／UX 改版
  - **驗證**：介面以正體中文引導「選物件 → 上傳 → 檢測」，一般文字至少 16px、
    輔助文字至少 14px；initial empty state 不顯示誤導性空白結果；真實影像可完成
    classification、mask、heatmap 與 provenance；桌機／手機無水平溢位、console error，
    核心文字與控制項對比達 WCAG 2.2 AA

- [x] **M29** 🤖 公開 Demo 資訊密度與可讀性精修
  - **驗證**：一般內文與控制項以 18px 為基準、輔助文字至少 16px；移除重複的
    三步驟卡片與表單編號，initial state 僅呈現可操作輸入，結果區採 progressive
    disclosure；桌機首屏可看見完整主要操作、手機與 200% zoom 無水平溢位，真實影像
    推論仍可完成 classification、mask、heatmap 與 provenance

- [x] **M30** 🤖 公開 Demo 工整格線與克制馬卡龍視覺精修
  - **驗證**：一般內文與控制項以 20px 為基準、輔助文字至少 18px；三步驟改為等寬
    全幅流程帶，以薄荷綠、粉藍與杏桃區分任務階段，淡紫只用於結果；非功能性框線與
    大圓角移除，輸入／結果頂端對齊並依內容高度收合；桌機 1680×958、平板
    768×1024、手機 390×844 均無水平溢位或 console error

- [x] **M31** 🤖 公開 Demo 物件選擇與範例體驗精修
  - **驗證**：桌機三個流程步驟在等寬欄內精準置中；物件選擇改為具 PCB／CAP
    識別標記、清楚選取狀態與鍵盤 focus 的雙卡片；殘留的小型標籤與輔助文字提高到
    18–20px；提供 5 張具 CC BY 4.0 attribution 與 SHA256 inventory 的 VisA 範例，
    點選後自動帶入正確物件與影像；桌機與手機均可完成範例載入及推論且無水平溢位

- [x] **M32** 🤖 公開 Demo 流程精準對齊與結果報告重構
  - **驗證**：三個流程欄位固定為相同寬度與 80px 高度，編號欄、文字欄與垂直中心
    的誤差皆小於 1px；完成推論後輸入區維持全寬，結果另以全寬報告呈現模型判定、
    自訂 confidence bars、mask 與 heatmap；桌機 1680×958 與手機 390×844 無水平
    溢位，範例載入、結果顯示與定位影像皆可正常完成

- [x] **M33** 🤖 公開 Demo Mask 結果引導與探索模式
  - **驗證**：預設先執行正式 threshold 0.50；若 Mask 全黑，結果區直接說明不是操作
    失敗、顯示最高 pixel probability，並提供「探索弱訊號 0.20」選項。使用者切換後
    自動重新產生 Mask，不需返回輸入區或理解 threshold；正式與探索狀態有不同文案、
    圖例與研究邊界。Playwright 桌機與手機可來回切換且無水平溢位

- [x] **M34** 🤖 公開 Demo Probability heatmap 可解讀性
  - **驗證**：Binary mask threshold 與 Heatmap 顯示改為兩個獨立控制；Heatmap 提供
    「疊加原圖／純 Heatmap」模式、固定 0–1 色階與此張影像最高 pixel probability。
    單元測試確認切換 threshold 只改變 Mask、Heatmap 像素保持一致；Playwright 確認
    純 Heatmap 顯示確實改變，桌機與手機均無水平溢位

- [x] **M35** 🤖 乾淨原始碼重現與持續驗證
  - **驗證**：從 Git 封存檔建立全新 Python 3.12／uv locked 環境，完整 lint、pytest
    與 publication audit 全綠；所有產生後會被 SHA256 綁定的文字證據固定使用 LF，
    確保 Windows 工作目錄與 GitHub Source ZIP 位元組一致；GitHub Actions 使用完整
    Git 歷史、唯讀權限、不可變 action SHA 與 frozen lock，在每次 main push／PR
    自動重跑相同 gate，並稽核作者、committer 與 co-author trailer

- [x] **M36** 🤖 GitHub 正體中文資訊架構與 Repository 重新命名
  - **驗證**：Repository 更名為 `defectforge-visa-synthetic-data`，名稱直接揭露
    Synthetic Data 主題；README、系統架構圖、About、v1.0.0／v1.1.0 Release title
    與 notes 均以正體中文為主，專有名詞保留原文；Semantic Versioning tag 不改名；
    舊 GitHub URL 可 redirect，新 URL、CI badge、clone command 與公開連結皆有效；
    完整 pytest、publication audit、GitHub Actions 與 Contributors 稽核全綠

- [ ] **M37** 🤖 Release Hardening、引用資訊與永久封存
  - **驗證**：根目錄 `LICENSE` 可由 GitHub 辨識為標準 MIT，第三方資產義務移至
    `THIRD_PARTY_NOTICES.md`；`CITATION.cff`、Security／Contributing Policy、
    Issue Form、Changelog 與 1280×640 Social Preview 通過自動與目視驗證；
    Hugging Face Dataset／Model／Space Metadata 同步新 Repository URL；`main`
    禁止 Force Push／刪除並啟用 Dependency Alerts／Private Vulnerability Reporting，
    不開 Bot 自動 Commit；完整 CI 全綠後從精確 Commit 發布 `v1.2.0`，最後再次確認
    Release、License、公開連結與 Contributors 僅有 `kuotunyu`

---

## 交接與追蹤

每完成一個里程碑：跑驗證欄 → 全綠才勾選 → 追加 [docs/worklog.md](docs/worklog.md) 一筆 →
建立該階段的 skill（若尚未建）→ `git commit` → 給使用者「換你做」清單。

無人值守時額外：把當晚結果寫進 `reports/handoff/<date>.md`（格式見
[autonomy_policy.md](docs/autonomy_policy.md)）。
