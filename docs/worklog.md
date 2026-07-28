# Worklog

> **每個里程碑結束追加一筆，只追加不改寫。** 這是「隔一段時間回來能想起來」的主力文件。
> 固定格式：做了什麼／決策／驗證／未決與風險／下一步／換你做。
> 恢復脈絡的最快方式：呼叫 `/defectforge` skill，它會讀這份 + PLAN.md + decisions.md。

---

## 2026-07-27 — Session 01：專案初始化與文件凍結（M0）

### 做了什麼
- 讀完上層 `SDG_portfolio_plan_v2.md`（三案作品集總戰略 v2.1）
- 從 `gtc_tw_lab.pptx`（17 張投影片）萃取完整課程方法論，寫成 `docs/methodology.md` 的逐格對照表
- 上網查證：VisA 大小與內容、spot-diff 協定、inpainting 底模授權、套件版本、Colab 費率
- 本機環境健檢：4090 / uv / git / gh / 磁碟 / OneDrive
- `git init`（branch `main`）＋ 建立完整目錄骨架
- 寫入 `CLAUDE.md`、`PLAN.md`（M0–M15，每項含驗證方法）、`README.md` 骨架、
  `instructions_for_me.md` 骨架、`.gitignore`、`LICENSE`、`pyproject.toml`、`configs/paths.yaml`
- 寫入 `docs/` 九份文件與 ADR-001~006
- 建立第一個專案級 skill：`.claude/skills/defectforge/`（orchestrator，對應課程 `anomalygen`）

### 決策
六筆 ADR，全文見 [decisions.md](decisions.md)：

| ADR | 決策 | 誰決定 |
|---|---|---|
| 001 | 底模 SD2-inpainting 主線 + SDXL-inpainting-0.1 對照，SD2 先綠燈 | 使用者 |
| 002 | 瑕疵分型用非監督分群 → 使用者確認命名 → 凍結 | Claude（使用者授權） |
| 003 | few-shot = k=10 瑕疵 + train pool 全部正常圖，第 1–4 組真實資料完全相同 | Claude（使用者授權） |
| 004 | crop-to-ROI → 模型解析度 inpaint → 全解析度縫合回去 | Claude |
| 005 | 大檔放 `D:\sdg-data\01-defectforge`，HF 快取留 C: | 使用者 |
| 006 | `nn_score` / `mnn_score` / FID / KID 的明確定義（在 crop 上算） | Claude |

### 查證到的關鍵事實（改變了原始計畫）
1. **VisA 公開版沒有 per-image 瑕疵類型標註** → 原 prompt 的「每種瑕疵抽 K=10 張」不可執行 → ADR-002
2. **官方 `2cls_fewshot` 是 k=5/10 從兩個類別各抽** → 照字面走只有 20 張真實圖，
   合成會退化成單張圖的變體 → ADR-003 重新界定 few-shot
3. **VisA tar = 1.80 GB**（1,929,840,640 bytes，HTTP HEAD 實測），未達 2GB 報備門檻
4. **VisA 瑕疵明顯小於 MVTec AD** + 影像約 1500×1000 → 必須 crop-to-ROI → ADR-004
5. 環境：Desktop 未被 OneDrive 接管（否則 40GB 影像會被同步上雲）

### 驗證
- 15 份 md、79 個相對連結全部指到存在的檔案（0 broken）；`decisions.md` 的 6 個 ADR anchor 全部對得上
- grep 全部 md：無 `/mnt/c`、無 `~/sdg-portfolio`、無金鑰前綴字串（僅有的命中是「禁止使用」規則本身）
- `pyproject.toml` 與 `configs/paths.yaml` 都能被解析器讀進去
- **驗證抓到一個 bug**：`.gitignore` 的 `data/` 沒有 root-anchored，把 `src/data/` 一起吃掉了 →
  改成 `/data/`、`/runs/`、`/results/*`；同時補上 `.gitattributes` 統一換行為 LF（repo 要在 Colab/Linux 上跑）
- 33 個檔案全部進 stage，`src/data/.gitkeep` 確認已被追蹤

### 未決與風險
| 項目 | 說明 | 何時解決 |
|---|---|---|
| ~~GitHub 帳號~~ | ~~`gh` 登入的是 `tun0000`，但發佈流程文件寫 `kuotunyu`~~ → **已解決，見下方補充 1**（就是 `kuotunyu`，我誤讀了 gh 的快取名稱） | ✅ 2026-07-27 |
| ~~Colab L4 費率~~ | ~~L4 未查到確切值~~ → **L4 約 2.5–5.0 CU/hr（中位約 4.0）**，T4 約 1.5–2.0、A100 約 13–15。來源是第三方彙整、Colab 官方不公佈費率，**M15 仍要實測校正** | 部分解決 2026-07-27 |
| ~~套件版本 / CUDA index~~ | ~~torch 2.13.0 但文件寫 cu128~~ → **已修正為 cu130**。實測 cu128 index 最高只到 torch 2.11.0，原本「cu128 ＋ 2.13.0」的組合不存在 | ✅ 2026-07-27 |
| **transformers v5 相容性** | transformers 已進入 v5（5.14.1），是破壞性改版（image processor 改名、預設 dtype 改為 `"auto"`）。**依賴解析已實測無衝突**（暫存區 `uv lock` 得 transformers 5.14.1 ＋ diffusers 0.39.0 ＋ peft 0.19.1 ＋ torch 2.13.0+cu130，175 套件）。但解析成功 ≠ 執行期相容 | M1 除了 `uv lock` 還要實際 import 並跑最小推論 |
| ~~commit 信箱洩漏學校信箱~~ | 早期 3 筆 commit 曾含 `[redacted-school-email]`，其後已完成歷史改寫。2026-07-27 提交後稽核：所有目前 commits 的 author／committer shortlog 只剩 `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`，`Co-authored-by` 0、remote 0。repo-local 身分亦固定為同一 noreply 地址 | ✅ 2026-07-27 |
| 分型可用性 | 10 張 seed 分群後每型可能只剩 2–4 個元件，trigger token 可能學不起來 | M6 看實際分群結果決定是否啟動 fallback |
| SD2 vs SDXL 額度 | 兩個底模都做會吃掉較多 Colab units | M15 估算後回報，必要時把 SDXL 排到下個月 |

### 下一步
**M1** — 建立 uv 虛擬環境並鎖版（Python 3.12、torch **2.13.0+cu130**）。
開工前先重新查證各套件當時最新版本，並確認 `diffusers` 0.39 與 `transformers` v5 能否共存。

### 換你做
0. ~~改寫前 3 筆 commit 的作者信箱~~ → ✅ 已完成歷史改寫；所有目前 commits 的 author
   與 committer 均只剩 `kuotunyu` noreply 身分，且沒有 co-author trailer。
   repo-local 設定可保護本專案；其他新 repo 仍須各自確認身分設定。
1. ~~確認 GitHub 帳號~~ → ✅ 已確認是 `kuotunyu`（見補充 1）
2. **確認 Hugging Face 帳號**：Phase 2 上傳合成資料集與 LoRA 權重要用哪個（文件記載 `steven0226`）
3. **翻一下這幾份文件有沒有跟你想的不一樣**，尤其：
   - [`docs/decisions.md`](decisions.md) 的 ADR-002 與 ADR-003（我替你決定的兩題）
   - [`PLAN.md`](../PLAN.md) 的 M0–M15 拆法
   - [`docs/methodology.md`](methodology.md) 的「刻意不復刻」那一節
4. 準備好 Colab Secrets 裡的 `HF_TOKEN`（M10 才會用到，先確認存在即可）
5. 我要開始 M1 之前，會先問你要不要下載 VisA（1.80 GB）

---

### 補充 1（同日）— Git 署名規則與 GitHub 帳號釐清

**使用者要求**：GitHub 的 Contributors **只能出現 `kuotunyu` 一個人**，不要出現 Claude 或任何其他人。

**釐清**：先前把 `gh auth status` 顯示的 `tun0000` 當成「另一個帳號」是**我誤判**。
`gh api user` 回傳 `kuotunyu`（id 61350295）——那是同一個帳號，`tun0000` 只是 gh 設定檔裡的
**舊快取名稱**（帳號改過名）。以 `gh api user` 為準。

**稽核結果**：M0 那筆 commit 本來就乾淨——
author/committer 當時都是 `kuotunyu <[redacted-school-email]>`，且無 `Co-Authored-By` trailer。

**已落實的防護**：
1. [CLAUDE.md](../CLAUDE.md) 新增「Git 署名規則（不可違反）」一節：禁止 `Co-Authored-By`、
   禁止 `🤖 Generated with Claude Code` 之類署名（commit message 與 PR 內文都算），
   並附 commit 前的自檢指令
2. 設定 **repo-local** `user.name` / `user.email`，不依賴 global 設定，避免日後漂移
3. [environment.md](environment.md) 的 gh 帳號欄位已更正

**日後注意**：`publish-repo` 之類的發佈流程若預設會加產生器標記，要在執行前關掉。

---

## 2026-07-27 — Session 02：Phase 2 規劃、split 洩漏修正、按圖施工三本

### 做了什麼
- 讀完 Phase 2 的 prompt（主題 3），把它展開成完整協定與里程碑 M16–M24
- **下載兩份官方 split CSV 實際計算交集**，抓到 ADR-003 的跨 partition 洩漏
- 新增 ADR-007 ~ ADR-012 共六筆決策
- 新增三份「按圖施工」文件：`autonomy_policy.md`、`interfaces.md`、`publish_spec.md`
- `PLAN.md` 全面改寫：加 Phase 2 里程碑，每個里程碑標上 🤖 / 👀 / 🙋 值守屬性
- `experiment_protocol.md` 全面改寫成可執行的 Phase 2 規格
- 同步修正 `CLAUDE.md`、`data_protocol.md`、`synthesis_spec.md`、`environment.md`、
  `skills_roadmap.md`、`README.md`

### 🔴 抓到的洩漏 bug（本次最重要的一件事）
ADR-003 原本讓 Full-real 上限組用 `2cls_highshot` train、其餘組用 `2cls_fewshot` test。
下載官方 CSV 計算後確認：

```
highshot TRAIN(anomaly) ∩ fewshot TEST(anomaly) = 40   ← 測試瑕疵的一半
highshot TRAIN(normal)  ∩ fewshot TEST(normal)  = 401 (pcb1) / 241 (capsules)
```

照原計畫跑，**Full-real 那組的訓練集會包含一半的測試瑕疵**，數字會很漂亮但完全是假的。
若不是在寫 Phase 2 規格時把兩個 phase 的資料流對起來看，這個 bug 會一路跑到實驗結束才爆。

**修正**（[ADR-007](decisions.md#adr-007)）：改用 `2cls_highshot` 當唯一基底。
意外收穫是兩套切法是巢狀的（`fewshot_train ⊂ highshot_train`、`highshot_test ⊂ fewshot_test`），
所以免費得到一條「真實瑕疵 10 → 20 → 60 張」的縮放曲線——
這讓我們可以回答「**合成資料相當於多少張真實瑕疵**」，比原本只有單一上限點強得多。

### 決策（ADR-007 ~ 012）
| ADR | 決策 | 誰決定 |
|---|---|---|
| 007 | 基底 split 改 `2cls_highshot`，取代 ADR-003 的基底部分 | 使用者 |
| 008 | SD2 LoRA 本機 4090 訓（估 20–30 分）、SDXL 上 Colab；訓練邏輯單一實作，notebook 只是薄封裝 | 使用者 |
| 009 | 分類 = 每物件一個二元 ConvNeXt-Tiny；分割 = 五組鐵律 + 四組來源消融（實跑 8 組） | Claude |
| 010 | 合成量掃描改用絕對值 {125, 250, 500}；SD2 生 500／物件、SDXL 250 | Claude（使用者授權） |
| 011 |「程序化-only」正名為「零真實瑕疵**像素**」，主動揭露用了真實 mask 統計量，並提供 `--no-real-stats` 對照組 | Claude |
| 012 | 無人值守 = 跑到底，任何驗證失敗即停 | 使用者 |

### 順手修掉的另外三個問題
1. Phase 2 prompt 的「合成量 0.5x/1x/2x（相對真實正樣本）」算出來只有 **5/10/20 張**
   （真實正樣本只有 10 張），掃不出訊號 → 改用課程自己的絕對刻度 125/250/500
2. 分割實驗原本只有 4 組合成來源、**沒有 Real-only 基準**，「程序化-only」那組會沒東西可比
   → 補成五組鐵律 + 四組來源消融
3. 「程序化-only ＝ 零真實瑕疵」字面上不成立（用了真實 mask 的統計量）→ 正名並揭露

### 驗證
- 21 份 md、相對連結全部指到存在的檔案；12 個 ADR anchor 全部對得上
- grep 無 WSL 路徑、無金鑰前綴
- split 交集的四項斷言由實際下載的 CSV 計算得出，數字已寫進 ADR-007 與 `data_protocol.md`

### 未決與風險
| 項目 | 說明 | 何時解決 |
|---|---|---|
| Test 變小 | 採 highshot 基底的代價：每物件只有 40 張測試瑕疵，指標變異變大 | Phase 2：所有表附樣本數，Real-only 與最佳 Filtered 組**必須**補 3 seeds |
| SD2 本機訓練時間 | 估 20–30 分鐘但未實測；超過 30 分鐘要改回 Colab | M10 實測 |
| Colab L4 費率 | 仍未查到確切 CU/hr | M15 到 Colab 頁面實測 |
| 分型可用性 | 10 張 seed 分群後每型可能只剩 2–4 個元件 | M6 |
| 套件版本 | 需在 M1 重新查證後 `uv lock` | M1 |
| Cosmos stretch goal | A100 約 10–15 CU/hr，會明顯吃額度 | Phase 2 收尾時**主動問**使用者，不准自作主張啟動 |

### 下一步
**M1** — 建立 uv 虛擬環境並鎖版。開工前先重新查證各套件當時最新版本。

### 換你做
1. **翻一下 [ADR-007](decisions.md#adr-007)** —— 這是今天最重要的修正，確認你認同「用 highshot 當基底、
   換來 test 從 80 張縮到 40 張瑕疵」這個取捨
2. **看一眼 [autonomy_policy.md](autonomy_policy.md) 的 §4「必須停下來等人」**，
   確認清單符合你的預期（尤其「>2GB 下載」「Colab」「任何 push」）
3. 我開始 M1 之前會先問你要不要下載 VisA（1.80 GB）；
   如果你想讓我今晚就一路跑到 M14，**現在就可以先授權下載**，我會照 🤖/👀 標記跑到 M10，
   停在 M11（SDXL 需要你上 Colab）

---

## 2026-07-27 — Session 03：M1 環境鎖版與 Git 歷史清理

### 做了什麼
- 稽核本機 repository：無 remote、工作樹原本乾淨、repo-local 身分已鎖為
  `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`
- 改寫既有 commit metadata，author / committer 統一為 `kuotunyu` 的 GitHub noreply；
  沒有加入 `Co-Authored-By:` 或工具署名
- 重新查證官方 cu130 index 與 PyPI 最新版本，補上遺漏的 `timm` 直接相依
- 產生 `uv.lock`，建立 uv 管理的 CPython 3.12.13 `.venv`
- 移除無現代 Windows wheel 的 `noise` 1.2.2；程序噪聲改由 NumPy/scikit-image 實作
- 更新 `CLAUDE.md`、環境、發佈與 troubleshooting 文件的實測資訊

### 決策
- 沒有新增 ADR：移除 `noise` 是實作相依調整，不改方法、資料、CLI 或實驗契約
- 不修改 global Git 設定，以免干擾同時進行的 `2_SafeSynth` / `3_FormosaNLU`；
  本 repo 的 local 設定已足以保證 Contributor 歸屬
- `df-setup` 同時涵蓋 M1–M2，依 skills roadmap 等 M2 真正跑通後再建立，避免先寫未驗證 SOP

### 驗證
- `uv lock --python 3.12`：175 packages 解析成功
- Python：3.12.13
- torch：`2.13.0+cu130`；CUDA runtime 13.0；`torch.cuda.is_available() == True`
- GPU：`NVIDIA GeForce RTX 4090`；capability `(8, 9)`
- `diffusers` 0.39.0、`transformers` 5.14.1、`peft` 0.19.1、`timm` 1.0.28、
  `accelerate` 1.14.0、OpenCV 5.0.0 與其餘 M1 套件全部 import 成功
- `from diffusers import AutoPipelineForInpainting` 成功；只有上游 Siglip2 Fast 名稱的
  deprecation warning，沒有 runtime 例外

### 未決與風險
| 項目 | 說明 | 何時解決 |
|---|---|---|
| M2 下載授權 | VisA tar 為 1,929,840,640 bytes；M2 明確標記 🙋，無人值守不可自行下載 | 使用者醒來後授權 |
| `df-setup` skill | M1 已驗證，M2 尚未實跑，因此不先把未驗證下載流程寫成 SOP | M2 完成時 |
| Transformers 棄用警告 | 目前僅 warning；若日後升級變成錯誤再評估版本取捨 | 首次 DINOv2 實跑 |

### 下一步
**M2** — 使用者明確同意後，下載 VisA 1.80 GB 到 D:、核對精確 bytes / SHA256，
解壓並驗證 pcb1 / capsules 的 normal / anomaly 張數。

### 換你做
1. 回覆是否授權執行 M2 的 VisA 1.80 GB 下載。
2. 不需要建立 GitHub repo；目前仍保持純本機，等你醒來再決定。
3. Hugging Face 帳號 `steven0226` 與 Colab Secrets 可留到 M10/M15 前再確認。

---

## 2026-07-27 — Session 04：M2 VisA 落地與可重跑 setup

### 做了什麼
- 使用者明確要求不要因保守授權邊界停下，視為 M2 下載授權並恢復持續執行
- 從 AWS 官方 URL 下載 VisA tar 到 D:，精確大小 1,929,840,640 bytes
- 新增 `src/common/paths.py`，讓所有腳本從 `configs/paths.yaml` 解析絕對路徑
- 新增 `scripts/download_visa.py`：續傳、bytes/SHA256、安全解壓、inventory 斷言與 JSONL log
- 新增 tar path traversal 與 path loader 測試
- 建立並通過 validator 的 `.claude/skills/df-setup/` 已驗證 SOP
- 把官方 spot-diff commit `2a692ab575001cbde74d402d897a7286086c6199` clone 到 D: 上游快取，
  供 M3 直接執行官方 `prepare_data.py`

### 決策
- 原始資料固定放在 `D:\sdg-data\01-defectforge`；repo 只提交 checksum 與程式碼
- 官方 tar 沒有外層 `VisA/`，下載器直接以 `${visa_raw}` 為 extraction root
- 刪除第一次誤解壓的重複副本被破壞性操作護欄拒絕；D: 空間充足，因此保留而不繞過護欄

### 驗證
- tar bytes：`1,929,840,640`（精確相符）
- SHA256：`2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`
- 解壓：12,122 tar members；正確資料 12,037 files / 1,920,559,633 bytes
- pcb1：1,004 normal / 100 anomaly / 100 masks
- capsules：602 normal / 100 anomaly / 100 masks
- `ruff check` 全綠；pytest 2 passed；`df-setup` skill validator 通過

### 未決與風險
| 項目 | 說明 | 何時解決 |
|---|---|---|
| D: 重複副本 | 首次誤解壓約多佔 1.92 GB；路徑明確但刪除被環境護欄拒絕 | 使用者醒來後可手動刪除，非阻塞 |
| upstream 固定 | M3 必須記錄 spot-diff commit 與兩份 split CSV SHA256 | M3 |

### 下一步
**M3** — 執行官方 spot-diff `prepare_data.py` 產生 highshot/fewshot，並跑八格張數、
集合關係與 mask 對應四類斷言。

### 換你做
目前沒有；我會直接繼續 M3。GitHub repo、remote、push 仍等你醒來。

---

## 2026-07-27 — Session 05：M3 官方 split 產生與獨立驗證

### 做了什麼
- 新增 `scripts/prepare_splits.py`，只包裝官方 spot-diff `prepare_data.py`，不重寫上游切分邏輯
- 依官方行為修正 prepared 路徑：`save-folder` 下由上游自動附加
  `2cls_fewshot/` 與 `2cls_highshot/`
- 完整產生兩套 12 類 VisA PyTorch 目錄；DefectForge 對 pcb1 / capsules 執行鎖定驗證
- 新增 `reports/split_preparation.json`，記錄上游 commit、三個來源檔 SHA256、八格計數與斷言
- 新增缺檔負向測試，確保 CSV 指向不存在的原圖時會立即失敗

### 上游凍結
- spot-diff commit：`2a692ab575001cbde74d402d897a7286086c6199`
- `split_csv/2cls_fewshot.csv`：
  `5ca490e84cd7664f9d93ba3d82399d991edb5a6cbcc41359452ad7ec24be354d`
- `split_csv/2cls_highshot.csv`：
  `c3331eede15f2da8a75b380d4fbfb24f75ee036ec9a25147eca898da6e972f09`
- `utils/prepare_data.py`：
  `6e70f97b16b589dc3cf6eab55e104bf13bbd7949909dc2279ebad5cb9f4c1c40`

### 驗證
- few-shot pcb1：train 201 normal / 20 anomaly；test 803 normal / 80 anomaly
- few-shot capsules：train 120 normal / 20 anomaly；test 482 normal / 80 anomaly
- high-shot pcb1：train 602 normal / 60 anomaly；test 402 normal / 40 anomaly
- high-shot capsules：train 361 normal / 60 anomaly；test 241 normal / 40 anomaly
- `highshot_train ∩ highshot_test == ∅`
- `fewshot_train ⊆ highshot_train`
- 所有 anomaly image 與 mask 的 stem 一一相符
- 兩套輸出各 12,021 files / 1,917,965,564 bytes
- `ruff check` 全綠；pytest 3 passed

### 決策
- 保留官方工具的全 12 類輸出以確保可追溯性；訓練與 manifest 仍只納入 pcb1 / capsules
- `configs/paths.yaml` 指向官方真實輸出層級，避免用另一套自訂資料夾命名掩蓋上游行為

### 下一步
**M4** — 以 high-shot 為基底計算 SHA256 / pHash、處理跨 split 近重複群，
凍結 manifest 並建立 test blocklist。

### 換你做
目前沒有；GitHub repo、remote、push 仍等你醒來。

---

## 2026-07-28 — Session 32：M24 GitHub／Hugging Face 正式公開

### 使用者授權與 Contributors 紅線
- 使用者再次明確要求 GitHub Contributors 只能有 `kuotunyu`，並授權在此前提下完成
  專案內後續動作
- 發佈前逐欄稽核 65 個 commits：author / committer 全部是
  `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`；`Co-Authored-By` = 0
- GitHub CLI 登入帳號為 `kuotunyu`（account id `61350295`）

### GitHub 發佈
- 建立並公開 `https://github.com/kuotunyu/01-defectforge-visa`
- 初次推送後 `main` remote HEAD 與本機
  `779b4918f434096a9d39d8793e2e9f1dc3d7ecfd` 相同
- GitHub Contributors API 始終只回傳精確一筆 `kuotunyu / User`；
  contributions 數只隨本專案自己的後續提交增加
- 發佈後再次查詢仍只有 `kuotunyu`，沒有 bot 或第二位 contributor

### Hugging Face 發佈
- HF CLI 身分確認為文件指定的 `steven0226`
- 既有 bundle 重新核對目前 frozen sources：
  - dataset：15,555 bundle files、14,115,088,771 bytes、7,770 image/mask views，
    release manifest SHA256
    `ab1202f69890947e1f0abfeb4837899f618fbb176fe1f5af979854f3f2dcb08d`
  - model：38 bundle files、75,457,038 bytes、10 safetensors，
    release manifest SHA256
    `acc3f8dd260ebb44b85198d5edb80077ed45ecfd64dd98472baf602d0360fa37`
- 先上傳至 private repos，遠端精確 inventory 只比 bundle 多 HF 自動建立的
  `.gitattributes`
- 從遠端重新下載 image、mask、metadata、test blocklist、兩張 card、
  release manifests 與 SD2／SDXL safetensors；SHA256 全部等於本機 manifest
- 驗收後公開：
  - `https://huggingface.co/datasets/steven0226/defectforge-visa-synthetic`
    （15,556 remote files；revision
    `c1c929d34a554133e380567afc306e11e54f4f95`）
  - `https://huggingface.co/steven0226/defectforge-visa-lora`
    （39 remote files；revision
    `d915a7878159e29a6de66285378c7ea0eeb275f2`）
- 最後以 `token=False` 匿名重新讀取 repo inventory 與 README，兩者皆 public、
  ungated、可下載

### 安全與執行狀態
- 上傳腳本第一次因未注入 `HF_TOKEN` 在網路寫入前 fail closed；之後只把本機憑證庫
  token 傳入子程序記憶體，沒有輸出或寫入 token
- 本次發布不使用 GPU，也沒有修改 FormosaNLU

### 下一步／換你做
M24 已完成；目前不需 Colab、GPU 或人工上傳。之後若接受外部 PR，合併前仍必須檢查
commit author，否則 GitHub Contributors 可能不再只有 `kuotunyu`。

---

## 2026-07-28 — Session 30：M19 L4 回收、M20 raw 聚合與 M21 正式圖表

### 做了什麼
- 使用者在 Colab L4 依序完成 pcb1 與 capsules 的八個 SegFormer-B0 formal group；
  每個物件的 notebook validator 均通過並回傳原始結果 ZIP
- 保留兩個 ZIP 原始位元，先以 M20 聚合器執行 Zip Slip、symlink、Windows ADS、
  白名單、檔案數、展開大小與 CRC fail-closed 預檢，再原子匯入
- 從 16 份 raw `training_report.json` 與 `data_manifest.json` 重建
  `results/segmentation.csv`；沒有採信 notebook 畫面或包內暫存 CSV
- 修正 `.gitignore` 的精確單檔 allowlist，讓小型 `results/segmentation.csv`
  進入可發布 source tree；原始 Colab ZIP、匯入目錄與 SafeTensors 仍維持忽略
- 產生 `reports/segmentation_results.md` 與
  `reports/segmentation_validation.json`
- 從 verified classification／segmentation CSV 產生四張 M21 正式圖：
  `real_scaling_curve.png`、`synthetic_volume_curve.png`、
  `main_comparison_table.png`、`segmentation_table.png`

### M19 實測與原始回收
- pcb1：八組合計 2,509.070 秒（41.82 分）；ZIP SHA256
  `d87e3caaf6b090c65bb460ed8da4e1fa779e89d1010b434f2c3cd59e43161321`
- capsules：八組合計 2,321.437 秒（38.69 分）；ZIP SHA256
  `b68a7bfd2f43fbe41627c37f1c5bd61c6292f042c878d4626e3094c1e3c9e8d1`
- 每包恰有 43 個白名單檔案；兩包 CRC 與安全路徑檢查通過
- 帳戶 compute units 前後值未記錄，不從牆鐘時間反推

### M20 驗證與結果
- 16 個 physical run + 2 個 `all_mixed → filtered_syn` logical alias，共 18 列
- 兩物件各八組、alias reruns = 0、frozen manifest／selection SHA 相符
- `procedural_only` 的 real defect train count 為 0；所有 raw report/model hash
  與 exact frozen test inventory 經獨立 validator 重算
- `results/segmentation.csv` SHA256：
  `fbe4f97379e2597713c1c7255bfd6b5df7ac49c948b396469f598d34b03615aa`
- 結果如實保留：Filtered Synthetic 相對 Real-only 的 mean Dice 為負向；
  capsules 的 Dice／AUPRO 有改善，但 pcb1 Dice 明顯下降，不能宣稱跨物件一致提升

### M21 程式與目視驗證
- `reports/phase2_figures_validation.json` status = passed，綁定兩份輸入 CSV 與四張圖 SHA
- 四張新圖均以原始解析度逐張打開；標題、座標、圖例、表格列欄、alias 說明與數字可讀，
  沒有裁切、重疊或錯誤圖例
- Real-only 10/20/60 scaling 與 filtered-synthetic 落點顯示兩物件皆為 `≤10` real；
  不做觀察範圍外的外插

### 未決與風險
- M22 必須用正式分類與分割 checkpoint 對 frozen test 真圖做 GPU 推論並產生 GIF；
  當下 RTX 4090 仍由 FormosaNLU 使用，因此先維持 CPU-only，不搶占跨專案 GPU
- M23 README 必須保留分類與分割的負面結果，不可改選較好看的 group

### 下一步
**M22** — GPU 釋放後執行 deterministic checkpoint selection、四筆 frozen-test
真模型輸出、GIF 與本機 Gradio UI 目視驗證；其後完成 M23 README 與 M24 gate。

### 換你做
目前沒有；M19 已完成。GitHub repo、remote、push 與 public visibility 仍等待 M24
最終過目與使用者明確同意。

---

## 2026-07-27 — Session 21：M16 guarded ConvNeXt trainer 與兩物件真模型 smoke

### 做了什麼
- 執行 `df-guard`：manifest 1,806 張、1,805 個 pHash 群、0 crossing group、
  blocklist 803 unique hashes；GPU free 23.15 GB，D 槽 free 1.675 TB
- 新增 `configs/classifier.yaml` 與 `classifier_data.py`：development 完全不建立 test
  inventory，formal 才載 frozen high-shot test；每筆 synthetic 保留 train-side manifest
  provenance 並逐檔 SHA256 防洩漏
- 新增 ConvNeXt-Tiny @384 trainer：鎖定 HF revision 與 114,374,272-byte
  safetensors SHA256，label-balanced sampler、固定 optimizer steps、real/synthetic
  exposure、Macro-F1／anomaly F1／AUROC／normal FPR
- 新增 ADR-020，預註冊 15 canonical groups、3 個 alias 與 38 個正式實跑；三 seed 主結果
  事前固定為 `real_only` 與 `filtered_syn`，不看 test 後換組
- 依路線圖建立並通過官方 validator 的 `df-downstream` skill

### Smoke 與資料 dry-run
- pcb1 Real-only development：552 train、66 real-only validation、test inventory 0；
  1 step training 7.25 s、peak 2.963 GiB
- capsules：335 train、42 real-only validation、test inventory 0；
  1 step training 3.58 s、peak 2.963 GiB
- CPU-only smoke verifier 驗 locked base hash、data manifest、model hash、exposure 與
  test 未載入，`status=passed`
- `filtered_syn/pcb1` formal dry-run：612 real + 500 deterministic filtered synthetic，
  frozen test 442，train/test SHA overlap 0

### 下一步
只用 Real-only validation 跑共同 learning-rate／step-budget 調校；凍結後才准啟動
formal groups。另需先用 M11 adapters 補齊 `stageB_sdxl/searched` 250 張／物件，
否則 `base_sdxl` 依規格 fail closed。

### 換你做
目前沒有；本機 smoke 與下一階段調校都不需要 Colab 或人工操作。

---

## 2026-07-27 — Session 22：M16 Real-only 超參調校與共同設定凍結

### 搜尋設計
- 只用兩物件 frozen Real-only validation；六個 development run 的 test inventory
  全部為空
- 固定 weight decay 0.05、batch 16、搜尋 budget 300 steps，比較 learning rate
  `1e-5 / 3e-5 / 1e-4`
- 選擇規則事前固定為兩物件 mean Macro-F1 → mean AUROC → 較低 learning rate

### 結果
- `1e-5`：mean Macro-F1 **0.799986**、mean AUROC **0.939352**；pcb1 / capsules
  best step 100 / 75
- `3e-5`：mean Macro-F1 0.755161、mean AUROC 0.941667
- `1e-4`：mean Macro-F1 0.773602、mean AUROC 0.880093
- 凍結共同設定：learning rate `1e-5`、weight decay `0.05`、batch 16、
  final refit **100 optimizer steps**
- `verify_classifier_tuning.py` 從六份 raw report 重算 winner 並確認 config 完全相符，
  `status=passed`

### 下一步
使用唯一 frozen setting 跑 38 個 formal canonical runs；任何 synthetic 組不得調參。
在 `base_sdxl` 前先補齊 250 張／物件 SDXL searched 來源。

### 換你做
目前沒有；formal 分類與 SDXL 生成都在本機 RTX 4090 執行。

---

## 2026-07-28 — Session 23：M16 正式分類矩陣與 M17 品質—下游分析

### M16 正式矩陣
- 執行 `df-guard` 後，以唯一凍結設定完成 38 / 38 個 formal ConvNeXt-Tiny run：
  seed 42 的 15 canonical groups × 2 objects，加上事前指定的 Real-only 與
  Filtered synthetic seed 43 / 44 重複；0 skipped、0 quarantined
- 所有 run 使用相同 base revision／weight SHA、100 optimizer steps、batch 16、
  learning rate `1e-5` 與 weight decay `0.05`；每列記錄 real／synthetic exposure、
  portable data manifest、run signature 與 frozen test inventory
- `scripts/verify_classifier_matrix.py` 獨立重驗 38 個 run artifacts、CSV、train/test
  disjoint hashes、blocklist、aliases 與三 seed 統計，`status=passed`、blocklist hits 0

### M16 結果
- Real-only 三 seed AUROC mean ± std：pcb1 **0.9265 ± 0.0231**，
  capsules **0.8160 ± 0.0224**
- Filtered synthetic 三 seed AUROC mean ± std：pcb1 **0.1677 ± 0.0502**，
  capsules **0.3243 ± 0.0426**；兩物件均穩定劣於 Real-only，不能解讀為改善
- seed 42 的 Full-real AUROC：pcb1 **0.9294**、capsules **0.8583**，確認 trainer
  本身可從真實瑕疵學習；PCB 的 125 synthetic 尚有 0.8779，但 250 synthetic
  降至 0.1336，支持 exposure／domain-gap 風險
- SDXL searched 250-image bucket AUROC：pcb1 **0.1420**、capsules **0.2967**；
  影像觀感或 refine 分數改善不代表下游分類改善

### M17 品質—下游分析
- 依事前固定口徑，只 join 三個 unfiltered source-only generation rows 與對應物件的
  seed-42 source-ablation classifier，Real-only 為同物件基準；沒有看結果後換來源或指標
- 六個點的 Macro-F1 delta 全為負，範圍 **-0.0869 到 -0.4713**
- Pearson `r(KID, ΔMacro-F1) = -0.6084`；只有六點，報告明確標為描述性統計，
  不當成顯著性檢定或因果證據
- `reports/figures/quality_vs_downstream.png` 已用原始解析度目視，標籤、零線、
  座標與六個來源點均正確且無裁切

### 產物與下一步
- `results/classification.csv`
- `reports/classifier_results.md`、`reports/classifier_matrix_validation.json`
- `reports/quality_vs_downstream.md`、`reports/quality_vs_downstream.json`
- `reports/figures/quality_vs_downstream.png`
- 下一步是 M18：先在本機對兩物件跑 SegFormer-B0 one-step smoke 與獨立驗證，
  通過後才封裝 Colab notebook／兩個資料 ZIP 並提供使用者精確操作路徑

### 換你做
目前沒有；M18 本機 smoke 與封裝完成前，不需要開 Colab 或搬檔案。

---

## 2026-07-28 — Session 24：M18 SegFormer smoke、Colab 封裝與精確交接

### 本機 smoke 與相容性修正
- `df-guard` 再次確認 manifest SHA、1,806 images／1,805 groups、0 crossing group、
  blocklist 連結、Git clean／remote 0、D 槽 1,665 GiB free、RTX 4090 23,253 MiB free
- 首次 PCB smoke 在 best-model fresh reload 抓到 Transformers SegFormer 參數 key
  migration：`save_pretrained()` 的 SafeTensors 不能直接用 raw `load_file()` strict-load
  回目前 class；改成官方對稱的 `from_pretrained(local_dir)` 做 key conversion，再將
  current state dict strict-load 回訓練中的模型，checkpoint resume 同步採用相同路徑
- 首次獨立 verifier 又抓到 portable data manifest 使用 `manifest_sha256`，而凍結契約
  要求 `split_manifest_sha256`；修正 producer 與介面文件，沒有放寬 verifier
- 兩個舊 smoke 目錄保留為
  `D:\sdg-data\01-defectforge\runs\seg_smoke\obsolete_pre_contract_fix_*`，未刪除證據

### 通過結果
- pcb1：1 step、552 train／66 Real-only validation／0 test、peak VRAM 0.977 GiB、
  wall 6.46 s
- capsules：1 step、335 train／42 Real-only validation／0 test、peak VRAM
  0.977 GiB、wall 4.81 s
- `scripts/verify_segmenter_smoke.py` 重算 run signature、data manifest SHA、frozen
  split SHA、model SHA 與 development inventory，兩物件 `status=passed`
- `scripts/validate_m18_colab_notebook.py`：11 cells、5 sections、8 formal groups、
  alias reruns 0、duplicated training loop false、literal credentials 0

### Colab 包
- `defectforge_m18_source.zip`：104,900,929 bytes，SHA256
  `d6c9f71b25796c549eba0163f8f9911fccc2a8376ea967bd3258e1749f8a49f8`
- `m18_seg_pcb1.zip`：3,764,604,320 bytes、5,206 files、training blocklist hits 0，
  SHA256 `50a8ab3b7eaf927089d18a5f2c86612e77cb1226dfdd8b34485efbafde4c573c`
- `m18_seg_capsules.zip`：3,948,056,619 bytes、4,804 files、training blocklist hits 0，
  SHA256 `d2c0672f843b5daec2e7798495900faa4c26e25aca8a3f72ab8ec752a19f7688`
- 三個 ZIP 均另跑完整 CRC `testzip()`，全部通過；manifest 在
  `D:\sdg-data\01-defectforge\colab\m18\m18_colab_bundle.json`

### 交接
- `instructions_for_me.md` 已填 Notebook 2 的五項具體資料：C 槽 notebook、D 槽
  三 ZIP、Drive 目的地、T4 ≥14 GiB、不需 Secret、6–12 小時／9–20 CU 規劃估計、
  兩個結果 ZIP 名稱與本機回收路徑
- `df-downstream` skill 只在本機 smoke 通過後才加入 M18 命令、fresh reload 規則、
  8-group／alias 邊界與 Colab 封裝流程

### 下一步／換你做
**M19**：依 `instructions_for_me.md` 上傳 Notebook 2 與三個 ZIP，先跑 pcb1，再跑
capsules；完成後把兩個未改名的結果 ZIP 放到 `results/colab/segmentation/`，並回報
每個 group 的 timings 與實際 CU 差值。

---

## 2026-07-28 — Session 25：M19 等待期間的 M20／M22 fail-closed 補強

### M19 → M20 回收閉環
- 發現操作手冊正確要求使用者只放回兩個未解壓 ZIP，但 M20 聚合器原本只接受已解壓的
  `{object}/runs/`，形成必然的人工作業斷點
- `aggregate_segmentation.py` 現在會直接讀固定檔名
  `m18_seg_results_{pcb1,capsules}.zip`，先防 Zip Slip、symlink、Windows ADS、
  case-insensitive duplicate path、member／expanded-size 異常與 CRC 錯誤
- 每包必須恰有 43 個白名單檔案；核對 notebook validator `status=passed`、8 runs、
  fresh model reload、`all_mixed` alias 0 rerun，以及八組正數 finite timings
- 匯入使用同磁碟 temporary directory + atomic rename；`import_manifest.json` 綁定原始
  ZIP SHA256。重跑可重用同一包，ZIP 改變則 fail closed
- 真實缺檔狀態已實測：缺 `m18_seg_results_pcb1.zip` 時，在建立
  `results/segmentation.csv`、報告或 validation JSON 前停止，三者皆不存在

### 正式結果與 Demo gate 補強
- `validate_segmenter_runs.py` 從「executed steps > 0」收緊為
  `requested == executed == frozen 500`；防止不完整 formal run 通過公平性驗證
- M22 Demo 改用 M18 已驗證的 `from_pretrained(local_dir)` SegFormer key conversion，
  不再 raw strict-load 版本不相容的 SafeTensors keys
- 補上 direct-file Windows entrypoint bootstrap；實測
  `uv run python src/inference/demo_gradio.py --help` 成功且 `--share` 預設 off
- README 與 CLAUDE 狀態更新為 M16 已驗證、M19 分割執行中；數字區仍保持原子 TBD，
  不在 segmentation 缺失時提前寫半份結果

### 驗證與邊界
- 相關 22 tests passed、Ruff passed
- `verify_readme.py` 與 `build_phase2_figures.py` 在缺
  `results/segmentation.csv` 時均先停止，未留下 README validation 或任何 M21 正式圖
- M19、M20、M22 仍未勾選；正式結果 ZIP、raw aggregation 與真模型 UI/GIF 缺一不可

### 換你做
仍是 M19：完成兩個 Colab 物件並放回兩個未改名 ZIP；不需自行解壓。

---

## 2026-07-28 — Session 26：M19 等待期間的 M21／M24 發佈前封裝補強

### M21／README fail-closed
- `build_phase2_figures.py` 現在先驗 38 筆分類與 18 筆分割 exact logical rows，
  四張圖在同磁碟 temporary directory 全部成功後才移入正式目錄
- 用真 M16 數字與僅在 `%TEMP%` 的 dummy segmentation 做版面 preflight；
  四張圖逐張看過，沒有裁切。dummy 圖未進 repo，M21 仍未勾選
- `verify_readme.py --write` 改為先在記憶體完成全部 block／Limitations／TBD 驗證，
  通過才原子寫入；失敗測試證明 README byte-for-byte 不變

### HF 本機發佈包
- 修正 dataset card／publish spec：只有 M13 正式三來源池具 filtered/unfiltered；
  norealstats、SD2 original 與 SDXL original/searched 明列為未篩選 ablation/diagnostic
- 新增 `scripts/package_hf_release.py`：預設只讀 inventory，`--build` 才在 D 槽以
  staging + atomic rename 建包；不建 HF repo、不連網、不改 visibility
- 真實資料逐檔 SHA 與 test blocklist 驗證通過：6,000 個獨立生成樣本、
  7,770 個公開 sample views、7,770 masks、test hits 0
- dataset payload 14,111,746,980 bytes；15,546 payload files 用 hardlink 建立，
  僅 8 個跨磁碟 card/split 檔複製
- model payload 75,449,968 bytes；SD2／SDXL × pcb1／capsules 的 10 個
  SafeTensors adapter 均與正式 validation hash 相符
- 本機輸出：`D:\sdg-data\01-defectforge\publish\{hf_dataset,hf_model}`

### 上傳 gate 與 Contributors 紅線
- `upload_hf.py` 上傳前必須逐檔吻合 `release_manifest.json`；封裝後任一 payload
  改動會 fail closed
- dry-run 通過並明列 `creates_or_updates_private_repositories=false`；
  `reports/hf_upload_plan.json` 只保存統計與 manifest SHA，不重複 15k 筆 file rows
- 真 Git repo 整合測試證明其他 author／committer 與真
  `Co-Authored-By:` trailer 都會被 M24 audit 抓到
- 完整回歸：Ruff passed、161 tests passed；目前仍無 remote／push／HF upload

### 換你做
仍是 M19：Colab 跑完後只需把兩個未改名結果 ZIP 放到
`results/colab/segmentation/`。HF `--confirm` 與所有 public 動作都先不要做。

---

## 2026-07-28 — Session 27：M22 自動選模與 raw checkpoint 綁定

### 做了什麼
- 新增 ADR-022，明定 M22「最佳模型」只是正式評估後的展示選擇，不得改稱 M16/M20
  主結果或改寫負面結論
- `demo_gradio.py --object {pcb1,capsules}` 現在要求完整 38-row M16 與 18-row M20
  CSV，自動選 seed-42 physical runs
- 固定排序：分類 Macro-F1 → AUROC → run name；分割 Dice → AUPRO → run name。
  `all_mixed` alias 不重複列為候選
- 入選 CSV row 必須逐項吻合 raw report 的 object、seed、canonical group、
  run signature、metrics 與 SafeTensors SHA256，才會配置 CUDA
- selection evidence 保存 CSV／report／model hashes 與
  `selection_is_post_evaluation_demo_only=true`，不保存本機絕對 checkpoint 路徑

### 真實 CPU-only preflight
- 目前本機 GPU 被另一 Python 工作占用約 23.6/24.6 GiB；未啟動或終止任何 CUDA 工作
- 真實 38-row M16 選模成功：
  - pcb1 → `m16_full_real_pcb1_seed_42`，Macro-F1 0.682569、AUROC 0.929415
  - capsules → `m16_full_real_capsules_seed_42`，Macro-F1 0.674849、AUROC 0.858299
- 真實 `--object pcb1` 在缺 `results/segmentation.csv` 時於 CUDA 前 fail closed，
  exit 1 且沒有寫出半份 `reports/demo_checkpoint_selection.json`
- Ruff passed、164 tests passed

### 換你做
仍只需要完成 M19 並放回兩個原始結果 ZIP；M20 通過後可直接用
`uv run python src/inference/demo_gradio.py --object pcb1 --inbrowser` 啟動 M22。

---

## 2026-07-28 — Session 28：GitHub Source ZIP 位元重現與 M18 最小 source bundle

### 做了什麼
- 從 staged tree 建立完全不含 `.git` 的獨立 archive，並在其中新建 Python 3.12
  `.venv`、依 `uv.lock` 安裝 171 packages，模擬日後 GitHub Source ZIP 使用者
- 發現 frozen JSON 在 Windows 工作樹與 Git blob 間被 CRLF→LF 轉換，造成
  `MANIFEST.sha256` 在公開 archive 失配；新增 ADR-023 並以 `.gitattributes`
  對凍結證據保存原始位元
- 發現 `package_m18_colab.py` 依賴 `git ls-files`，在 Source ZIP 無法執行；
  改為 deterministic allowlist，排除 reports、results、`.venv`、模型與秘密檔
- 新增無 `.git`、排除非白名單輸出及拒絕秘密命名的測試

### 全新 archive 驗證
- staged-tree `split_manifest.json` SHA256：
  `3d3c385cf0ff78479ecf90b4faf25fc07c88830e043616fb15aefb1282983e8c`
- Ruff passed、166 tests passed、`verify_splits.py` passed
- M18 package dry-run exit 0：
  - source：97 files / 2,330,613 uncompressed bytes
  - pcb1：5,206 files / 3,769,958,779 bytes / 2,000 pooled synthetic records
  - capsules：4,804 files / 3,953,298,637 bytes / 2,000 pooled synthetic records
  - 兩物件 training blocklist hits = 0，frozen test 僅供 evaluation
- 測試期間誤啟動的重複 DefectForge CPU dry-run 已精確終止；最後只保留一份受控
  session 並取得 exit 0。未終止或修改 FormosaNLU 程序，也未配置 GPU。

### 換你做
仍只需要完成 M19，將 Colab 下載的 `m18_seg_results_pcb1.zip` 與
`m18_seg_results_capsules.zip` 原檔放進 `results/colab/segmentation/`。

---

## 2026-07-28 — Session 29：M22 真模型 GIF 與 M24 evidence graph

### 做了什麼
- 重新跑 Phase 1 acceptance 時發現 verifier 永遠要求 Notebook 2 寫「尚未建立」；
  改為接受歷史 M18 ownership 或目前五項交接完整，仍拒絕 M15↔M18 dependency cycle
- Phase 1 重新驗證：M0–M15、10 份正式 JSON、3 支獨立 validator、62 commits、
  Contributors 唯一身分與零 co-author trailers 全綠
- 重跑 M16 38 raw runs、M13 filter、M14 quality、license chain；blocklist hits 皆 0
- 經 HF model_info API 重查 SD2、SDXL、DINOv2、SegFormer pinned revision、
  license、public／ungated 狀態，全數通過
- 新增 ADR-024 與強化 `verify_publish.py`：M0–M23、README、12 份 evidence、
  current hashes、24 小時授權新鮮度、HF safe dry-run、PNG/GIF、目視 review、
  acceptance report 任一缺口都 fail closed
- 新增 `record_demo_artifacts.py`：兩物件各 frozen test normal／anomaly，共四筆
  真模型 output 產生 1280×720 GIF，保存 input／mask／heatmap／selection hashes，
  不啟動 share URL
- 新增 `record_phase2_visual_review.py` 與 `build_release_acceptance.py`；前者需要明確
  review confirmation，後者目前負向測試正確拒寫並列出所有 M19–M23 缺口

### 驗證
- Ruff passed、176 tests passed
- 強化後 M24 checkpoint 正確回報：
  - 已通過：單一 Git 身分、零 trailer／secret／個人路徑／超大 tracked file、
    13 skills、HF dry-run、安全授權新鮮度
  - 尚缺：M19–M23、segmentation raw aggregation、四張正式 M21 圖、README final、
    demo GIF／validation／目視 review、一頁 acceptance
- `results/colab/segmentation/` 仍只有 `.gitkeep`；沒有生成、解壓或冒充 M19 產物
- 全程 CPU-only，未配置 GPU，未終止或修改 FormosaNLU

### 換你做
仍是把兩個未改名 M19 result ZIP 放回 `results/colab/segmentation/`。收到後可依目前
gate 清單一次完成 M20→M24，不需再補設計工具。

---

## 2026-07-27 — Session 20：M15 Phase 1 獨立驗收與交接邊界修正

### 做了什麼
- 發現原 M15 要求先填完 M18 SegFormer notebook 的五項交接，但 Phase 2 又必須等
  M15 全綠，形成無法執行的循環依賴
- 新增 ADR-019：M15 關閉已完成的 Phase 1；M18 負責分割 notebook、smoke 與五項
  具體交接，M19 才由使用者執行 Colab
- 新增 CPU-only `scripts/verify_phase1.py` 與單元測試，驗 M0–M15 勾選、凍結 evidence、
  M11 五項交接、三支獨立 validator 與每個 milestone 的 commit 覆蓋
- verifier 逐 commit 檢查 author／committer 僅
  `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`，並禁止
  `Co-Authored-By` trailer
- M11 的 L4 CU 在執行前未記錄，文件明列 unavailable，不用第三方費率倒推

### 下一步
**M16** — 先用 `df-guard` 重新驗證分類資料展開與 test blocklist，再實作固定
ConvNeXt-Tiny 協定、1-run smoke 與約 40 個正式 run。

### 換你做
目前沒有；M16 在本機 RTX 4090 執行。M18 準備好分割 notebook 前不需再開 Colab。

---

## 2026-07-27 — Session 18：M11 SDXL Colab 正式 fresh run 回收與 CPU 驗收

### 做了什麼
- 使用者在 Colab L4 跑完 `01_train_inpaint_lora_sdxl.ipynb`；pcb1 / capsules 各
  400 steps，wrapper 分別 922.2 / 855.8 秒
- 下載 147,164,017-byte、49-member 結果 ZIP，SHA256
  `9b4ff4f06cc2e0d3fdde84c257d0b709b8c2ac81487e0bc5ca8a89dec9da8660`
- 新增 `verify_colab_lora_results.py`，在不載入 SDXL、不配置 GPU 的情況下重驗 ZIP、
  frozen locks、adapter/tokenizer bundles、reports、sidecars、panel/background hashes
  與 803-entry test blocklist
- 逐張打開 8 張 3072×1024 sample panel，完整保留 raw artifact 的負面證據

### 正式結果
- pipeline 0.3.0、locked revision `115134f...e41e`、兩物件各 4 checkpoints
- pcb1：training 826.28 s、wall 910.91 s、0.4841 steps/s、peak 9.680 GiB
- capsules：training 830.61 s、wall 846.21 s、0.4816 steps/s、peak 9.680 GiB
- 兩物件 UNet + text encoder 1/2 adapters hashes 逐檔相符，fresh `PeftModel` reload
  由 Colab validator 通過；本機 CPU import validator `status=passed`、blocklist hits=0
- raw panel 非 seed copy，但 PCB 有文字／走線變形，capsules type0 有圓環、刻度、
  綠色球狀 artifact，type1 偏弱；不得宣稱可直接發佈

### 未完成與下一步
- M11 仍不勾：正式 run 只走 fresh branch，沒有實跑 checkpoint restore；PLAN 也明文
  要求本機 one-step smoke
- FormosaNLU 仍占用本機 GPU，因此現在只完成 CPU closeout；GPU 釋放後跑短 smoke +
  controlled resume，再決定是否勾 M11
- 不需再跑正式 Colab；Drive `runs/lora_sdxl/` checkpoints 暫時保留

### 換你做
目前沒有。不要再開 Colab，也不要中止 FormosaNLU；等 GPU 解除通知。

---

## 2026-07-27 — Session 19：M11 本機 real-model smoke／resume 完整閉環

### 做了什麼
- 收到 GPU 解除通知後先跑 `df-guard`：1,806 manifest images、1,805 pHash groups、
  0 crossing groups、803 unique blocklist hashes；兩物件 SDXL dry-run 均通過
- 經使用者明確同意後，把 13,875,747,454-byte locked SDXL cache 放到 D 槽；四份
  weight SHA256 全部符合 `configs/lora_sdxl.yaml`
- pcb1 / capsules 各跑一次真實 one-step smoke；另用兩個 Python process 做 pcb1
  step 1 controlled stop → `latest` resume → step 2
- 新增 `verify_local_sdxl_checks.py`，CPU-only 驗 model cache、smoke、run signature、
  checkpoint progression、三份 adapters、雙 tokenizer、samples 與 blocklist
- GPU 再度讓給 FormosaNLU 前，三個必要 GPU run 已全部正常結束；後續只做 CPU closeout

### 實測結果
- pcb1 smoke：training 4.27 s、wall 550.99 s、loss 0.097596、peak 9.619 GiB
- capsules smoke：training 4.51 s、wall 473.65 s、loss 0.187778、peak 9.619 GiB
- resume：第二個 process 明確 `Resuming pcb1 from step 1`，完成 step 2；cumulative
  training 9.77 s、wall 475.03 s、final loss 0.009820、peak 9.663 GiB
- 三個 final bundle fresh `PeftModel` reload 均通過；CPU verifier `status=passed`、
  blocklist hits=0
- one-step panels 幾乎 near-normal；resume step-2 PCB 只有局部藍／暗痕，這些只證明
  execution/resume，不取代 400-step formal visual review

### 結論
- M11 所有 formal、local smoke、actual resume、reload 與 VRAM 門檻已齊，PLAN 可勾完成
- 不需再跑 M11 Colab 或本機 GPU；large weights/runs 只留 D 槽 ignored paths

### 換你做
目前沒有。Drive 的 M11 checkpoints 與本機 cache 先保留。

---

## 2026-07-27 — Session 16：M14 mask-centered 生成品質評估與 sanity gate

### 做了什麼
- 新增 `src/evaluation/`、`scripts/evaluate_generation_quality.py`、
  `scripts/verify_generation_quality.py` 與 `configs/quality.yaml`
- 對 3,000 筆合成樣本、35 個 real components、35 個 deterministic noise controls
  建立 ratio 2.5 mask-centered crop；逐一雜湊 generated payload 與完整 provenance
- DINOv2 CLS 計算 nearest-real cosine 與 mutual-NN coverage；clean-fid 0.1.35
  clean-mode Inception features 計算未偏 KID與低秩精確 FID
- copy-paste / SD2 以 frozen `type0/type1` real group 對照；procedural 四類明列
  `real_scope=object_all`，不把兩套 taxonomy 假裝成等價
- 建立 `df-filter` 與 `df-eval` 兩個已驗證專案 skill

### 校準與決策
- 第一次 sanity run 的 NN、mNN、FID、noise control 全正常，但未偏 KID 用同一
  real set 對自身比較時為 `-0.0399` 到 `-0.1270`
- 沒有放寬門檻；改正 estimator 定義：formal rows 維持未偏 KID，identity/noise
  sanity 改用包含對角線的 biased polynomial MMD，同一集合對自身精確為 0
- clean-fid FID wrapper 因 SciPy 1.17 移除 `sqrtm(disp=...)` 無法使用；
  以 covariance factor nuclear norm 實作數學等價 FID，並與標準公式單元比對

### 正式結果與驗證
- source audit：6,909 paths / 6,909 hashes，test blocklist hits = 0；
  audit SHA256 `1056d0e5e19238af84fe523c54b6e5a4c7649764408cff8c21fa26e361d94b67`
- immutable crop cache：3,070 張（real 35 / generated 3,000 / noise 35）
- sanity 4/4：self NN min ≈ 1、self mNN = 1、biased KID = 0、FID ≈ 0；
  noise NN mean 0.0003–0.4240，均低於各 object `tau_low`；noise KID 0.4364–0.7402
- 正式表 44 列，唯一 empty 為已知的 filtered `pcb1/copy-paste/type1`
- 獨立 verifier 通過；report summary SHA256
  `906f768b65654e96d2e41c724b019d613d48760934f395017f0cf24f2a626a07`
- 兩張正式圖均已人工目視，標籤、分布、real/generated 對照與 EMPTY 格正確
- 提交前完整回歸：Ruff 全綠、pytest 66 passed、M13/M14 verifier 與兩個 skill
  官方 validator 全部通過

### 下一步
M11 SDXL notebook／Colab L4 執行與 M15 交接仍需使用者醒來後接續；
本機先完成可離線準備與紙上驗證。

### 換你做
目前沒有。GitHub repo、remote、push 仍依要求不建立、不操作。

---

## 2026-07-27 — Session 17：M11 SDXL 雙 encoder trainer 與 Colab 離線交接 foundation

### 做了什麼
- 新增 `configs/lora_sdxl.yaml`，鎖定 public SDXL inpainting revision 與
  text encoder 1/2、UNet、VAE 四個 LFS SHA256
- 延伸既有單一 trainer：SDXL 雙 tokenizer／雙 TrainableTokens adapter、
  concatenated penultimate states、pooled projection、time IDs、雙 adapter save/resume/reload
- 保留 SD2 pipeline 0.2.0 與既有 bundle schema；SDXL 獨立標 0.3.0
- 新增 11-cell 薄封裝 Colab notebook、靜態 validator 與最小 bundle builder
- 依 `df-finetune` SOP 與官方 Diffusers SDXL conditioning 文件更新介面與 ADR-018

### 驗證與產物
- pcb1 / capsules SDXL dry-run 皆通過：23 / 12 components、10 source images/object、
  frozen 三組 checksum、model revision 與四個 LFS hashes 全相符
- 第一次 dry-run 曾因手抄 VAE hash 少字而 fail closed；從 HF metadata 修正後才通過
- dual-conditioning 單元測試驗證 hidden concat、pooled embedding 與 1024 time IDs
- notebook validator：11 cells、5 sections、0 literal credentials、0 duplicated training loop
- 完整 CPU regression：67 tests passed；Ruff、既有 SD2 bundle validator 與 skill validator
  全部通過
- bundle：188 source files / 48 data files，test blocklist hits = 0；位置
  `D:\sdg-data\01-defectforge\colab\m11\`
  - archive bytes、file counts 與 SHA256 以同目錄 `m11_colab_bundle.json` 為唯一準據
  - 舊版 bundle 保留在 `archive-pre-final-20260727\`，可回復、未刪除

### 未完成與下一步
- SDXL base weights 超過無人值守 2 GB 下載界線，未下載、未冒充 real-model smoke
- 3_FormosaNLU 正式生成期間遵守跨專案 GPU 鎖；本輪只做 CPU／文件／測試，
  未啟動 CUDA、Ollama 或修改其程序
- M11 PLAN 保持未勾；等使用者同意下載後跑本機 1-step smoke、fresh/resume、
  fresh PEFT reload 並記錄 peak VRAM，再上 Colab L4 正式訓練

### 換你做
醒來後決定是否同意下載 pinned SDXL 權重並執行本機 smoke；GitHub repo/remote/push
仍未建立或操作。

---

## 2026-07-27 — Session 12：M10 SD2 LoRA 正式訓練、resume 與獨立重載

### 做了什麼
- 重新查證 SD2 模型來源；原 `stabilityai` repo 已不可用，依 ADR-014 改用 preservation
  mirror，鎖定 immutable revision 與 UNet／text encoder／VAE 三個 LFS SHA256
- 新增 `configs/lora_sd2.yaml` 與單一訓練實作 `src/training/train_inpaint_lora.py`：
  frozen component crop、9-channel inpainting loss、UNet attention LoRA、PEFT
  TrainableTokens、FP16、checkpoint、resume、held-out sample 與 Drive sync
- one-step smoke 先發現 Diffusers 0.39 的 PNDM final latent dtype promotion；修正後兩物件
  smoke 均完成訓練、sample、save 與 fresh reload，峰值 3.18 GiB
- 新增受控 `--stop-after-steps`，以 max=2 在 step 1 中斷，再從 `latest` 恢復 optimizer、
  scheduler、LoRA、token rows 與 micro-step，完成 step 2
- 正式訓練 pcb1 / capsules 各 400 steps；每 100 steps 存完整 checkpoint 與 20-step sample
- sample 依序輪替兩個 frozen trigger token，並寫 prompt、placement、crop、seed、model
  revision、背景與 panel SHA256 sidecar
- 新增 `scripts/validate_lora_run.py`，獨立重驗 checkpoint inventory、PEFT config、
  adapter/sample/background hash、blocklist、frozen checksum 與兩物件 `PeftModel` 重載
- 新增並驗證 `.claude/skills/df-finetune/SKILL.md`，把模型／資料 hash、GPU preflight、
  smoke、受控 resume、正式訓練、重載、目視退件條件與 Git 歸屬封成 fail-closed SOP
- 完整回歸為 Ruff 通過、24 個 pytest 通過，正式 validator 再次含 fresh reload 通過

### 正式結果
- pcb1：10 source images / 23 components / type 16+7；400 steps；訓練 128.66 秒；
  wall 134.82 秒；峰值 3.203 GiB
- capsules：10 source images / 12 components / type 9+3；400 steps；訓練 125.65 秒；
  wall 131.66 秒；峰值 3.203 GiB
- 兩者都遠低於 ADR-008 的 30 分鐘門檻，不啟動 Colab fallback
- validator 含 fresh PeftModel reload 最終 `status=passed`；
  詳見 `reports/lora_sd2_report.md` 與 `reports/lora_sd2_validation.json`

### 目視結論
- 所有 raw samples 都不是某張 few-shot seed 的逐張複製，M10 overfit 退回條件未觸發
- pcb1 type1 能產生局部板面／焊點變化；capsules type0 能產生斑點／色變
- pcb1 type0 受 placement 語意相容性影響，capsules type1 會出現文字／浮雕 artifact；
  binary mask raw patch 也可能有接縫
- 不把 raw patch 宣稱成最終品質；M12 必須做 guidance×crop refine 與全解析度 blend，
  M13 必須刷掉文字、語意錯位、接縫與 near-copy

### 下一步
**M11** — 準備 SDXL Colab L4 薄封裝 notebook 與本機 1-step smoke；實際 Colab 訓練等
使用者醒來操作。可平行先實作不依賴 SDXL 權重的 **M12 SD2 generation**。

### 換你做
目前沒有；M10 已全本機完成。GitHub repo／remote／push 仍等你醒來。

---

## 2026-07-27 — Session 13：M12 SD2 formal original／searched 生成與單調 refine

### 做了什麼
- 新增 `configs/generate_sd2.yaml`、`src/synthetic/generate_diffusion.py`：凍結
  model／adapter／M9 placement checksum，40-step crop inpainting、全解析度 feather blend、
  atomic sidecar、canonical metadata、fail-closed resume 與 contact sheet
- 新增 `scripts/validate_diffusion.py`：獨立重建 placement inventory，核對 source／adapter／
  config／pipeline hash chain、blocklist、mask identity、crop 外像素、候選排程與輸出 SHA
- 新增 `scripts/compare_diffusion.py`，以同 sample ID、union crop 固定視野產生
  clean／mask／original／searched 四欄圖，並在 type0/type1 間 round-robin 抽樣
- 兩次同 seed fresh smoke 得到 image/mask mismatch 0；完整 resume 得
  generated=0、skipped=2、peak VRAM=0，沒有載入模型
- 建立並通過 `.claude/skills/df-sdg/` 與 `.claude/skills/df-refine/`

### Formal original
- pcb1 / capsules 各 500 張，耗時 866.33 / 859.41 秒，峰值皆 3.051 GiB
- 合併 1,000 records；type quota 精確為 pcb1 348/152、capsules 375/125
- validator：1,000 unique image SHA、1,000 unique mask SHA、374 source files 重算、
  0 blocklist hit、M9 mask byte identity 與 full-resolution blend support 全通過
- 目視保留棋盤格、絲印扭曲、RGB 條紋、capsule halo／label-like highlight 等失敗樣本，
  不挑圖掩蓋

### Refine 排程修正
- 初版四候選雖覆蓋搜尋網格，卻未保證含 original pair；前 111 筆為
  97 改善 / 5 相同 / 9 退步
- 立即中止並新增 ADR-015；舊輸出移到兩個可復原隔離目錄，不混入 canonical metadata
- searched 升 v0.6.0：candidate 0 固定重現 original `(g=7.5, crop=2.5, seed index 0)`，
  另外三候選 deterministic greedy coverage；validator 逐筆要求 evidence 相同與 score 不退步
- v0.6 smoke 四筆 candidate 0 evidence 全相同，score delta 全非負，再從乾淨目錄正式重跑

### Formal searched 與目視
- pcb1 / capsules 各 500 張，4,000 candidate evaluations；耗時 3,228.21 / 3,200.48 秒，
  峰值皆 3.051 GiB
- 合併結果：778 改善 / 222 baseline / 0 退步；mean score 0.778089，
  對 original 平均 +0.087527
- validator：1,000 original baseline comparisons、1,000 unique image/mask SHA、
  0 blocklist hit、0 regression，全數通過
- PCB 的 checkerboard、黑色結構塊與 RGB stripes 多數變成低對比刮痕；capsule halo
  多數變成細痕／斑點
- 仍保留 near-invisible、silkscreen/text distortion、紫色高對比點、深色圓斑與
  stamp-like artifact；明列為 M13 filtering 的輸入，不宣稱 refine 已取代 final filter

### 驗證與資源
- Ruff 全綠；37 pytest 全過；兩個 skill quick validator 通過
- original 與 searched combined validators 最終皆 `status=passed`
- searched formal resume 兩物件皆 generated=0 / skipped=500 / peak VRAM=0，沒有載入模型
- formal bulk outputs 留在 D:；Git 僅納入 code/config/tests/reports/JSON/正式視覺證據
- 全部 Git author/committer 維持
  `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`；無 remote／push

### 下一步
**M11** — 完成 SDXL 單一訓練實作的 Colab L4 薄封裝與 smoke；實際 Colab 訓練等使用者。
本機 critical path 可接著做 **M13** 六道 filtering 與漏斗報表。

### 換你做
目前仍沒有本機操作需求。GitHub repo、remote、push 繼續等使用者醒來；SDXL 正式訓練
才需要使用者在 Colab 操作。

---

## 2026-07-27 — Session 14：M13 filtering 純規則與校準 foundation

### 做了什麼
- M12 乾淨提交後沒有停在里程碑邊界，直接稽核三個 formal input、legal ROI 重建路徑、
  frozen M9 DINO score cache、35 個真實 component embedding 與 real mask 統計
- 新增 `configs/filters.yaml`，鎖六道規則初值、DINO revision、輸出與 contact-sheet 契約
- 新增 `src/filtering/metrics.py`：ROI containment、log-reference area/aspect z-score、
  clipped square context、局部 pHash、外環帶 gradient-histogram normalized
  Wasserstein seam score
- 新增 `src/filtering/rules.py`：文件指定的八個拒絕 enum、固定 funnel 順序、
  per-rule disable 與 previously-accepted pHash 最近距離
- seam 測試發現 mask 內平滑改色也會因 Sobel 支撐域在外環帶留下小梯度；保留正確演算法，
  把測試從錯誤的 `score == 1` 改為「高分且顯著高於人工硬接縫」

### 驗證
- filtering foundation Ruff 全綠、9 tests 全過
- 無 GPU／filesystem side effect；尚未假裝 M13 runner 或全量校準已完成
- M13 PLAN 維持未勾選，直到 DINO、atomic filtered/unfiltered、漏斗 verifier 與目視全綠

### 下一步
接上 frozen real-reference DINO calibration、`run_filters.py` 與 `verify_filter_report.py`，
先 fresh smoke 再跑 Stage A / Stage B 全量。

### 換你做
目前沒有；這一層不需要使用者操作。

---

## 2026-07-27 — Session 15：M13 全量六道 filtering、atomic views 與 verifier

### 做了什麼
- 實作 `scripts/filter_synthetic.py` 與 `src/filtering/{dataset,embeddings,pipeline}.py`
- DINOv2 固定 `facebook/dinov2-base@f9e44c8...`，用 20 個 real few-shot
  component context crops 校準每個 object 的 leave-one-out `tau_low` 與
  centroid-distance `tau_outlier`
- 三個 formal input 共 3,000 筆完整跑 ROI、area、aspect、pHash、DINOv2、seam；
  generated embedding 以內容鍵控 NPZ 快取，背景 RGB LRU 限制 32 張
- 第一次 full run 發現遮罩列表會常駐約 4.5 GB，立即停止未發布程序，改為每 128 張
  重讀、用完釋放；修正版才從頭正式執行
- 發布 `${synthetic}/unfiltered` 3,000 筆與 `${synthetic}/filtered` 1,770 筆；
  image/mask 都是來源 hardlink，metadata 採暫存檔 fsync 後原子替換
- 新增 exact embedded-summary verifier、漏斗報表與 deterministic 12 accepted /
  12 rejected contact sheets；額外抽查 `pcb1/copy-paste/type1` 的 12 張 rejected

### 正式結果
- accepted 1,770 / 3,000；rejected 1,230
- funnel survivors：ROI 3,000 → area 2,635 → aspect 2,579 → pHash 2,542 →
  DINOv2 1,770 → seam 1,770
- reason counts：AREA 365、ASPECT 103、pHash 37、NN_TOO_LOW 979、
  EMBEDDING_OUTLIER 897；ROI / NN copy / seam 為 0
- thresholds：
  - capsules `tau_low=0.6304827929`、`tau_outlier=0.4422172606`
  - pcb1 `tau_low=0.7391234636`、`tau_outlier=0.3955494761`
- `pcb1/copy-paste/type1` 為 0/152 accepted；專項抽查顯示全部 area
  (`median=1.085e-4`) 低於 real p05 (`1.267e-3`)，且 151/152 同時 NN 過低，
  因此不為了非零覆蓋率而放寬門檻

### 驗證
- full verifier：3,000 records、1,770 accepted、9,540 published payload hardlinks 全通過
- report summary SHA256：
  `831ec0936a7e3068c30f46f5fac1240634ae0de23aa8dc3e8ad9db87dd88a8a0`
- accepted/rejected contact sheets 均已人工目視：輪廓對位、標籤可讀、無空白格
- M13 tests 17 passed；全專案測試與 Ruff 另於提交前重跑

### 下一步
**M14** — 生成品質評估（mNN / KID 與 per-generator quality report）；
**M11** SDXL Colab 仍需要使用者醒來後提供執行環境。

### 換你做
目前沒有。GitHub repo／remote／push 仍照要求等待使用者醒來。

---

## 2026-07-27 — Session 10：M8 程序化 Stage A 與 no-real-stats 對照

### 做了什麼
- 新增 `src/synthetic/procedural.py`，用 NumPy/OpenCV 實作 perlin、crack、scratch、
  spot 四類決定性 mask 與羽化色彩／明暗效果，不讀任何真實瑕疵像素
- 正式 real-stats 版只讀 10 張 few-shot training masks 的 aggregate p05/p95
  area-ratio / aspect-ratio；先驗證 manifest 與 selection checksum 鏈
- 依 ADR-011 實作 `--no-real-stats`：只用 `configs/stage_a.yaml` 的手訂圓整範圍，
  並安裝 Python audit hook，任何開啟 `real_mask_stats.json` 的嘗試都會致命失敗
- 兩版皆各生成 pcb1 500 + capsules 500；每物件四類形狀精確各 125
- 新增 `scripts/validate_procedural.py`，獨立重建 ROI、核對 frozen train-good
  provenance、零真實瑕疵來源欄位、mask geometry、統計範圍、blocklist 與輸出 hash
- 依 roadmap 建立 `.claude/skills/df-stage-a/`，skill validator 通過

### 正式輸出與驗證
- real-stats：1,000 images + 1,000 masks + 1,000 metadata；610 個背景 ROI 重建，
  2,000 unique output SHA256，area/aspect outlier rate 全部 0%，blocklist hits = 0
- no-real-stats：同樣 1,000 + 1,000 + 1,000；612 個背景 ROI 重建，
  2,000 unique output SHA256，outlier rate 全部 0%，blocklist hits = 0；
  生成、resume 與獨立驗證全程 audit guard 未觸發
- 兩個獨立 no-real-stats `n=4` 行程比較 16 個 image/mask PNG：0 mismatch，
  byte-for-byte 可重現
- 兩版都用 `--resume` 重驗 1,000 筆，沒有重寫成功樣本
- 實際開啟四張正式 24 格 grid：四類形狀皆出現；PCB mask 全在板體，
  capsule mask 全在綠色膠囊；未見桌布／陰影誤貼或矩形接縫
- pytest 12 passed；完整 Ruff、skill validator 與 contributors audit 於提交前執行

### 下一步
**M9** — 依真實 mask 元件數比例產生 SDG-ready legal ROI placement 清單，
驗證不重疊、面積範圍與 24 格視覺化。

### 換你做
目前沒有；GitHub repo／remote／push 仍等你醒來，本機可直接繼續 M9。

---

## 2026-07-27 — Session 11：M9 SDG testcase placement 全量生成與獨立驗證

### 做了什麼
- 新增 `src/synthetic/mask_placement.py` 與 `configs/placement.yaml`：鎖定
  `facebook/dinov2-base` revision
  `f9e44c814b77203eaa57a6bdbbd535f21ede1415`，以 patch-token 局部／全域 cosine
  heterogeneity 建結構 ROI，再與 object-specific Otsu／saturation 前景取 intersection
- 從 M6 frozen components 按元件數比例精確排 type 配額；旋轉、縮放、翻轉後只接受
  面積比與長寬比落在 training-mask p05–p95 的候選，使用 sample-local PCG64 決定性放置
- 每個 train-good 背景產生 3 個 sibling placement；後一個 placement 會避開前面的 mask
  與 5px clearance，metadata 記錄完整背景／component／affine／ROI／geometry／seed provenance
- DINOv2 score cache key 納入 frozen manifest、model id/revision/config、每張背景路徑與
  SHA256；支援 fail-closed resume，已用 48 筆 smoke output 實際驗過
- 新增 `scripts/validate_placements.py`，獨立核對 exact inventory/type schedule、所有來源
  checksum、test blocklist、二值 PNG、重建 ROI、mask geometry、real-stat bounds 與 sibling overlap
- 新增 4 個 placement 單元測試與 `.claude/skills/df-prep-testcase/`；skill quick validator 通過

### 正式輸出與驗證
- pcb1：602 個背景、1,806 個 placements；type0 1,256 / type1 550
- capsules：361 個背景、1,083 個 placements；type0 812 / type1 271
- 合計：963 個背景、2,889 個 placements、2,889 個 unique mask SHA256
- 獨立 validator 重建 963 個 legal ROI、hash 963 個背景與 40 個實際使用的來源檔；
  area/aspect outliers = 0、sibling overlaps = 0、test blocklist hits = 0
- frozen manifest SHA256 仍為
  `3d3c385cf0ff78479ecf90b4faf25fc07c88830e043616fb15aefb1282983e8c`；
  defect-types SHA256 仍為
  `0c7669287fb1b8f48b3f6aff202eaefc914e8bfc1d892b9e63f88b20996cb41a`
- 實際開啟兩張正式 24 格圖：PCB red masks 全在板體、capsules red masks 全在膠囊；
  未見灰色背景落點，所有 red mask 都位於 cyan intersection ROI
- 兩個獨立 `limit-backgrounds=4, n-per-image=1` 行程比較 8 個 PNG：0 mismatch，
  byte-for-byte 可重現

### 下一步
**M10** — 建立 SD2 inpainting LoRA 本機訓練資料封裝、訓練／resume／載回驗證與樣本圖；
先做 one-step smoke 和 VRAM／耗時量測，再決定正式 steps。

### 換你做
目前沒有；GitHub repo／remote／push 仍等你醒來。本機會直接繼續 M10。

---

## 2026-07-27 — Session 09：M7 copy-paste Stage A 完整生成與獨立驗證

### 做了什麼
- 新增 `src/synthetic/copy_paste.py` 與共用 `src/common/imaging.py`：從 M6 frozen
  components 決定性抽樣，做旋轉／縮放／翻轉／色彩擾動，再以 Poisson 或 feather-alpha
  混合到 frozen train-good 背景
- PCB 使用最大飽和前景的內縮 bbox；膠囊使用清理後的飽和區域，放置以
  `cv2.matchTemplate` 精確找出能讓非零 mask 100% 位於 legal ROI 的座標
- 以 largest-remainder 配額生成 pcb1 500 張（type0 348 / type1 152）與 capsules
  500 張（type0 375 / type1 125），輸出到
  `D:/sdg-data/01-defectforge/synthetic/stageA_copypaste`
- 新增 exact-field metadata schema 與每張來源、仿射、blend、mask geometry、seed provenance
- 新增 `scripts/validate_synthetic.py`，獨立重新開啟全輸出、重建 ROI、核對 frozen
  background/component、blocklist、tree inventory 與 metadata

### 驗證結果
- 正式輸出：1,000 images + 1,000 binary masks + 1,000 metadata records
- 獨立驗證重建 594 個 background ROI、雜湊 634 個來源檔與全部輸出；
  2,000 個輸出 SHA256 全部唯一，test blocklist hits = 0
- Blend：Poisson 493、feather-alpha 507；每種物件／型別皆達到精確配額
- 兩個獨立 `n=2` 同 seed 行程比較全部 8 個 image/mask PNG：0 mismatch，
  byte-for-byte 可重現
- `--resume` 對正式 1,000 筆重驗成功，沒有重寫既有成功樣本
- 實際開啟兩張 24 格正式 grid：PCB mask 全在板體，膠囊 mask 全在綠色膠囊；
  兩者均未見矩形接縫或貼到桌布／背景的瑕疵
- pytest 9 passed；完整 Ruff 與最終 contributors audit 於提交前執行

### 下一步
**M8** — 程序化生成每物件 500 張，另做完全不讀 real statistics 的
`--no-real-stats` 對照；逐張驗證並目視正式 grids。

### 換你做
目前沒有；GitHub repo 尚未建立不影響本機推進，remote／push 仍留待你醒來。

---

## 2026-07-28 — Session 31：M22 真模型 demo、UI 實測與 M23 README

### 做了什麼
- 經使用者確認 RTX 4090 當下為 10.7 / 24.0 GB 後，在不終止、不修改
  FormosaNLU 的前提下完成 M22；DefectForge 真模型推論沒有 OOM
- `scripts/record_demo_artifacts.py` 從 frozen highshot test 決定性選四張圖，
  重新核對 classification / segmentation CSV、formal raw reports 與 checkpoint hashes，
  原子產生 `assets/demo.gif`、`reports/demo_checkpoint_selection.json` 與
  `reports/demo_validation.json`
- 以 headless Playwright 實際啟動本機 Gradio、上傳
  `pcb1` frozen test anomaly `000.JPG`、按下 Run inspection，確認分類機率、mask、
  heatmap 與 latency 都有渲染；沒有開 `share=True`
- UI 實測抓到 confidence / latency 與標題對比不足；改用此版 Gradio 可穩定套用的
  component classes 修正，重啟後再次上傳與截圖，畫面完整可讀
- README 加入正式 GIF、真實負面結果與 demo 限制；`verify_readme.py --write`
  重新生成全部驗證表格並通過
- 打開七張 final PNG 原尺寸檢查，另逐幀檢查 GIF 四幀，寫入 hash-bound
  `reports/phase2_visual_review.json`

### 正式證據
- GIF：4 frames、1280×720、SHA256
  `1b77f4ebcef820d2634d10563374d24b8d214186682981b2faf8314b4c3b3cef`
- post-evaluation checkpoint selection：
  - pcb1 classifier `m16_full_real_pcb1_seed_42`
  - pcb1 segmenter `m18_full_real_pcb1_seed42`
  - capsules classifier `m16_full_real_capsules_seed_42`
  - capsules segmenter `m18_full_real_capsules_seed42`
- Gradio frozen test upload：Defect 73% / Normal 27%；最終 headless run latency
  1240.2 ms，7 個 image DOM elements，share URL = false
- README SHA256：
  `bea4430ff05fb97bec0a0a2558d8da329ec3475ed59126e5adfabf5b9811a0cd`
- `demo_validation.json`、`readme_validation.json`、
  `phase2_visual_review.json` 皆為 `status=passed`

### 誠實限制
- 四張固定 demo 圖在 preregistered threshold 0.5 的 binary mask coverage 均為 0%
- GIF 仍保留原始 classifier probability、空 mask、probability heatmap 與 latency；
  沒有為了展示效果重選樣本或改 threshold
- live UI anomaly 也得到 0% mask；這與 formal segmentation 指標的負面／不穩定結果一致，
  已明列於 README Limitations

### 下一步
執行 M24 本機 release acceptance、完整測試、secret / contributor audit，整理成不公開的
release candidate；建立 GitHub repo、push 與轉 public 必須等使用者最後過目後才做。

### 換你做
目前不用操作 GPU 或 Colab。待本機 M24 gate 全綠後，只需人工看 README / GIF，
再決定是否建立並公開 GitHub repository。

---

## 2026-07-27 — Session 07：M5 決定性 few-shot／validation 與 mask 統計

### 做了什麼
- 先執行 `df-guard`：frozen manifest checksum 與 blocklist 連結相符，
  40 張 few-shot 候選 image + 40 張 mask 逐檔 SHA256 均未命中 test
- 新增 `scripts/sample_fewshot.py`，以 `random.Random(42)` 從每物件官方 20 張
  few-shot anomaly pool 決定性抽 k=10
- 依 ADR-013 從 fewshot pool 之外抽固定 10% development validation：
  pcb1 60 good / 6 bad；capsules 36 good / 6 bad
- 產生 `splits/fewshot_selection.json` 與 checksum sidecar；未修改 frozen manifest
- 產生 `reports/real_mask_stats.json`（raw values、bbox、面積比、長寬比、位置與百分位）
  與 `reports/fewshot_stats.md`
- 產生並實際開啟兩張 2×5 contact sheet；建立已驗證的 `.claude/skills/df-split/`

### 凍結結果
- Selection SHA256：`7021234d0bef51926832591d60c205fa7273e0cc32fd0ae5348740094b060ea2`
- Manifest SHA256 前後皆為
  `3d3c385cf0ff78479ecf90b4faf25fc07c88830e043616fb15aefb1282983e8c`
- 每物件 few-shot seed = 10，與 validation 完全互斥

### Mask 統計摘要
- pcb1 area ratio：median 0.00262168；p05–p95 0.00126741–0.0370321
- capsules area ratio：median 0.00162967；p05–p95 0.000101167–0.0157801
- pcb1 aspect ratio：median 1.25768；p05–p95 0.917172–3.61198
- capsules aspect ratio：median 1.03947；p05–p95 0.66352–1.14792

### 驗證與目視
- 兩次獨立行程重跑，selection 檔案 SHA256 byte-for-byte 相同
- 所有 seed / validation image 與 mask hash 均不在 test blocklist
- 兩張 contact sheet 各 10 格都有紅色 GT 輪廓；mask 非空、與肉眼可辨瑕疵位置一致
- pcb1 涵蓋接腳、板面／焊點與元件區；capsules 涵蓋小斑點、破損及較大區域缺陷
- `ruff check` 全綠；pytest 5 passed；`df-split` validator 通過

### 下一步
**M6** — 只使用這 20 張 seed 的 GT 元件，抽 DINOv2 + morphology 特徵、
以 silhouette / min-cluster-size 選群，輸出暫用 token 與 contact sheets。

### 換你做
目前沒有；M6 暫用型別 token 會依 ADR-012 自動命名，不等人工命名。

---

## 2026-07-27 — Session 08：M6 DINOv2 + morphology 瑕疵分型凍結

### 做了什麼
- 再次執行 `df-guard`，20 張 seed image + mask 全數未命中 test blocklist
- 查證目前官方 Hugging Face 用法，以 `AutoImageProcessor + AutoModel` 載入
  `facebook/dinov2-base`，取 `last_hidden_state[:, 0, :]` 的 768-D CLS embedding
- 新增 `scripts/cluster_defect_types.py`：連通元件、32px 雜點過濾、DINOv2 crop embedding、
  八項 morphology、block-standardization/balancing、Ward agglomerative 與 silhouette 選 k
- 模型鎖定 revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`（Apache-2.0）
- 凍結 `splits/defect_types.json` + `splits/DEFECT_TYPES.sha256`，
  產出兩張 clustering contact sheet 與 `reports/defect_type_report.md`
- 建立並通過 validator 的 `.claude/skills/df-types/`

### 分群結果
- pcb1：23 components（另濾 3 個 <32px 雜點），k=2，群大小 16 / 7；
  k=3–5 雖 silhouette 稍高但都有 singleton，依硬性 min-size=3 排除
- capsules：12 components，k=2，群大小 9 / 3；k=3–5 均有 1–2 張小群而排除
- Defect types SHA256：
  `0c7669287fb1b8f48b3f6aff202eaefc914e8bfc1d892b9e63f88b20996cb41a`
- 暫用 token：`<pcb1-type0>`、`<pcb1-type1>`、`<capsules-type0>`、
  `<capsules-type1>`；`confirmed_by_user=false`

### 目視與驗證
- pcb1 type0 以接腳／大型結構區為主，type1 以小型板面局部區為主
- capsules type0 以局部圓斑／凹點為主，type1 是較大破損或整顆受損
- 每個紅色輪廓皆對準非空 GT component，沒有空背景 crop；微小瑕疵 crop 模糊但合理
- 每群 ≥3；所有來源都來自 frozen seed 且不在 blocklist
- frozen manifest / selection checksum 前後不變
- `ruff check` 全綠；pytest 7 passed；`df-types` validator 通過

### 未決
- 暫用 display names 可由使用者日後改成語意名稱；trigger token 不得變，且不阻塞後續
- Hugging Face cache 的 Windows symlink warning 不影響結果，已記 troubleshooting

### 下一步
**M7** — 實作並生成每物件 500 張 copy-paste 合成（mixed blend），先做 ROI/placement
自動斷言，再抽 24 張 grid 目視。

### 換你做
目前沒有；顯示名稱可以等你醒來再改，M7 不需要等待。

---

## 2026-07-27 — Session 06：M4 pHash 分群、manifest 凍結與防洩漏 guard

### 做了什麼
- 新增 `scripts/freeze_manifest.py`：讀 high-shot 基底、計算 image/mask SHA256、
  `imagehash.phash(hash_size=16)`、union-find 傳遞閉包與 test blocklist
- 先用 `--dry-run` 對 1,806 張影像校準，確認閾值後才第一次正式寫入，沒有使用 `--force`
- 凍結 `splits/split_manifest.json`、`splits/MANIFEST.sha256`、
  `splits/test_blocklist.json`，並輸出 `reports/split_report.md`
- 新增 `src/common/integrity.py`，提供 manifest checksum 與 blocklist 的共用 fail-closed guard
- 依已驗證流程建立 `.claude/skills/df-guard/`，官方 skill validator 通過
- 新增 ADR-013，解決「M4 manifest 不可變」與「M5 回寫抽樣欄位」的規格衝突

### pHash 校準
- pcb1：1,104 images / 1,103 groups；最近鄰距離 min 6、median 20；只有 1 對 ≤6
- capsules：702 images / 702 groups；最近鄰距離 min 34、median 92；0 對 ≤6
- 閾值保留 6；跨 train/test group = 0，因此沒有影像需要移到 test

### 凍結結果
- manifest SHA256：`3d3c385cf0ff78479ecf90b4faf25fc07c88830e043616fb15aefb1282983e8c`
- manifest：1,806 images / 1,805 pHash groups
- 最終 train：pcb1 602 good / 60 bad；capsules 361 good / 60 bad
- 最終 test：pcb1 402 good / 40 bad；capsules 241 good / 40 bad
- blocklist：723 test images + 80 test bad masks = 803 unique SHA256

### 驗證
- manifest checksum 由獨立 loader 重算相符
- 每個 `group_id` 只屬於一個 final set
- 每張 test image 與 bad mask 的 SHA256 都精確包含在 blocklist，集合完全相等
- blocklist 的 manifest SHA256 反向連結相符
- `ruff check` 全綠；pytest 4 passed；`df-guard` validator 通過

### 決策
- Manifest 只保存不可變的來源與 partition 事實；M5 抽樣寫 sidecar，不破壞凍結 checksum
- Validation 為 train pool 每 object × label 的固定 10% 開發 holdout，排除官方 fewshot pool；
  正式比較用凍結超參 refit 完整 pool，保留 10 / 20 / 60 口徑

### 下一步
**M5** — 決定性抽 k=10 與 validation sidecar、mask 統計、疊輪廓 contact sheets，
並確認 manifest SHA256 前後完全不變。

### 換你做
目前沒有；GitHub repo、remote、push 仍等你醒來。
