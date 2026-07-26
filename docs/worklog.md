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
| 🔴 **commit 信箱洩漏學校信箱** | 前 3 筆 commit（`9a3038c`／`bf2c89f`／`1867b7c`）的 author 與 committer 都是 `[redacted-school-email]`。**全域 `git config user.email` 當時也是這個值**，所以任何沒有覆寫的 repo 都會中招。`publish-repo` 第 1 關要求只能出現 `61350295+kuotunyu@...`。repo-local 身分已於 2026-07-27 修正，之後的 commit 乾淨；**但既有 3 筆需要 `git filter-repo --email-callback` 改寫歷史，這屬於使用者親自執行的動作** | 發佈前必須解決 |
| 分型可用性 | 10 張 seed 分群後每型可能只剩 2–4 個元件，trigger token 可能學不起來 | M6 看實際分群結果決定是否啟動 fallback |
| SD2 vs SDXL 額度 | 兩個底模都做會吃掉較多 Colab units | M15 估算後回報，必要時把 SDXL 排到下個月 |

### 下一步
**M1** — 建立 uv 虛擬環境並鎖版（Python 3.12、torch **2.13.0+cu130**）。
開工前先重新查證各套件當時最新版本，並確認 `diffusers` 0.39 與 `transformers` v5 能否共存。

### 換你做
0. 🔴 **改寫前 3 筆 commit 的作者信箱**（它們寫進了學校信箱，發佈就會公開且永久）。
   repo-local 設定已修好，之後的 commit 乾淨；既有的要靠改寫歷史：
   ```bash
   git filter-repo --email-callback 'return b"61350295+kuotunyu@users.noreply.github.com" if b"<school-domain>" in email else email' --force
   ```
   跑完用 `git log --all --format='%ae' | sort -u` 確認只剩 noreply 那一個。
   **順便考慮把全域設定也改掉**，否則下一個新 repo 又會中招：
   ```bash
   git config --global user.email "61350295+kuotunyu@users.noreply.github.com"
   ```
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
