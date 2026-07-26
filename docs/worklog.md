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
| Colab L4 費率 | 只查到 T4 約 1.76–1.96 CU/hr、A100 約 10–15 CU/hr，**L4 未查到確切值** | M15 到 Colab 頁面實測 |
| 套件版本 | 2026-07-27 查到 torch 2.13.0 / diffusers 0.39.0 / peft 0.19.1，但 pytorch.org 的頁面回傳疑似快取的舊值 | M1 重新查證後 `uv lock` |
| 分型可用性 | 10 張 seed 分群後每型可能只剩 2–4 個元件，trigger token 可能學不起來 | M6 看實際分群結果決定是否啟動 fallback |
| SD2 vs SDXL 額度 | 兩個底模都做會吃掉較多 Colab units | M15 估算後回報，必要時把 SDXL 排到下個月 |

### 下一步
**M1** — 建立 uv 虛擬環境並鎖版（Python 3.12、torch cu128）。
開工前先重新查證各套件當時最新版本。

### 換你做
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
author/committer 都是 `kuotunyu <[redacted-school-email]>`，且無 `Co-Authored-By` trailer。

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
