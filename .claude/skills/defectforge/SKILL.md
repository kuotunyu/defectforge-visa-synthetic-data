---
name: defectforge
description: 恢復 defectforge-visa-synthetic-data 專案的脈絡並決定下一步。當使用者在這個專案裡說「我們做到哪了」「繼續」「接著做」「現在該做什麼」，或隔一段時間回來重啟工作、或不確定該用哪個 df-* 階段 skill 時使用。也用於每個里程碑收尾（跑驗證欄、更新里程碑清單、追加 worklog、git commit、產出「換你做」清單）。
---

# DefectForge Orchestrator

這個專案的入口與階段路由器，對應 NVIDIA GTC 2026 課程 agentic flow 的 `anomalygen`
orchestrator skill。**它不直接做管線工作**，只負責兩件事：

1. **恢復脈絡** — 讓任何一個新 session 在三分鐘內知道現在在哪
2. **收尾** — 每個里程碑結束時，確保驗證、里程碑清單、worklog、git 四件事都做到

> **給閱讀公開 repository 的人**：本專案把 13 個 agent skill 中的
> `defectforge`（本檔）與 `df-guard`（防洩漏護欄）公開，其餘 11 個階段 skill
> 與 owner-only 的工作記憶（`CLAUDE.md`、`PLAN.md`、`docs/worklog.md` 等）
> 保留在本機、不進 Git。下文提到這些檔名時一律以 `程式碼樣式` 呈現而非連結，
> 因為它們在公開 repository 中不存在。設計理由見
> [`docs/decisions.md`](../../../docs/decisions.md) 的 ADR-028。

---

## A. 恢復脈絡（開工時）

依序讀這四份，**不要跳過**：

| 順序 | 檔案 | 讀出什麼 | 公開 repo |
|---|---|---|---|
| 1 | `CLAUDE.md` | 分工／實驗鐵律／工作方式／Windows-native 規則 | 否 |
| 2 | `PLAN.md` | 里程碑的勾選狀態 → **第一個沒勾的就是當前里程碑** | 否 |
| 3 | `docs/worklog.md` | **最後一筆**：上次做了什麼、未決風險、下一步、換你做 | 否 |
| 4 | [`docs/decisions.md`](../../../docs/decisions.md) | 已定案的 ADR，避免重新爭論已經決定的事 | 是 |

`PLAN.md` 末尾另有一份「已知未完成項目」清單，記錄**已勾選但科學上尚未收尾**的缺口。
恢復脈絡時要一併讀，否則會誤以為全部完成。

然後檢查磁碟上的實際狀態，**不要只信文件**：

```powershell
Get-ChildItem .\splits
uv run --frozen python -c "from src.common.paths import load_paths; from pathlib import Path; p=load_paths(); print('VisA:', Path(p.visa_raw).exists())"
git log --oneline -5
git status --short
```

第二行刻意從 [`configs/paths.yaml`](../../../configs/paths.yaml) 解析路徑，
**不得硬編 `C:` 或 `D:` 絕對路徑**（ADR-005）。

**文件說已完成、但磁碟上東西不在 → 以磁碟為準，並回報這個落差。**

### 回報格式（給使用者，繁體中文，簡短）

```
目前在 M<n>：<里程碑標題>
上次做到：<worklog 最後一筆的「下一步」>
磁碟狀態：<split 是否已凍結 / VisA 是否已下載 / 有無未 commit 變更>
未決風險：<worklog 最後一筆與「已知未完成項目」中還沒解掉的>
建議下一步：<一句話>
```

---

## B. 階段路由

| 里程碑 | Skill | 規格文件 |
|---|---|---|
| M1–M2 環境與資料 | `df-setup` | [environment.md](../../../docs/environment.md)、[data_protocol.md](../../../docs/data_protocol.md) §1–2 |
| M3–M5 split 凍結 | `df-split` | [data_protocol.md](../../../docs/data_protocol.md) §3–5 |
| M6 瑕疵分型 | `df-types` | [data_protocol.md](../../../docs/data_protocol.md) §5.3、ADR-002 |
| **任何讀圖動作之前** | **`df-guard`** | [data_protocol.md](../../../docs/data_protocol.md) §4.3、§6 |
| M7–M8 Stage A 合成 | `df-stage-a` | [synthesis_spec.md](../../../docs/synthesis_spec.md) §3 |
| M9 auto mask placement | `df-prep-testcase` | [synthesis_spec.md](../../../docs/synthesis_spec.md) §4.1、§2 |
| M10–M11 LoRA 微調 | `df-finetune` | [synthesis_spec.md](../../../docs/synthesis_spec.md) §4.2 |
| M12 生成與 refine | `df-sdg` / `df-refine` | [synthesis_spec.md](../../../docs/synthesis_spec.md) §4.3–4.4 |
| M13 品質過濾 | `df-filter` | [filtering_spec.md](../../../docs/filtering_spec.md) §1 |
| M14 生成品質指標 | `df-eval` | [filtering_spec.md](../../../docs/filtering_spec.md) §2 |
| M16/M18/M20 下游實驗 | `df-downstream` | [experiment_protocol.md](../../../docs/experiment_protocol.md) |

發佈相關的 `df-release` **從未建立**：M24 的發佈稽核最後是由
[`scripts/verify_publish.py`](../../../scripts/verify_publish.py) 這支 fail-closed
腳本承擔，不需要再包一層 skill。要跑發佈檢查就直接執行該腳本。

---

## C. 每次動手前的紅線檢查

`df-guard` skill 會執行完整版；以下是它的摘要，任何階段都適用：

- [ ] **Test 不可觸碰**：要讀的影像路徑先比對 `splits/test_blocklist.json` 的 SHA256
- [ ] **Split 已凍結**：`splits/split_manifest.json` 的 SHA256 必須與
      `splits/MANIFEST.sha256` 相符；不符表示有人動過，**停下來回報，不要自行修復**
- [ ] **路徑不硬編**：所有路徑從 `configs/paths.yaml` 讀
- [ ] **CLI 照契約**：參數一律照 [interfaces.md](../../../docs/interfaces.md)，不自行發明
- [ ] **金鑰不進 git**：不把 `.env` 內容寫進任何檔案或印出來
- [ ] **>2GB 下載、花錢、`git push`、Hugging Face 上傳 → 先問使用者**

---

## D. 里程碑收尾（做完一個 M 就跑一次）

1. **驗證**：跑該里程碑「驗證」欄的每一項，逐項回報通過與否。
   **有任何一項沒過就不准勾選。**
2. **自己看圖**：若有產出圖表或樣本圖，用 Read 工具打開檢視；不合理就修，
   不要只確認檔案存在就當完成。
3. **勾選** `PLAN.md` 對應項目（`[ ]` → `[x]`），並附「完成證據」。
4. **追加 `docs/worklog.md` 一筆**，用該檔開頭規定的六段格式。
5. **重大選型追加 ADR** 到 [`docs/decisions.md`](../../../docs/decisions.md)（**只追加不改寫**）。
6. **踩到的坑追加**到 [`docs/troubleshooting.md`](../../../docs/troubleshooting.md)。
7. **建立該階段的 skill**（若尚未建立），內容必須是**已驗證可行**的 SOP，不是想像中的步驟。
8. **git commit**，訊息用 `<type>(M<n>): <英文簡述>`。
   **不得加 `Co-Authored-By:` trailer 或任何產生器署名**——本 repository 的
   Contributors 必須只有 `kuotunyu`。
9. **給使用者「換你做」清單**（繁體中文，具體到可以照做）。

### 改動已發佈結果時的額外要求

README 的結果表與結論是**產生的**，不是手寫的。改動流程有 hash 相依，順序固定：

```powershell
uv run --frozen python scripts/verify_model_licenses.py
uv run --frozen python scripts/verify_readme.py --write
uv run --frozen python scripts/verify_license_chain.py
uv run --frozen python scripts/verify_publish.py
```

順序錯了會把上游舊 hash 寫進下游報告，`final_evidence_hashes_current` 會一直不過。

---

## E. 常見情境

| 使用者說 | 你要做 |
|---|---|
| 「我們做到哪了」「繼續」 | 執行 A → 回報 → 問要不要開始下一個里程碑 |
| 「我 Colab 跑完了」 | 先執行 A，再對照 `instructions_for_me.md` 盤點 `results/colab/` 的完整性；缺件就停下來列清單問使用者，不要硬做。**所有表格數字一律從 raw 輸出重新聚合，不抄 notebook 畫面上的值** |
| 「這一步卡住了」＋錯誤訊息 | 執行 A 恢復脈絡 → 判斷根因 → 列 1–3 個修法與取捨 → 直接修最推薦的 → 最小重現驗證 → 記進 [`docs/troubleshooting.md`](../../../docs/troubleshooting.md) |
| 「可以發佈了嗎」 | 跑 `scripts/verify_publish.py`。它會 fail-closed 檢查 evidence graph、授權、金鑰與 Contributors。**通過不等於可以 push**——push、tag 與 Hugging Face 上傳一律等使用者明確同意 |
