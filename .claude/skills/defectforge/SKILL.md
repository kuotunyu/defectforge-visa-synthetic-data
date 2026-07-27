---
name: defectforge
description: 恢復 01-defectforge-visa 專案的脈絡並決定下一步。當使用者在這個專案裡說「我們做到哪了」「繼續」「接著做」「現在該做什麼」，或隔一段時間回來重啟工作、或不確定該用哪個 df-* 階段 skill 時使用。也用於每個里程碑收尾（更新 PLAN.md、追加 worklog、git commit、產出「換你做」清單）。
---

# DefectForge Orchestrator

這個專案的入口與階段路由器。對應 NVIDIA GTC 2026 課程 agentic flow 的 `anomalygen`
orchestrator skill（見 [docs/skills_roadmap.md](../../../docs/skills_roadmap.md)）。

**這個 skill 不直接做管線工作**，它負責兩件事：
1. **恢復脈絡** — 讓任何一個新 session 在三分鐘內知道現在在哪
2. **收尾** — 每個里程碑結束時，確保 PLAN / worklog / git 三件事都做到

---

## A. 恢復脈絡（開工時）

依序讀這四份，**不要跳過**：

| 順序 | 檔案 | 讀出什麼 |
|---|---|---|
| 1 | `CLAUDE.md` | 分工／實驗鐵律／工作方式／Windows-native 規則 |
| 2 | `PLAN.md` | M0–M15 的勾選狀態 → **第一個沒勾的就是當前里程碑** |
| 3 | `docs/worklog.md` | **最後一筆**：上次做了什麼、未決風險、下一步、換你做 |
| 4 | `docs/decisions.md` | 已定案的 ADR，避免重新爭論已經決定的事 |

然後檢查磁碟上的實際狀態，**不要只信文件**：

```powershell
Get-ChildItem .\splits
Test-Path D:\sdg-data\01-defectforge\raw\VisA
git log --oneline -5
git status --short
```

**文件說已完成、但磁碟上東西不在 → 以磁碟為準，並回報這個落差。**

### 回報格式（給使用者，繁體中文，簡短）
```
目前在 M<n>：<里程碑標題>
上次做到：<worklog 最後一筆的「下一步」>
磁碟狀態：<split 是否已凍結 / VisA 是否已下載 / 有無未 commit 變更>
未決風險：<worklog 最後一筆的風險項，只列還沒解掉的>
建議下一步：<一句話>
```

---

## B. 階段路由

依當前里程碑導向對應 skill（尚未建立的 skill 表示還沒做到那個階段，
此時直接依 `PLAN.md` 的該項描述與對應規格文件執行，做完再把 skill 建起來）：

| 里程碑 | Skill | 規格文件 |
|---|---|---|
| M1–M2 | `df-setup` | [docs/environment.md](../../../docs/environment.md)、[docs/data_protocol.md](../../../docs/data_protocol.md) §1–2 |
| M3–M5 | `df-split` | [docs/data_protocol.md](../../../docs/data_protocol.md) §3–5 |
| M6 | `df-types` | [docs/data_protocol.md](../../../docs/data_protocol.md) §5.3、ADR-002 |
| （任何生成前） | `df-guard` | [docs/data_protocol.md](../../../docs/data_protocol.md) §4.3、§6 |
| M7–M8 | `df-stage-a` | [docs/synthesis_spec.md](../../../docs/synthesis_spec.md) §3 |
| M9 | `df-prep-testcase` | [docs/synthesis_spec.md](../../../docs/synthesis_spec.md) §4.1、§2 |
| M10–M11 | `df-finetune` | [docs/synthesis_spec.md](../../../docs/synthesis_spec.md) §4.2 |
| M12 | `df-sdg` / `df-refine` | [docs/synthesis_spec.md](../../../docs/synthesis_spec.md) §4.3–4.4 |
| M13 | `df-filter` | [docs/filtering_spec.md](../../../docs/filtering_spec.md) §1 |
| M14 | `df-eval` | [docs/filtering_spec.md](../../../docs/filtering_spec.md) §2 |
| Phase 2 收尾 | `df-release` | [docs/experiment_protocol.md](../../../docs/experiment_protocol.md) |

---

## C. 每次動手前的紅線檢查

不論做哪個階段，先確認：

- [ ] **Test 不可觸碰**：任何要讀的影像路徑先比對 `splits/test_blocklist.json` 的 SHA256
- [ ] **Split 已凍結**：M4 之後，`splits/split_manifest.json` 的 SHA256 必須與
      `splits/MANIFEST.sha256` 相符；不符表示有人動過，**停下來回報，不要自行修復**
- [ ] **路徑不硬編**：所有路徑從 `configs/paths.yaml` 讀
- [ ] **不用 WSL**：不得出現 `/mnt/c` 或 `~/sdg-portfolio` 形式的路徑
- [ ] **金鑰不進 git**：不把 `.env` 內容寫進任何檔案或印出來
- [ ] **>2GB 下載或花錢動作先問使用者**

M4 之後這些檢查由 `df-guard` skill 執行；在它建立之前，在這裡手動逐項確認。

---

## D. 里程碑收尾（做完一個 M 就跑一次）

1. **驗證**：跑 `PLAN.md` 該里程碑的「驗證」欄位每一項，逐項回報通過與否。
   **有任何一項沒過就不准勾選**。
2. **自己看圖**：若該里程碑有產出圖表或樣本圖，用 Read 工具打開檢視；
   不合理就修，不要只看檔案存在就當完成。
3. **勾選** `PLAN.md` 對應項目（`[ ]` → `[x]`）。
4. **追加 `docs/worklog.md` 一筆**，用檔案開頭規定的六段格式。
5. **重大選型追加 ADR** 到 `docs/decisions.md`（只追加不改寫）。
6. **踩到的坑追加** 到 `docs/troubleshooting.md`。
7. **建立該階段的 skill**（若尚未建立），照
   [docs/skills_roadmap.md](../../../docs/skills_roadmap.md) 的撰寫規範。
8. **git commit**，訊息用 `<type>(M<n>): <英文簡述>`，例如 `feat(M4): freeze split manifest with pHash grouping`。
9. **給使用者「換你做」清單**（繁體中文，具體到可以照做）。

---

## E. 常見情境

| 使用者說 | 你要做 |
|---|---|
| 「我們做到哪了」「繼續」 | 執行 A → 回報 → 問要不要開始下一個里程碑 |
| 「我 Colab 跑完了」 | 先執行 A，再對照 `instructions_for_me.md` 盤點 `results/colab/` 的產出完整性；缺件就停下來列清單問使用者，不要硬做。**所有表格數字一律從 raw 輸出重新聚合，不抄 notebook 畫面上的值** |
| 「這一步卡住了」＋錯誤訊息 | 執行 A 恢復脈絡 → 判斷根因 → 列 1–3 個修法與取捨 → 直接修最推薦的 → 最小重現驗證 → 記進 `docs/troubleshooting.md` |
| 「可以發佈了嗎」 | 這是 Phase 2 的事。用 `df-release`（尚未建立時，依 `docs/experiment_protocol.md` §4 與 `publish-repo` skill 執行） |
