# Changelog

本專案採用 [Semantic Versioning](https://semver.org/) 管理公開 Release。

## [1.2.2] - 2026-07-29

### 結果呈現（ADR-027）

- README 新增預註冊 3-seed 複跑的 mean ± std；先前只呈現 seed 42 的單次結果。
- README 新增 Dice（threshold 0.5）與 AUPRO（threshold-free）的併列比較。
  兩者對同一批 Segmentation run 給出**相反**方向；預註冊主指標與負面結論不變。
- 限制章節補上兩項既有但未揭露的缺口：Segmentation 完全沒有 seed 複跑，
  以及合成組的真實瑕疵曝光量遠低於 Real-only。
- 未重跑任何訓練，`results/*.csv` 位元不變。

### Agent 工作流（ADR-028）

- 公開 `defectforge`（Orchestrator）與 `df-guard`（防洩漏護欄）兩個 Agent Skill，
  README 新增對應段落；其餘 11 個階段 Skill 維持 owner-local。
- 公開前清除硬編絕對路徑、指向 owner-local 檔案的失效連結與過期敘述。
- `.claude/` 的追蹤邊界由 `.git/info/exclude` 移入版控的 `.gitignore`，
  並在 Publication Audit 中同時強制「兩個 Skill 必須存在」與「其餘必須不被追蹤」。

## [1.2.1] - 2026-07-29

- 精簡 README，保留研究問題、防洩漏設計、正式結果、限制、重現與授權資訊。
- Owner 工作規則、Milestone、Agent Skills、Worklog 與 Colab 交接文件改為本機保存，
  不再出現在公開 Repository。
- 移除已完成階段的 GitHub Workflow／Template、Handoff 與重複佔位檔。
- Publication Audit 新增公開版面檢查，防止 Owner-only 檔案再次被 Git 追蹤。

## [1.2.0] - 2026-07-29

### 新增

- 正體中文 README 與五層 Synthetic Data 系統架構圖。
- `CITATION.cff`、`THIRD_PARTY_NOTICES.md`、`SECURITY.md` 與
  `CONTRIBUTING.md`。
- GitHub Social Preview 與公開發布 Metadata 驗證。

### 變更

- Repository 更名為 `defectforge-visa-synthetic-data`。
- GitHub About、Release Note 與公開文件以正體中文為主。
- Hugging Face Dataset、Model Card 與 Space 更新為新 Repository URL。
- `LICENSE` 回復為標準 MIT 原文，第三方資產條款移至獨立 Notices。

### 安全與維護

- `main` 禁止 Force Push 與刪除，維持 Linear History。
- 啟用 Dependabot Vulnerability Alerts 與 Private Vulnerability Reporting；
  不啟用 Bot 自動 Commit。

## [1.1.0] - 2026-07-28

- 上線正體中文公開 Demo。
- 完成 v2 Domain-balanced Sampling pilot，並依預先註冊 gate 正確停止。
- 新增乾淨 Source Archive 重現與 GitHub Actions 持續驗證。

## [1.0.0] - 2026-07-28

- 首次公開完整的 VisA `pcb1`／`capsules` 少樣本 Synthetic Data 研究。
- 發布防洩漏 Split、Generation Pipeline、Classification／Segmentation
  正式結果、Synthetic Dataset 與 LoRA Weights。
- 如實保留 Synthetic Data 未優於 Real-only 的負面結果。
