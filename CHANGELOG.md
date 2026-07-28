# Changelog

本專案採用 [Semantic Versioning](https://semver.org/) 管理公開 Release。

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
