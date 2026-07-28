# Security Policy

## 支援範圍

目前僅維護 Latest Release。DefectForge 是 research／evaluation 專案，不是
production AOI、品質放行或安全關鍵系統。

| 版本 | 安全更新 |
|---|---|
| `1.2.x` | 支援 |
| `< 1.2` | 不再主動維護 |

## 回報安全問題

請使用 GitHub 的
[Private Vulnerability Reporting](https://github.com/kuotunyu/defectforge-visa-synthetic-data/security/advisories/new)
私下回報。請勿在公開 Issue 放入 Token、個資、未公開漏洞細節或可直接利用的 Payload。

回報內容建議包含：

- 受影響的 Commit、版本或公開端點
- 最小重現步驟與預期影響
- 是否涉及上傳影像、Model Artifact、Path Traversal 或任意程式執行
- 已知的緩解方式

本專案不提供正式 SLA；確認問題後會先限制影響範圍，再公開修正與必要的
Release Note。

## 研究邊界

- 公開 Demo 不保存使用者上傳影像，但使用者仍不應上傳機密或個人資料。
- Dataset、Synthetic Images 與 Model Weights 不受 Source Code 的 MIT License
  涵蓋；詳見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- 安全問題不應透過重新挑選研究樣本或改寫凍結結果來處理。
