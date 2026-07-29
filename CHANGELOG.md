# Changelog

本專案採用 [Semantic Versioning](https://semver.org/) 管理公開 Release。

## [未發布]

### 零 Dice 分割 run 的診斷（ADR-030）

- 新增 `scripts/diagnose_zero_dice_segmentation.py`：重跑全部 16 個 physical run 的推論，
  **先重算已發佈指標並要求相符**，再回報預測機率分布。
- 量測結果：所有零 Dice 的 run，其最高預測機率都低於 threshold 0.5；
  所有非零 Dice 的 run 都高於。空 Mask 是**算術上必然**，與模型排序能力無關。
- 峰值位置區分「排序正確但信心不足」與「最有信心的像素是偽陽性」。
- `capsules / std_aug` 的真實瑕疵曝光為 100%，與 `real_only` 僅差標準增強，
  機率天花板卻從 0.9920 掉到 0.3157 —— 至少一個零 Dice 案例與合成資料無關。
- **不調校 threshold、不更換主指標**：在 test 上挑 threshold 等同於用 test 做模型選擇。
  未重跑訓練，`results/*.csv` 位元不變。

### 持續驗證（ADR-029）

- 恢復 GitHub Actions workflow。M38 停止追蹤 `.github/` 時把它一併從遠端移除，
  持續驗證自此停用；2026-07-29 實查遠端只剩 GitHub 自動的 `Dependency Graph`。
- 只公開 `workflows/verify.yml`；PR 與 Issue Template 維持 owner-local。
- 把發佈閘門與持續驗證分開：`model_license_verification_fresh` 的 24 小時時效
  改為僅發佈時強制，CI 改以 `--allow-stale-license-check` 執行，並新增一步
  **重新連 Hugging Face Hub 實查上游授權**（寫入暫存路徑，不動已提交的 hash 鏈）。
- 豁免不竄改回報值：過期時仍如實回報 `false`，只是不列入 `failed_checks`；
  輸出新增 `waived_checks` 與 `failed_checks`，未知的豁免名稱直接拒絕。

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
