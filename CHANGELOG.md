# Changelog

本專案採用 [Semantic Versioning](https://semver.org/) 管理公開 Release。

## [1.3.0] - 2026-07-30

### 分割 3-seed 複跑與跨機器重現（ADR-032／033）

- 8 個 formal group × 2 物件 × 3 seeds 全部完成；`results/segmentation.csv` 由 M20 聚合器
  從 raw `training_report.json` 重建為 54 列（48 實跑 ＋ 6 列逐 seed alias）。
- 兩條預註冊規則判定完畢：Dice／AUPRO 方向矛盾為**真實現象**（`pcb1` 2/3 seeds）；
  `capsules/std_aug` 崩潰為**系統性**，ADR-031 維持。
- **推翻一個已發表的結論**：seed 42 的 AUPRO 提升**沒有重現**。兩物件 macro AUPRO Δ
  從 seed 42 的 `+0.1046` 變成 3-seed 的 `-0.1224 ± 0.1976`，負面結論因此**變強**。
- **跨機器 bit-identical 重現**：seed 42 的 16 個模型 SHA256 與已發佈值逐一相同，
  四項指標最大絕對差 `0.00000000`。新增 `scripts/verify_seed42_reproduction.py` 與
  複跑前就已 commit 的基準表 `reports/segmentation_seed42_baseline.csv`。

### 零 Dice 診斷擴大並修正一個過強推論（ADR-034）

- 診斷從 16 個 run 擴大到全部 48 個。22/23 個零 Dice 確實是機率天花板，
  但**有 1 個是「有正像素卻完全打偏」**——因此「零 Dice 一律與空間定位能力無關」不成立。
- **修正工具缺陷**：原腳本把結論寫成固定字串、只填數值，**從未驗證主張是否成立**，
  在單 seed 資料上碰巧正確。改為由資料推導，並新增以本次反例為輸入的回歸測試。

### v3–v5：三個機制候選，四次 gate 全數未過（ADR-035 → ADR-041）

- **v3 歸因 pilot**（ADR-035／036）：查出既有的合成來源排序是**取樣汙染**的產物——
  三個來源消融全在 v1 壞掉的 sampler 下跑，修正後排序**反轉**，舊排序作廢。
- **v4 面積 pilot**（ADR-038／039）：判定 `uninformative`——主判準訂在**已知無鑑別力**的
  指標組合上，**實驗沒有檢驗到自己的假說**。護欄正確觸發，但設計本身有缺陷。
- **v5 三 seed 複跑**（ADR-040／041）：換成有鑑別力的指標後得到**有效的陰性結果**
  `no_effect`。判定**只計入未被看過的 seed 43／44**，seed 42 照常報告但不進判定式。
- 四次 gate 全部 `stopped`，**frozen test 從未被讀取**。曝光、外觀、面積三個機制候選都已排除。

### M9 放置階段直接量測（ADR-037）

- 新增 `scripts/diagnose_placement_geometry.py`：查出 **`pcb1` 的合成放置面積是真實瑕疵的
  6.16 倍**，連最小的放置都比真實中位數大——既有結果中未被發現的缺陷。
- 但 `capsules` 的面積 **100%** 落在真實區間內，因此放置幾何**不是共通解釋**。
- **修正自己的量測錯誤**：第一版拿真實 mask 底下（已有瑕疵）比放置 mask 底下（乾淨背景），
  改為兩邊都量 mask 外的環狀帶，並加測試斷言環狀帶不得包含 mask 本身。

### 揭露的洩漏面終於進入 README（ADR-011／042）

- ADR-011 承諾並列「用統計量」與「不用統計量」兩版結果，分類版做了但 **README 從未提及**。
- 新增 `CLASSIFICATION_LEAKAGE_SURFACE` verified block：**不用統計量反而比較好**
  （pcb1 Macro-F1 `-0.0577`、AUROC `-0.0820`）。被揭露的洩漏面是**負作用**。
- 分割版對照組**決定不補**，理由與成本分析記於 ADR-042，缺口仍留在誠實清單上。


### `capsules/std_aug` 崩潰的診斷（ADR-031）

- 新增 `scripts/diagnose_augmentation_mask_loss.py`。
- **推翻**「增強把小瑕疵裁出畫面」的假設：重放 trainer 實際 draw 序列後，
  pcb1 的空 mask 比率為 0.00%、capsules 僅 1.44%，且唯一被切掉的是瑕疵**最大**的一張。
- 改以訓練 loss 軌跡定位：BCE 在四個 run 都正常下降，只有 `capsules/std_aug` 的
  **Dice 項從頭到尾平的** —— 崩潰發生在**訓練期**，不是測試期的校準假象。
- **不修增強設定、不加步數、不重跑**：那會動到凍結的訓練預算，且是在看過 test 之後才調。
- 明確標示：單一 seed 無法區分系統性與雜訊，此診斷卡在「分割補 3 seeds」。

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
