# CLAUDE.md — 01-defectforge-visa

> 每個 session 開工前必讀。與 [PLAN.md](PLAN.md)、[docs/worklog.md](docs/worklog.md)、[docs/decisions.md](docs/decisions.md) 一起構成本專案的長期記憶。
> 快速恢復脈絡：直接呼叫 `/defectforge` skill。

---

## 【分工】本機 vs Colab

**Colab notebook**（超過約 30 分鐘的 GPU 訓練一律走這裡）
- 資料先解壓到 `/content/data` 再訓練，**絕不**直接從掛載的 Drive 讀圖訓練
- checkpoint 定期同步回 Drive 的 `MyDrive/sdg-portfolio/01-defectforge-visa/`
- 必須支援**斷點續跑**（偵測既有 checkpoint 自動接續）
- 每個平行 notebook 用**唯一輸出目錄**：`runs/<group>/<model>/seed_<n>/`
- token 只從 Colab Secrets 讀（`from google.colab import userdata`），絕不寫進 notebook

**本機 RTX 4090（24GB, Windows 11 native）**
- 資料處理、split 凍結、大量離線合成生成、DINOv2/pHash 批次過濾、評測、Gradio demo
- ≤30 分鐘的訓練、以及所有 notebook 的 1-step smoke test
- 生成優先跑本機以節省 Colab compute units

**本專案的實際分配**（[ADR-008](docs/decisions.md#adr-008)）：

| 工作 | 跑在哪 |
|---|---|
| SD2-inpainting LoRA（865M @512、10 張圖） | **本機**，估 20–30 分鐘。實測超過 30 分鐘就改回 Colab 並更新 ADR |
| SDXL-inpainting LoRA（2.6B @1024） | Colab **L4** |
| Phase 2 SegFormer-B0 分割（9 組） | Colab **T4** |
| Phase 2 ConvNeXt-Tiny 分類（五組 + 合成量掃描） | 本機 |
| 所有生成、過濾、評測、demo | 本機 |

訓練邏輯只寫一份在 `src/training/`，Colab notebook 只是薄封裝，**不得複製訓練迴圈**。
Colab 產出由使用者放回 `results/colab/` 後，Claude Code 才接手分析。

---

## 【實驗鐵律】

**五組對照**（Phase 2 執行，協定見 [docs/experiment_protocol.md](docs/experiment_protocol.md)）
1. Real-only
2. + Standard Augmentation
3. + Unfiltered Synthetic
4. + Filtered Synthetic
5. Full-real 上限（適用時）

**不可違反的規則**
- Validation / Test **只用真實資料**；generator、過濾器、分型器都**不得接觸 Test**
- **基底 split 是 `2cls_highshot`，唯一 test set 是 highshot test**。
  絕不可拿 `2cls_fewshot` 的 test 來評測——那兩套切法混用會洩漏一半的測試瑕疵（[ADR-007](docs/decisions.md#adr-007)）
- 先凍結 split manifest（`splits/*.json`＋seed=42＋來源檔 SHA256）**才能**開始生成
- 近似圖片先用 pHash 分群，**同群必須同 split**
- 全組合先跑 1 seed；Real-only 與最佳 Filtered 組補到 3 seeds 報 mean±std
- 合成樣本的標籤**一律由生成流程自動產生**，人工只做抽查觀察
- 每筆合成樣本記 provenance：generator、來源圖/mask、seed、參數、filter score、拒絕原因
- **若 synthetic 沒有提升，如實報告並分析原因**，不准挑選性隱藏實驗
- 第 1–4 組吃的**真實影像必須完全相同**（見 [ADR-003](docs/decisions.md#adr-003)），合成只能是增量

---

## 【工作方式】

- 開工先把本階段拆成 [PLAN.md](PLAN.md) 里程碑，**每項附驗證方法**，做完勾掉
- **繁體中文**溝通；程式碼註解與 README 用**英文**
- >2GB 下載或任何花錢動作**先問使用者**
- 套件版本、模型名稱、價格**一律先上網查證再選型**，並把來源連結寫進文件
- 自己產的圖表與樣本圖**要自己打開檢視**，不合理就修
- 每階段結束：更新 `PLAN.md` + 追加 `docs/worklog.md` 一筆 + `git commit` + 給使用者「**換你做**」清單
- 重大選型寫成 ADR 追加到 [docs/decisions.md](docs/decisions.md)，**只追加不改寫**
- 踩到的坑寫進 [docs/troubleshooting.md](docs/troubleshooting.md)
- **API key 絕不進 Git**

### Git 署名規則（不可違反）

**GitHub 的 Contributors 只能出現 `kuotunyu` 一個人。**

- commit message **不准**加 `Co-Authored-By:` trailer（任何人都不行，包含 Claude）
- commit message **不准**出現 `🤖 Generated with Claude Code` 之類的署名
- PR 內文同樣不加任何 Claude 署名或產生器標記
- commit 的 author 與 committer 一律是
  `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`
- 已設 repo-local `user.name` / `user.email` 鎖住身分，不依賴 global 設定

commit 前自檢：
```powershell
git log -1 --format="%an <%ae>"; git log -1 --format="%B" | Select-String "Co-[Aa]uthored-[Bb]y|Generated with"
```
第一行必須是 `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`，
第二行必須沒有任何輸出。

---

## 【本機環境】Windows-native，不使用 WSL

細節見 [docs/environment.md](docs/environment.md)。以下是每次都會踩的重點：

- **這個專案不使用 WSL**。所有路徑用 Windows 形式，不准出現 `/mnt/c/...` 或 `~/sdg-portfolio/...`
- Shell 是 **PowerShell 5.1**：沒有 `&&` 與 `||`，用 `A; if ($?) { B }`；沒有三元運算子；`head`/`tail`/`which`/`touch` 都不存在
- **不得硬編絕對路徑**。所有路徑一律讀 [configs/paths.yaml](configs/paths.yaml)
- 大檔（VisA 原始資料、合成影像、runs）放 `D:\sdg-data\01-defectforge`；專案資料夾只留程式碼、設定、文件、`splits/`、`reports/` 小圖
- Hugging Face 快取沿用預設使用者快取位置（`%USERPROFILE%\.cache\huggingface`）
- 密鑰放在上層的 `..\.env`（**不在** repo 內），用 `python-dotenv` 讀，絕不 print 內容
- PyTorch DataLoader 在 Windows 用 spawn：`num_workers>0` 時所有進入點必須包 `if __name__ == "__main__":`，寫腳本時預設 `num_workers=0`

---

## 【無人值守執行】

使用者可能在睡覺時讓你連續執行多個里程碑。模式是**跑到底，任何驗證失敗即停**
（[ADR-012](docs/decisions.md#adr-012)）。完整規則書：**[docs/autonomy_policy.md](docs/autonomy_policy.md)**。

三條最重要的：
1. 每個里程碑跑完 `PLAN.md` 的驗證欄，**全綠才前進**。任何一項不過就**立刻停止**，
   寫 `reports/handoff/<date>.md`，**不得降低標準、不得跳過、不得「先繼續之後再修」**
2. 腳本的 CLI 參數一律照 [docs/interfaces.md](docs/interfaces.md)，**不准自己發明參數名稱**
3. 需要花錢、>2GB 下載、push/發佈、Colab 訓練 → **一律停下來等人**

---

## 【專案座標】

| 項目 | 值 |
|---|---|
| 本機資料夾 | git checkout 的目前工作目錄 |
| 未來 GitHub repo | `kuotunyu/01-defectforge-visa` |
| 未來 Hugging Face | `steven0226`（合成資料集 + LoRA 權重，見 [docs/publish_spec.md](docs/publish_spec.md)） |
| 資料集 | VisA（CC BY 4.0），先做 `pcb1` 與 `capsules` |
| 目標 | 少樣本瑕疵情境下，證明合成資料能提升**瑕疵分類**與**瑕疵區域分割** |
| 方法論來源 | NVIDIA GTC 2026 *Few-shot Industrial SDG (Cosmos AnomalyGen)*，用開源工具復刻，見 [docs/methodology.md](docs/methodology.md) |
| 目前階段 | Phase 2 M16（ConvNeXt-Tiny 分類實驗）。Phase 1 M0–M15 已由 `scripts/verify_phase1.py` 驗收通過；分割仍等 M18 |
