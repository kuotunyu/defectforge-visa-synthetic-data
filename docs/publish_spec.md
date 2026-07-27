# Publish Spec — GitHub 與 Hugging Face 發佈規格

> 對應里程碑 **M24**。**轉 public 前一定要使用者過目。**
> 相關：[CLAUDE.md](../CLAUDE.md) 的 Git 署名規則、[experiment_protocol.md §7](experiment_protocol.md) 防洩漏檢查表。

---

## 1. 目標位置

| 平台 | 帳號 | Repo | 授權 |
|---|---|---|---|
| GitHub | **`kuotunyu`** | `01-defectforge-visa` | MIT（**僅程式碼**） |
| Hugging Face Dataset | **`steven0226`** | `defectforge-visa-synthetic` | CC BY 4.0（繼承 VisA） |
| Hugging Face Model | **`steven0226`** | `defectforge-visa-lora` | CreativeML Open RAIL++-M（繼承底模） |

### ⛔ Contributors 紅線
commit 不得含 `Co-Authored-By`，PR 內文不得含任何 Claude／產生器署名。
發佈前必跑：
```powershell
git shortlog -sne --all
git log --format="%(trailers:key=Co-authored-by,valueonly)" | Select-String "\S"
```
第一個指令必須只有
`kuotunyu <61350295+kuotunyu@users.noreply.github.com>` 一列；
第二個必須沒有輸出。

---

## 2. 授權鏈（README 與兩張 card 都要照抄）

| 資產 | 授權 | 我們的義務 |
|---|---|---|
| VisA 原始資料 | CC BY 4.0 | 標註來源與論文引用；衍生物可公開發佈 |
| `sd2-community/stable-diffusion-2-inpainting` | CreativeML Open RAIL++-M | 原 repo 不可用後採鎖版保存 mirror；遵守 use-based restrictions |
| `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | 同上 | 同上 |
| `facebook/dinov2-base` | Apache-2.0 | 標註 |
| **我們生成的合成影像** | **CC BY 4.0**（衍生自 VisA） | 標註 VisA；同時揭露底模授權 |
| **我們訓練的 LoRA 權重** | **CreativeML Open RAIL++-M**（衍生自底模） | 隨附底模授權條文連結 |
| 本 repo 程式碼 | MIT | — |

**M24 前必須重新查證**兩個底模的 HF 頁面授權是否變更、是否變成 gated。

---

## 3. HF Dataset — `steven0226/defectforge-visa-synthetic`

### 3.1 內容
```
data/
  stageA_copypaste/{filtered,unfiltered}/{images,masks}/  metadata.jsonl
  stageA_procedural/…
  stageA_procedural_norealstats/…
  stageB_sd2/{original,searched}/{filtered,unfiltered}/…
  stageB_sdxl/…
splits/            # split_manifest.json, defect_types.json, test_blocklist.json, checksums
README.md          # dataset card
```

**只上傳合成影像，不重新散佈 VisA 原始影像**（使用者自己去 AWS Open Data 下載）。
`metadata.jsonl` 裡的 `source.background_image` 只存**相對路徑與 SHA256**，不存影像本身。

### 3.2 Dataset card 必含（雙語：英文為主 + 繁中摘要）
1. **這是什麼**：VisA pcb1/capsules 的合成瑕疵資料，few-shot（每物件 10 張真實瑕疵）生成
2. **怎麼做的**：Stage A（copy-paste / procedural）+ Stage B（SD2 / SDXL inpainting LoRA），
   連結 [methodology.md](methodology.md)
3. **標註**：mask 是生成時使用的 mask，**完全自動、零人工標註**
4. **filtered vs unfiltered**：兩版都提供，過濾規則與門檻連結 [filtering_spec.md](filtering_spec.md)
5. **每筆的 provenance 欄位說明**（schema 見 [synthesis_spec.md §1](synthesis_spec.md)）
6. **防洩漏聲明**：生成端從未接觸 test；基底 split 是 `2cls_highshot`，
   test set 的 SHA256 blocklist 一併附上，讓別人可以自己驗
7. **限制**：
   - 瑕疵型別是 **pseudo-type**（VisA 沒有官方 per-image type label），見 [ADR-002](decisions.md#adr-002)
   - 只有 2 個物件、每物件 10 張真實瑕疵種子，多樣性有限
   - procedural 預設版用了真實 mask 的**統計量**（[ADR-011](decisions.md#adr-011) 的正式措辭照抄）
   - 已知的失敗樣態（附幾張範例圖）
8. **授權與引用**：§2 的表格 + VisA / AnomalyDiffusion / DRAEM / Copy-Paste 的 citation
9. **下游成效**：連結 GitHub repo 的實驗結果（**含負面結果**）

---

## 4. HF Model — `steven0226/defectforge-visa-lora`

內容：`lora_sd2/{pcb1,capsules}/`、`lora_sdxl/{pcb1,capsules}/`，各含 adapter 權重 +
learned token embedding + 訓練用的最終 config。

Model card 必含：底模與版本、訓練資料（10 張／物件，指向 split manifest）、
trigger token 清單與對應的瑕疵型別、推論範例程式碼（含 crop-to-ROI 與 blend-back，
否則別人直接用會得到爛結果）、**過擬合警告**（10 張訓練圖）、授權、限制。

---

## 5. GitHub repo

### 5.1 必須存在
- `README.md`：Problem → Method（mermaid）→ Experiments（五組設計 + 防洩漏聲明）→
  Results（含負面結果）→ Reproduce → License & Citations，英文為主 + 繁中摘要
- `PLAN.md`、`docs/`（全部）、`CLAUDE.md`
- `uv.lock`（鎖版）
- `splits/`（manifest、checksums、defect_types、test_blocklist）— **這是可信度的核心，一定要公開**
- `results/*.csv` 原始數字 + `scripts/verify_readme.py`
- `assets/demo.gif`
- `notebooks/`（Colab 用）
- `.claude/skills/`（**13 個 skill，這是差異化賣點，要公開**）

### 5.2 必須不存在
- 任何 API key、token、`.env`
- 個人絕對路徑（例如 Windows 使用者目錄）→ 發佈前全 repo grep
- 大型二進位（影像、權重）→ 那些在 HF
- 任何 `Co-Authored-By` 或產生器署名

---

## 6. M24 發佈檢查表（逐項打勾，全綠才問使用者要不要轉 public）

**重現性**
- [ ] 開一個全新虛擬環境，照 README 的 Reproduce 章節從頭走一遍（訓練步驟用 `--dry-run` 或 `--smoke`）
- [ ] 走不通的地方修 README 或程式，不是修檢查表

**數字誠實性**
- [ ] `scripts/verify_readme.py` 全綠：README 每張表都能從 `results/` 重算出來
- [ ] `scripts/verify_filter_report.py` 全綠
- [ ] 負面結果（若有）已寫進 Limitations，沒有被靜靜拿掉

**防洩漏**
- [ ] `scripts/verify_splits.py` 重跑 [ADR-007](decisions.md#adr-007) 四項斷言全綠
- [ ] 確認評測用的是 highshot test；訓練集 ∩ test == ∅
- [ ] split manifest、seed、SHA256 齊全且與實際使用一致

**授權**
- [ ] §2 的授權表在 README、dataset card、model card 三處一致
- [ ] 兩個底模的 HF 頁面授權已重新查證，未變更
- [ ] `uv run python scripts/verify_model_licenses.py` 重查 pinned revision、license、public／ungated 狀態
- [ ] VisA 的 CC BY 4.0 標註完整

**安全**
- [ ] 全 repo 掃描 token 格式、被追蹤的 `.env`、Windows 使用者絕對路徑 → 0 命中
- [ ] `git shortlog -sne --all` 只有 `kuotunyu`
- [ ] trailer 掃描無輸出
- [ ] `uv run python scripts/verify_publish.py` 全綠（此命令只稽核，不發佈／不上傳）

**HF**
- [ ] dataset card 與 model card 完整（§3.2、§4）
- [ ] 上傳先 `--dry-run` 看清單，再加 `--confirm` 實際上傳
- [ ] 上傳後自己下載回來抽查幾筆，確認 metadata 與影像對得上

**最後**
- [ ] 產出一頁驗收報告（通過項／修正項／殘留風險）
- [ ] **等使用者說 OK，才把 GitHub repo 與 HF 轉 public**
