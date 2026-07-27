# Filtering & Quality Metrics Spec

> 對應里程碑 M13–M14。指標定義的決策見 [ADR-006](decisions.md#adr-006)。
> 對應課程 Step 2.3「Pseudo labeling & dataset curation」的 **KPI filtering** 與
> `eval` skill 的 per-type `nn_score` / `mnn_score` / `fid`（見 [methodology.md §2.5](methodology.md)）。

---

## 0. 設計原則

1. **輸出 filtered 與 unfiltered 兩版**，兩版都完整保留 `metadata.jsonl`。
   `unfiltered` 版含所有樣本並標 `passed=false` + 拒絕原因；`filtered` 版只含 `passed=true`。
   兩版都是 Phase 2 的實驗組（消融「過濾管線的價值」）。
2. **每一道規則都要能單獨關閉**（CLI flag），才做得出「哪一道規則貢獻最大」的分析。
3. **拒絕原因用 enum，不用自由文字**，這樣漏斗表才能自動聚合。
4. 所有參考統計只從 **few-shot seed 的真實資料**算出，**絕不碰 test**。

---

## 1. 六道過濾規則

| # | 規則 | 分數欄位 | 拒絕條件（初值，M13 依實測校準並記錄） |
|---|---|---|---|
| 1 | **ROI 越界** | `roi_containment` = mask 落在 legal ROI 內的像素比例 | `< 1.0`（要求完全包含） |
| 2 | **mask 面積分布** | `area_zscore`（對真實 mask 面積比的 log 常態擬合） | `\|z\| > 2.5`，或面積比落在真實 5–95 百分位之外 |
| 3 | **mask 長寬比分布** | `aspect_zscore` | `\|z\| > 2.5` |
| 4 | **pHash 去重** | `phash_min_dist` = 與已接受樣本的最小 Hamming 距離 | `< 5`（視為重複） |
| 5 | **DINOv2 語意檢查** | `nn_score`（見 §2）＋ `outlier_score`（到真實 crop 質心的馬氏距離） | `nn_score < τ_low` → 不真實；`nn_score ≥ τ_copy` → 幾乎是複製品；`outlier_score > τ_out` → 離群 |
| 6 | **邊界融合品質** | `seam_score` | `< 0.85` |

### 規則 6 的 `seam_score` 定義
沿 mask 邊界向外取 `k` 像素的環帶，比較「合成圖」與「原始背景圖」在該環帶上的
梯度分布差異：

```
band       = dilate(mask, k) XOR mask
g_syn      = |∇ synthetic[band]|
g_bg       = |∇ background[band]|
seam_score = 1 - normalized_wasserstein_distance(hist(g_syn), hist(g_bg))
```

直覺：好的縫合在 mask 外圍不應該留下人工梯度台階。`k` 初值 5 px。

### 拒絕原因 enum
```
ROI_OVERFLOW | AREA_OUT_OF_RANGE | ASPECT_OUT_OF_RANGE
| PHASH_DUPLICATE | NN_TOO_LOW | NN_TOO_HIGH_COPY | EMBEDDING_OUTLIER | SEAM_POOR
```
一個樣本可以有多個原因，`reject_reasons` 是陣列，漏斗表按「第一個觸發的原因」歸類，
另外提供「每道規則各自的觸發次數」（會大於總拒絕數，這在報表中要註明）。

---

## 2. 品質指標（[ADR-006](decisions.md#adr-006)）

**所有指標在瑕疵 crop 上計算，不是整張影像。** 整張圖被正常背景主導，指標會失去鑑別力。
Embedding 一律用 `facebook/dinov2-base` 的 CLS token，L2 normalize 後算 cosine。

設 `R` = 真實瑕疵 crop（僅 few-shot seed），`G` = 生成瑕疵 crop。

### `nn_score`
```
nn_score(g) = max_{r ∈ R} cos(emb(g), emb(r))
```
- **τ_low**：真實 crop 做 leave-one-out（每個 r 對其餘 R\{r} 取 max cosine）得到分布，
  取其 **5th 百分位**當 τ_low。低於此值 → 比真實樣本之間還不像，判為不真實。
- **τ_copy**：初值 **0.98**，M13 用「把 seed 影像本身輕微擾動後餵進去」校準——
  擾動版應該要被判為 copy。最終值寫進 `reports/filter_report.md`。
- 分瑕疵型各報一份，另報全體。

### `mnn_score`
```
mnn_score = |{ r ∈ R : ∃ g ∈ G, NN_G(r) = g ∧ NN_R(g) = r }| / |R|
```
即 `R` 中「與某個 `g` 互為最近鄰」的比例 → 生成集合對真實瑕疵流形的**覆蓋度／多樣性**。
只追求高 `nn_score` 會導致模式塌縮，`mnn_score` 是它的制衡。**兩個要一起看。**

### FID / KID
`clean-fid` 在 crop 上計算。樣本數只有數百，**FID 不可靠，主報 KID**，FID 僅列出供參考。
報表中要明寫這一點。

正式 generated-vs-real 表使用 degree-3 polynomial kernel 的**未偏 KID**。real-self
健全性檢查則使用包含對角線的 biased polynomial MMD：同一有限集合對自身必為 0；
未偏 U-statistic 排除 self-kernel 對角線，拿同一集合對自身計算時會因小樣本而呈現負值，
不適合拿來做 identity implementation check。noise sanity 使用相同 biased estimator。

### 健全性檢查（M14 必做，不通過不准往下走）
| 檢查 | 期望 |
|---|---|
| 把真實 crop 當作 `G` 餵進去 | `nn_score ≈ 1.0`、`KID ≈ 0`、`mnn_score` 高 |
| 把純雜訊 crop 餵進去 | `nn_score` 明顯低於 τ_low、KID 很大 |
| 所有輸入 crop 的來源路徑 | 全部不在 `splits/test_blocklist.json` 內 |

---

## 3. 輸出

### `reports/filter_report.md`
必含**漏斗表**（每個 generator × 每個瑕疵型一列）：

| generator | type | 生成 | ROI | 面積 | 長寬比 | pHash | DINOv2 | 縫合 | 最終 | 通過率 |
|---|---|---|---|---|---|---|---|---|---|---|

以及：各門檻的最終值與校準過程、`τ_low` / `τ_copy` 的來源分布圖、
被拒樣本的原因分布長條圖。

**驗證**：報表中每個數字都必須能從 `metadata.jsonl` 重新聚合出**完全相同**的值
（寫一支 `scripts/verify_filter_report.py` 做這件事）。

### `reports/generation_quality.md`
每個 generator × 每個瑕疵型的 `nn_score`（mean/median/分布圖）、`mnn_score`、KID、FID，
以及真假並排對照 grid。

### 人工檢查點（M13）
抽 12 張**被拒**與 12 張**通過**的樣本做並排 grid，**自己開起來看**。
若過濾把明顯好的樣本刷掉，調門檻並把調整過程記進 `reports/filter_report.md`——
不准偷偷改了門檻卻不記錄。
