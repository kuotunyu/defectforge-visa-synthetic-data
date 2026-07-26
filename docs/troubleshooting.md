# Troubleshooting

> 遇到的坑與解法。**每解掉一個坑就追加一筆**，不要只在對話裡講完就算。
> 已知的 Windows 環境陷阱（事前預防用）另見 [environment.md §4](environment.md)。

格式：

```
## <日期> — <一句話症狀>
**里程碑**：M?
**症狀**：完整錯誤訊息或觀察到的行為
**根因**：
**解法**：
**預防**：（要不要寫進 environment.md 的踩雷清單 / 加成程式斷言）
```

---

## 2026-07-27 — `noise==1.2.2` 在 Python 3.12 / Windows 無法建置

**里程碑**：M1

**症狀**：`uv sync --frozen --python 3.12` 呼叫 MSVC 編譯 `_simplex.c` 時失敗：
`fatal error C1083: 無法開啟包含檔案: 'io.h': No such file or directory`。

**根因**：`noise` 1.2.2 最後發布於 2015 年，PyPI 的 Windows wheel 只到 CPython 3.4；
Python 3.12 只能從 source build。本機有 MSVC Build Tools，但沒有 Windows SDK headers。

**解法**：從直接相依移除 `noise`，M8 的程序噪聲改用已鎖定的 NumPy/scikit-image
實作。重新 `uv lock` 後解析 175 個套件，第二次 `uv sync --frozen` 成功。

**預防**：已寫進 [environment.md §2](environment.md#noise-122-為何不納入) 與 Windows
踩雷清單。不要為這個非核心功能替整台電腦安裝系統 SDK。

---

## 2026-07-27 — VisA tar 沒有外層 `VisA/` 目錄

**里程碑**：M2

**症狀**：第一次把 tar 解到 `${raw}` 後，驗證找不到
`${raw}/VisA/pcb1/Data/Images/Normal`；實際物件目錄出現在 `${raw}/pcb1`。

**根因**：官方 tar 的 12,122 個 members 直接以 `pcb1/`、`capsules/` 等物件目錄開頭，
沒有規格原先假設的外層 `VisA/`。

**解法**：讓 `scripts/download_visa.py` 解壓到設定的 `${visa_raw}`，而不是其父層
`${raw}`。重跑後 pcb1 1,004/100、capsules 602/100 與 mask 數全部通過。

**預防**：下載器的 dry-run 現在明確印出 `${visa_raw}`；資料協定也記錄官方 tar
member 佈局。第一次誤解壓的重複副本因刪除護欄拒絕而暫時保留在 D:，不影響正確資料。

---

## 2026-07-27 — Hugging Face cache 在 Windows 顯示 symlink degraded warning

**里程碑**：M6

**症狀**：首次下載 `facebook/dinov2-base` 時，`huggingface_hub` 警告目前 Windows
未啟用 Developer Mode，cache 無法建立 symlink，可能多佔磁碟；模型仍正常載入。

**根因**：Windows 非系統管理員程序預設不能建立 symlink；這不是模型、權重或 CUDA 錯誤。

**解法**：不為單一約 346 MB 模型提升權限或改系統設定；保留預設 degraded cache。
本次鎖定 revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415` 且 35 個元件全數抽取成功。

**預防**：看到這個 warning 時先確認 cache 空間與模型 checksum/revision；不要把它誤判成
下載失敗，也不要在無人值守時自行開啟 Developer Mode。

---

## 2026-07-27 — 通用前景 ROI 把膠囊桌布與 PCB 陰影誤判為合法區域

**里程碑**：M7

**症狀**：早期 smoke grid 有膠囊瑕疵落在淺色桌布，PCB 則可能落到板體外的陰影；
單靠 border-color distance 與 saturation 聯集會把兩者都納入。

**根因**：兩個物件的可貼區幾何不同。PCB 是單一大致矩形的高飽和板體；膠囊是多個分離的
綠色前景。通用 union ROI 無法同時安全描述兩者。

**解法**：在 `configs/stage_a.yaml` 明確鎖定 object-specific ROI：PCB 取最大 saturation
component 的 bbox 再 erosion；膠囊保留清理後的 saturation components 並 erosion。
正式 1,000 筆由獨立 validator 從 594 張來源背景重建 ROI，逐張確認 100% containment。

**預防**：每個新 object 必須先用 smoke grid 校準 ROI mode；正式輸出同時保留
`roi_bbox`，但不可只信 metadata，必須從原始背景獨立重算。

---

## 2026-07-27 — 大型瑕疵 mask 用隨機中心點放置容易耗盡重試

**里程碑**：M7

**症狀**：大面積或細長 component 即使 legal ROI 內實際存在可放位置，隨機抽中心點仍可能
在最大重試數內全部失敗；提高重試只會讓正式生成變慢且不保證找到。

**根因**：中心點位於 ROI 不代表整個不規則 mask 位於 ROI；合法 top-left 座標通常只占
候選座標中的小部分。

**解法**：以 `cv2.matchTemplate(~legal_roi, component_mask, TM_CCORR)` 一次算出所有與
illegal pixels 零重疊的位置，再由 sample-local PCG64 從合法座標決定性選取。

**預防**：placement 的硬條件永遠以所有非零 mask pixels 為準；生成後再用
`np.all(legal_roi[placed_mask])` fail closed，且由獨立 validator 重驗。

---

## 2026-07-27 — `--no-real-stats` 不能只靠報告欄位自證沒有讀檔

**里程碑**：M8

**症狀**：單純在 metadata 寫 `no_real_stats=true` 無法證明深層 helper、resume 或 validator
未來不會意外開啟 `reports/real_mask_stats.json`，也無法滿足 ADR-011 的可稽核性。

**根因**：旗標與報告是程式自己的宣告，不是檔案存取證據；重構時仍可能把 real-stat
loader 放到兩個 branch 共用的初始化路徑。

**解法**：no-real-stats 行程在分支確定後安裝 Python audit hook，攔截 `open` event；
只要目標是 `real_mask_stats.json` 就 raise `StatsLeakageError`。生成、`--resume` 與獨立
validator 都在這個 guard 下完整跑過；單元測試另 monkeypatch loader，確認固定分布分支不呼叫它。

**預防**：任何新增的 no-real-stats code path 都必須沿用同一 audit guard；不得改成
「執行完寫一個 false」的弱證據，也不得在安裝 guard 前預載統計檔。

---

## 2026-07-27 — M9 只用外觀前景 ROI 無法表達物件內的結構合法區域

**里程碑**：M9

**症狀**：PCB 的飽和前景 bbox 能排除桌面，但會把整塊板都視為同質合法區；膠囊的
saturation ROI 能抓到每顆膠囊，卻無法分辨內部高光、邊界與較有結構的區域。

**根因**：色彩／Otsu 前景回答「物件在哪裡」，不回答「物件內哪些 patch 有可供瑕疵
條件化的局部結構」。單用 DINOv2 score 又可能保留背景紋理，兩者各自都不夠安全。

**解法**：以鎖定 revision 的 DINOv2 patch tokens 計算局部鄰域與全域 cosine
heterogeneity，形態學清理後再與 object-specific 前景取 intersection。score cache key
包含模型設定、manifest SHA256 與所有背景 SHA256，避免舊特徵誤用。

**預防**：新增 object 時先跑 24 格 smoke visualization；cyan ROI 必須只落在物件，
red mask 必須 100% 位於 cyan。正式 validator 必須從原始背景與 cache 重建 ROI，
不可只信 `roi_bbox` metadata。

---

## 2026-07-27 — 反覆標記完整來源 mask 讓 M9 placement 成本不必要地放大

**里程碑**：M9

**症狀**：每次 transform retry 都重新開啟 1.5MP mask 並執行 connected-component label；
2,889 個 placements 會把少數 frozen components 重複解碼、標記數千次。

**根因**：來源只有 35 個 frozen components，但早期實作把 component crop 當成每次抽樣
才建立的臨時資料，沒有利用其不可變性。

**解法**：以 resolved mask path + component id 為 key，在程序內快取驗過 area 的 cropped
binary component；仿射仍為每個 sample 獨立執行。正式 2,889 筆因此能在數分鐘完成。

**預防**：只快取 checksum 已由 frozen manifest 鎖定的不可變輸入；不可快取 placement
座標、亂數狀態或輸出，避免破壞 sample-local determinism 與 resume 語意。
