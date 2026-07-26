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
