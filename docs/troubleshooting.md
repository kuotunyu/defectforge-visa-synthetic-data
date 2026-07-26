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
