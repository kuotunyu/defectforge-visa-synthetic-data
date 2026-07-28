# Contributing

感謝你關注 DefectForge。

## 目前的貢獻政策

這個 Repository 是單一作者、SHA256-bound 的研究成果。為保留發布證據與 GitHub
Contributors 僅有 `kuotunyu` 的專案約束，目前**不接受外部 Code Commit 或 Pull
Request 合併**。

你仍可透過 Issue 提供：

- 可重現的 Bug Report
- 文件錯誤或失效連結
- 在其他 VisA 物件上的獨立重現結果
- 不接觸 Frozen Test Set 的新研究假設

安全問題請勿建立公開 Issue，請依 [SECURITY.md](SECURITY.md) 使用 Private
Vulnerability Reporting。

## Owner-managed Change

由 Owner 執行的變更必須：

1. 使用 `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` 作為 author 與
   committer。
2. 不加入 `Co-Authored-By` 或任何生成器署名。
3. 不改寫 v1 凍結表格、Test Protocol 或負面結果。
4. 通過 Ruff、完整 Pytest、Publication Audit 與 GitHub Actions。
5. 若更動 README、License Chain 或公開 Artifact，必須重新產生對應 SHA256
   evidence。

## 本機驗證

```powershell
uv sync --frozen --python 3.12
uv run --frozen ruff check .
uv run --frozen pytest -q
uv run --frozen python scripts/verify_publish.py
```
