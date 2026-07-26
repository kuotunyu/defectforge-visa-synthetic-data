# Environment — Windows-native (no WSL)

> 對應里程碑 M1。**這個專案不使用 WSL。**
> 所有路徑一律 Windows 形式，且一律從 [`configs/paths.yaml`](../configs/paths.yaml) 讀，不得硬編。

---

## 1. 本機規格（2026-07-27 實測）

| 項目 | 值 |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB, driver 591.86 |
| Shell | PowerShell 5.1（另有 Git Bash 可用） |
| uv | 0.11.18 |
| git / git-lfs | 2.41.0.windows.1 / 已安裝 |
| gh CLI | 已登入 **`kuotunyu`**（`gh api user` 回傳 id 61350295）。注意 `gh auth status` 顯示的是舊快取名稱 `tun0000`（帳號改過名），**以 `gh api user` 為準** |
| git 身分 | repo-local 已鎖 `user.name=kuotunyu` / `user.email=[redacted-school-email]`（見 [CLAUDE.md](../CLAUDE.md) 的 Git 署名規則） |
| C: 可用 | 202 GB |
| D: 可用 | 1728 GB |
| Desktop | **未**被 OneDrive 接管（已查 `User Shell Folders`，OneDrive 程序未執行） |

⚠️ 系統上的 `python` 指向 `D:\anaconda3\python.exe`（3.10.9）。
**本專案不用它**，一律用 uv 建的專案虛擬環境。

---

## 2. 建立環境（M1）

```powershell
uv venv --python 3.12
uv sync
```

Python 選 **3.12** 是為了與 Colab runtime 對齊（M1 開工時要先確認 Colab 當時的 Python 版本，
不一致就改成一致的）。

### PyTorch 要走 CUDA index，不能從 PyPI 裝
RTX 4090 是 Ada (sm_89)，用 **cu128** 輪子：

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

M1 執行時**先上 <https://pytorch.org/get-started/locally/> 確認當時的指令與 CUDA 版本**，
確認後把對應的 `[[tool.uv.index]]` / `[tool.uv.sources]` 區塊寫進 `pyproject.toml` 並 `uv lock`。

### 驗證
```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run python -c "import diffusers, peft, cv2, imagehash, sklearn, cleanfid; print('ok')"
```
第一行必須印出 CUDA `True` 與 `NVIDIA GeForce RTX 4090`。

### 版本查證紀錄（2026-07-27）
| 套件 | 當時最新 | 來源 |
|---|---|---|
| torch | 2.13.0（2026-07-08） | [PyPI](https://pypi.org/project/torch/) |
| diffusers | 0.39.0（2026-07-03，requires Python ≥3.10） | [PyPI](https://pypi.org/project/diffusers/) |
| peft | 0.19.1（2026-04-16） | [PyPI](https://pypi.org/project/peft/) |

M1 要**重新查證一次**再鎖版，並把 `uv.lock` 提交進 git。

---

## 3. 路徑與設定

| 用途 | 位置 |
|---|---|
| 專案（git repo） | `C:\Users\3Hml\Desktop\mySyntheticData\1_DefectForge` |
| 大檔 `data_root` | `D:\sdg-data\01-defectforge` |
| HF 快取 | C: 預設 `C:\Users\3Hml\.cache\huggingface`（約 8 GB） |
| 密鑰 | `..\.env`（**在 repo 外**），用 `python-dotenv` 讀 |

`..\.env` 內已有的變數名稱（**只列名稱，絕不 print 值**）：
`HF_TOKEN`、`WANDB_API_KEY`、`GEMINI_API_KEY`(+3 backups)、`OPENAI_API_KEY`、
`DISCORD_WEBHOOK_URL`、`KAGGLE_API_TOKEN`。本專案只需要 `HF_TOKEN`（下載模型與 Phase 2 上傳）。

### 帳號

| 平台 | 帳號 |
|---|---|
| GitHub | **`kuotunyu`**（repo `01-defectforge-visa`） |
| Hugging Face | **`steven0226`**（Phase 2 上傳合成資料集與 LoRA，見 [publish_spec.md](publish_spec.md)） |

---

## 4. Windows 踩雷清單

| 坑 | 症狀 | 解法 |
|---|---|---|
| PowerShell 5.1 沒有 `&&` / `\|\|` | Parser error | 用 `A; if ($?) { B }`；或改用 Bash 工具 |
| PowerShell 沒有 `head`/`tail`/`which`/`touch` | 指令不存在 | `Get-Content -TotalCount N` / `-Tail N`、`(Get-Command x).Source`、`New-Item -ItemType File` |
| DataLoader `num_workers > 0` | Windows 用 spawn，子行程重新 import 主模組 → 無窮遞迴或 pickling error | 腳本預設 `num_workers=0`；若要開多 worker，進入點一定要包 `if __name__ == "__main__":` |
| 路徑長度 260 字元上限 | 深層合成輸出目錄可能爆 | 檔名保持精簡；必要時啟用 Long Path 支援 |
| `cv2.seamlessClone` 的 mask | 要求 8UC1、且 ROI 不得貼到影像邊界 | 縫合前先檢查 bbox 有留邊，不足就退回羽化 alpha |
| `bitsandbytes` 8-bit optimizer | Windows 支援不穩 | 本機 4090 24GB 不需要；Colab（Linux）才用 |
| `xformers` | Windows 輪子常對不上 torch 版本 | **不裝**，用 torch 內建 SDPA |
| conda 的 python 混進來 | 匯入到錯誤環境的套件 | 一律用 `uv run python`，不要直接打 `python` |
| 檔名大小寫 | VisA 影像是 `.JPG`（大寫），Windows 不分大小寫但 Linux/Colab 分 | glob 時同時比對 `.JPG`/`.jpg`，寫進 manifest 的路徑用原始大小寫 |

---

## 5. Colab 端環境

| 項目 | 說明 |
|---|---|
| Runtime | SD2 LoRA → **T4** 足夠；SDXL LoRA → **需 L4**（[ADR-001](decisions.md#adr-001)） |
| Compute units | T4 約 **1.76–1.96 CU/hr**；A100 約 10–15 CU/hr。**L4 費率待 M15 到 Colab 頁面實測確認**。Pro 每月 100 CU |
| 資料 | 先解壓到 `/content/data` 再訓練，**不從掛載的 Drive 直接讀圖** |
| Checkpoint | 定期同步回 `MyDrive/sdg-portfolio/01-defectforge-visa/`，支援斷點續跑 |
| 輸出目錄 | `runs/lora_<model>/<object>/seed_<n>/`，每本 notebook 唯一 |
| Secrets | `from google.colab import userdata` 讀 `HF_TOKEN`，notebook 內無明文金鑰 |
| 檔案回收 | 產出下載後放進 `results/colab/`，路徑寫在 `instructions_for_me.md` |

**注意**：Colab 是 Linux，路徑分隔與大小寫規則與本機不同。跨平台的程式碼一律用
`pathlib.Path`，不要用字串串接。

來源：[Colab 計價與 compute units 說明](https://cloud.google.com/colab/pricing)
