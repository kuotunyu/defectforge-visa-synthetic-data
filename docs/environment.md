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
| git 身分 | repo-local 已鎖 `user.name=kuotunyu` / `user.email=61350295+kuotunyu@users.noreply.github.com`（見 [CLAUDE.md](../CLAUDE.md) 的 Git 署名規則） |
| C: 可用 | 182 GiB（M1 preflight） |
| D: 可用 | 1728 GB |
| Desktop | **未**被 OneDrive 接管（已查 `User Shell Folders`，OneDrive 程序未執行） |

⚠️ 系統上的 `python` 指向 `D:\anaconda3\python.exe`（3.10.9）。
**本專案不用它**，一律用 uv 建的專案虛擬環境。

---

## 2. 建立環境（M1）

```powershell
uv lock --python 3.12
uv sync --frozen --python 3.12
```

實際建立的是 uv 管理的 **CPython 3.12.13**。`--frozen` 保證安裝完全以已提交的
`uv.lock` 為準，不在安裝時悄悄重解依賴。

### PyTorch 要走 CUDA index，不能從 PyPI 裝

**為什麼**：PyPI 上的 `win_amd64` 輪子是 **CPU-only**（約 122 MB，對比 Linux CUDA 輪子的
約 527 MB——PyPI 的檔案大小限制使得 Windows 的 CUDA 版本從來不上傳 PyPI）。
裸的 `pip install torch` 會安裝成功、不報錯，然後 `torch.cuda.is_available()` 回 `False`，
4090 整場閒置。**這是本專案最容易安靜出錯的一步。**

**用 cu130，不是 cu128。**（2026-07-27 實地確認）

- **cu128 index 最高只到 torch 2.11.0**，沒有 2.12／2.13——CUDA 12.8 自 torch 2.12 起
  已從標準發佈矩陣移除。原本這份文件同時寫著「用 cu128」與「torch 2.13.0」，那個組合不存在
- cu130 index 上已確認存在：`torch-2.13.0+cu130-cp312-cp312-win_amd64.whl`、
  `torchvision-0.28.0+cu130-cp312-cp312-win_amd64.whl`
- 本機驅動 591.86 ≥ cu130 要求的 580.88，過關

> 附帶更正一個常見誤解：選 index 的依據**不是** sm_89。RTX 4090 的 sm_89 其實
> 不在任何官方輪子的公開 arch 清單裡；它能跑是因為 CUDA 保證 cubin 在同一 major
> compute capability 內前向相容（sm_86 的二進位可在 sm_89 上執行）。
> 只有在自己編譯第三方 CUDA extension 時才需要顯式設 `TORCH_CUDA_ARCH_LIST="8.9"`。

`pyproject.toml` 用 `explicit = true`，讓這個 index **只**服務 torch 與 torchvision，
其餘套件仍走 PyPI：

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu130" }]
torchvision = [{ index = "pytorch-cu130" }]
```

然後 `uv sync` 即可，**不要用 `pip install torch`**。

M1 已在 2026-07-27 **重新確認** CUDA 版本與可用輪子，並產生、提交 `uv.lock`。
（`pytorch.org/get-started/locally/` 的頁面曾回傳疑似快取的舊值，
比較可靠的做法是直接看 <https://download.pytorch.org/whl/cu130/torch/> 的檔名清單。）

### 驗證
```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run python -c "import diffusers, peft, cv2, imagehash, sklearn, cleanfid; print('ok')"
```
第一行必須印出 CUDA `True` 與 `NVIDIA GeForce RTX 4090`。

### 版本查證紀錄（2026-07-27）
| 套件 | 當時最新 | 來源 |
|---|---|---|
| torch | 2.13.0（2026-07-08），**要用 `+cu130` 變體** | [cu130 index](https://download.pytorch.org/whl/cu130/torch/) |
| torchvision | 0.28.0（`+cu130`） | 同上 |
| diffusers | 0.39.0（2026-07-03，requires Python ≥3.10） | [PyPI](https://pypi.org/project/diffusers/) |
| peft | 0.19.1（2026-04-16） | [PyPI](https://pypi.org/project/peft/) |
| transformers | **5.14.1**（2026-07-16） | [PyPI](https://pypi.org/project/transformers/) |
| timm | 1.0.28（2026-07-11） | [PyPI](https://pypi.org/project/timm/) |
| accelerate | 1.14.0（2026-06-11） | [PyPI](https://pypi.org/project/accelerate/) |
| clean-fid | 0.1.35 | [PyPI](https://pypi.org/project/clean-fid/) |

### clean-fid 0.1.35 與 SciPy 1.17 的 FID 相容性

M14 實測發現 clean-fid 0.1.35 的直接 FID 路徑仍呼叫
`scipy.linalg.sqrtm(..., disp=False)`，但目前鎖定的 SciPy 1.17 已移除 `disp` 參數。
本專案仍使用 clean-fid 的 clean-mode Inception feature extractor；FID 本身改用低秩
恆等式精確計算：若 centered covariance factors 為 `A`、`B`，則 covariance
square-root trace 等於 `A.T @ B` 的 nuclear norm。單元測試已在小維度與標準
covariance `sqrtm` 公式逐值比對。這也避免在每類只有 3–23 張 real crop 時建立
2048×2048 奇異 covariance。

### transformers v5 相容性驗證結果

`transformers` v5.0.0 於 2026-01-26 發佈，目前 5.14.1。v5 是**破壞性改版**：

- **image processor 的快慢版合併並改名**（例如 `XxxImageProcessorFast` 不再存在，
  原本的快版直接佔用 `XxxImageProcessor` 這個名字）。任何 2025 年寫的教學抄下來
  會 ImportError 或行為悄悄不同
- **`from_pretrained` 的預設 dtype 從 `float32` 改為 `"auto"`**，模型會以存檔時的精度載入，
  可能造成與 v4 的**靜默數值差異**。一律明確傳 `dtype=`（注意 `torch_dtype=` 已棄用）
- TF / Flax 類別全部移除；量化的捷徑 kwargs 移除

**本專案同時用 `diffusers` 0.39 ＋ `peft` ＋ `transformers`**（DINOv2 特徵抽取）。

**2026-07-27 實測：依賴解析與 runtime import 都通過。** `uv lock` 成功解析
175 個套件，得到：

| 套件 | 解析結果 |
|---|---|
| torch | **2.13.0+cu130**（來源 registry 確認是 `https://download.pytorch.org/whl/cu130`） |
| torchvision | 0.28.0+cu130 |
| transformers | 5.14.1 |
| diffusers | 0.39.0 |
| peft | 0.19.1 |
| accelerate | 1.14.0 |

解析只證明沒有宣告層級的版本衝突，因此 M1 另外實際載入全部核心套件與
Diffusers inpainting pipeline API：

```powershell
uv run python -c "import diffusers, peft, transformers; from diffusers import AutoPipelineForInpainting; print('ok', transformers.__version__, diffusers.__version__)"
```

結果成功。Transformers 會印出 `Siglip2ImageProcessorFast` 已棄用的警告，但沒有例外；
這是上游相容層訊息，不影響目前使用的 DINOv2 或 inpainting API。未來若變成錯誤，
再依實際堆疊決定是否降版並追加 ADR。

### `noise` 1.2.2 為何不納入

最初骨架曾使用 `noise` 產生 Perlin noise，但該套件最後發布於 2015 年，Windows wheel
只到 CPython 3.4。Python 3.12 會被迫現場編譯 C extension，而本機未安裝 Windows SDK
headers。M8 的程序噪聲因此改由 NumPy/scikit-image 實作，避免為非核心功能修改整台
電腦的 C++ 系統工具鏈；方法與 CLI 契約不變。

---

## 3. 路徑與設定

| 用途 | 位置 |
|---|---|
| 專案（git repo） | git checkout 的目前工作目錄 |
| 大檔 `data_root` | `D:\sdg-data\01-defectforge` |
| HF 快取 | 預設 `%USERPROFILE%\.cache\huggingface`（約 8 GB） |
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
| `noise==1.2.2` 在 Python 3.12 編譯失敗 | 找不到 Windows SDK `io.h`；PyPI 無現代 Windows wheel | 不使用該套件；程序噪聲以 NumPy/scikit-image 實作 |
| 檔名大小寫 | VisA 影像是 `.JPG`（大寫），Windows 不分大小寫但 Linux/Colab 分 | glob 時同時比對 `.JPG`/`.jpg`，寫進 manifest 的路徑用原始大小寫 |

---

## 5. Colab 端環境

| 項目 | 說明 |
|---|---|
| Runtime | SD2 LoRA → **T4** 足夠；SDXL LoRA → **需 L4**（[ADR-001](decisions.md#adr-001)） |
| Compute units | T4 約 **1.76–1.96 CU/hr**；A100 約 10–15 CU/hr。M11 的 L4 執行前未記錄 CU，不能誠實回推；已保留實測 wall time 與 peak VRAM，未虛構 CU 數字。Pro 每月 100 CU |
| 資料 | 先解壓到 `/content/data` 再訓練，**不從掛載的 Drive 直接讀圖** |
| Checkpoint | 定期同步回 `MyDrive/sdg-portfolio/01-defectforge-visa/`，支援斷點續跑 |
| 輸出目錄 | `runs/lora_<model>/<object>/seed_<n>/`，每本 notebook 唯一 |
| Secrets | `from google.colab import userdata` 讀 `HF_TOKEN`，notebook 內無明文金鑰 |
| 檔案回收 | 產出下載後放進 `results/colab/`，路徑寫在 `instructions_for_me.md` |

**注意**：Colab 是 Linux，路徑分隔與大小寫規則與本機不同。跨平台的程式碼一律用
`pathlib.Path`，不要用字串串接。

來源：[Colab 計價與 compute units 說明](https://cloud.google.com/colab/pricing)
