---
title: DefectForge VisA 瑕疵檢測
emoji: 🏭
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: other
short_description: VisA 瑕疵分類與分割的可驗證研究 Demo。
---

# DefectForge · VisA 瑕疵影像檢測

這是
[`kuotunyu/01-defectforge-visa`](https://github.com/kuotunyu/01-defectforge-visa).
的公開研究 Demo。選擇 `pcb1` 或 `capsules` 並上傳待檢影像，即可執行正式選定的
ConvNeXt-Tiny classifier 與 SegFormer-B0 segmenter。

## 證據邊界

- 每個 SafeTensors 檔案載入時都會對照生成的 SHA256 manifest。
- Checkpoint 在 evaluation 後才選定，不會改寫任何已發布指標。
- 正式 segmentation threshold 是 `0.50`；UI slider 只影響視覺化。
- 上傳影像只在 volatile application memory 中處理，本程式不會儲存檔案。
- v1 實驗的 filtered synthetic data 平均增益為負，本專案如實公開該結果。

## 授權邊界

- Space application code：MIT。
- ConvNeXt base model：Apache-2.0。
- SegFormer base 與衍生 segmenter checkpoints：NVIDIA Source Code License for
  SegFormer，**僅限 non-commercial research / evaluation**。
- VisA source dataset：CC BY 4.0。本 Space 不包含任何原始 VisA 影像。

完整聲明請見 `THIRD_PARTY_NOTICES.md` 與 `licenses/`。

本 Demo 不是 production AOI、品質放行、醫療或安全系統。
