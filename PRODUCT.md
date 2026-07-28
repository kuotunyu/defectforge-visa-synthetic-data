# Product

## Register

product

## Users

主要使用者是第一次打開公開 Demo 的作品集審閱者、電腦視覺研究者與工程師。他們
多半在一般桌機或筆電螢幕上操作，未必讀過專案文件，也不應先理解模型架構才能完成
任務。他們要做的工作是選擇與圖片相符的物件類型、上傳一張檢測影像，快速判讀分類
信心、瑕疵位置與模型證據。

## Product Purpose

DefectForge Demo 讓使用者以一條清楚的操作流程，親自執行 `pcb1` 或 `capsules` 的
anomaly classification 與 defect-region segmentation。成功的介面必須在五秒內說明
「這是什麼、要上傳什麼、按哪裡、結果在哪裡」，同時維持正式 checkpoint、研究限制
與不可變證據鏈的可信度。

## Brand Personality

清楚、可信、沉著。介面像一張整理良好的現代檢驗台：專業但不故作艱深，精準但不
冰冷，以正體中文引導操作，必要的模型與研究專有名詞保留原文。

## Anti-references

- 不要像需要工程背景才看得懂的 terminal、監控台或 cyberpunk dashboard。
- 不要以超大英文標題壓過任務，也不要用大量過小、全大寫、寬字距文字製造科技感。
- 不要讓使用者面對空白輸出而不知道下一步；empty state 必須教會操作。
- 不要用裝飾性格線、霓虹光、厚重陰影、彩色側邊條或過度卡片化掩蓋資訊層級。
- 不要把研究證據、授權與限制刪除；它們應採 progressive disclosure，而非搶占主流程。

## Design Principles

1. **先教會，再要求操作。** 每個初始狀態都要回答下一步，不依賴外部說明文件。
2. **任務永遠比品牌更醒目。** 產品名稱建立辨識度，主要視覺焦點仍是上傳與檢測。
3. **中文負責理解，原文負責精確。** 正體中文是介面主語言，模型、指標與物件名稱
   保留原文並在需要時補充中文。
4. **結果先摘要，證據後展開。** 先顯示可判讀結果，再讓專業使用者查看 checkpoint
   provenance 與 immutable evidence。
5. **研究誠信可見但不製造焦慮。** 清楚標示非 production AOI 與 non-commercial
   research boundary，不以警告牆阻擋主要任務。

## Accessibility & Inclusion

以 WCAG 2.2 AA 為最低標準。一般內文與控制項以 20px 為基準，輔助文字不得小於
18px；保留清楚的 keyboard focus、足夠對比、44px 以上操作目標與 reduced-motion
支援。狀態不可只靠顏色表達，手機與放大至 200% 時仍需維持可理解的閱讀順序。
