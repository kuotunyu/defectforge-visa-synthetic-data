"""Public DefectForge classification and segmentation demo."""

from __future__ import annotations

import os

import gradio as gr
from runtime import MODEL_ROOT, SpaceContractError, load_manifest, predict, preferred_device

GITHUB_URL = "https://github.com/kuotunyu/01-defectforge-visa"
DATASET_URL = "https://huggingface.co/datasets/steven0226/defectforge-visa-synthetic"

CSS = """
:root {
  color-scheme: light;
  --df-canvas: oklch(97% .009 184);
  --df-surface: oklch(100% 0 0);
  --df-surface-soft: oklch(94% .024 181);
  --df-surface-muted: oklch(98% .007 184);
  --df-result-soft: oklch(96% .018 181);
  --df-ink: oklch(27% .032 208);
  --df-muted: oklch(42% .027 197);
  --df-primary: oklch(45% .088 179);
  --df-primary-strong: oklch(34% .068 184);
  --df-primary-soft: oklch(91% .036 179);
  --df-accent: oklch(61% .122 65);
  --df-border: oklch(84% .023 184);
  --df-border-strong: oklch(70% .037 183);
  --df-danger: oklch(48% .17 28);
  --df-focus: oklch(55% .12 221);
  --df-radius-sm: 8px;
  --df-radius-md: 12px;
  --df-radius-lg: 16px;
  --df-space-xs: .5rem;
  --df-space-sm: .75rem;
  --df-space-md: 1rem;
  --df-space-lg: 1.5rem;
  --df-space-xl: 2rem;
  --df-font: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", Inter,
    system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
html, body, #root, .gradio-container {
  background: var(--df-canvas) !important;
  color: var(--df-ink) !important;
  font-family: var(--df-font) !important;
}
body.dark,
body.dark #root,
body.dark .gradio-container {
  background: var(--df-canvas) !important;
  color: var(--df-ink) !important;
}
.gradio-container {
  --text-xs: 1rem;
  --text-sm: 1rem;
  --text-md: 1.125rem;
  --text-lg: 1.25rem;
  --body-background-fill: var(--df-canvas);
  --body-text-color: var(--df-ink);
  --background-fill-primary: var(--df-surface);
  --background-fill-secondary: var(--df-surface-muted);
  --block-background-fill: var(--df-surface);
  --block-label-background-fill: var(--df-surface);
  --block-label-text-color: var(--df-ink);
  --input-background-fill: var(--df-surface);
  --border-color-primary: var(--df-border);
  --border-color-accent: var(--df-border-strong);
  max-width: 1400px !important;
  box-sizing: border-box !important;
  width: 100% !important;
  margin: 0 auto !important;
  min-height: 100vh !important;
  padding: 1rem 1.25rem 2rem !important;
  font-size: 1.125rem !important;
  line-height: 1.6 !important;
}
.gradio-container > .main,
.gradio-container > div:first-child {
  background: transparent !important;
}
.html-container:has(#df-header),
.html-container:has(.df-panel-heading),
.html-container:has(.df-privacy),
.html-container:has(.df-section-heading),
.html-container:has(.df-boundary) {
  padding: 0 !important;
}
.gradio-container,
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  font-family: var(--df-font) !important;
}
.gradio-container p,
.gradio-container label,
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container .prose {
  font-size: 1.125rem !important;
}
.gradio-container .prose p,
.gradio-container .prose li {
  line-height: 1.6 !important;
}
.gradio-container .info,
.gradio-container .secondary-wrap,
.gradio-container .secondary-wrap *,
.gradio-container .label-wrap,
.gradio-container .label-wrap * {
  font-size: 1rem !important;
  line-height: 1.5 !important;
}
.gradio-container a {
  color: var(--df-primary-strong) !important;
  text-underline-offset: 3px;
}
.gradio-container button:focus-visible,
.gradio-container input:focus-visible,
.gradio-container textarea:focus-visible,
.gradio-container [role="radio"]:focus-visible,
.gradio-container summary:focus-visible {
  outline: 3px solid color-mix(in srgb, var(--df-focus) 55%, transparent) !important;
  outline-offset: 2px !important;
}
#df-header {
  padding: 1.25rem 1.5rem;
  background: var(--df-surface);
  border: 1px solid var(--df-border);
  border-radius: var(--df-radius-lg);
}
.df-brand-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: .75rem;
}
.df-brand {
  display: inline-flex;
  align-items: center;
  gap: .65rem;
  color: var(--df-primary-strong);
  font-size: 1.0625rem;
  font-weight: 750;
}
.df-brand > span:last-child {
  color: var(--df-primary-strong) !important;
}
.df-mark {
  display: inline-grid;
  width: 38px;
  height: 38px;
  place-items: center;
  border-radius: var(--df-radius-sm);
  background: var(--df-primary);
  color: #fff;
  font-weight: 800;
  letter-spacing: -.02em;
}
.df-runtime {
  display: inline-flex;
  align-items: center;
  gap: .5rem;
  min-height: 36px;
  padding: .35rem .75rem;
  border-radius: 999px;
  background: var(--df-primary-soft);
  color: var(--df-primary-strong);
  font-size: 1rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
.df-runtime::before {
  content: "";
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--df-primary);
}
#df-header h1 {
  max-width: 28ch;
  margin: 0 0 .35rem;
  color: var(--df-ink);
  font-size: 2.25rem;
  font-weight: 800;
  line-height: 1.2;
  letter-spacing: -.025em;
  text-wrap: balance;
}
#df-header p {
  max-width: 100ch;
  margin: 0;
  color: var(--df-muted);
  font-size: 1.125rem;
  line-height: 1.55;
  text-wrap: pretty;
}
#df-header strong {
  color: var(--df-primary-strong);
}
.df-flow {
  display: flex;
  align-items: center;
  gap: .8rem;
  margin-top: 1rem;
  padding-top: .85rem;
  border-top: 1px solid var(--df-border);
}
.df-flow-label {
  flex: 0 0 auto;
  color: var(--df-primary-strong);
  font-size: 1rem;
  font-weight: 800;
}
.df-flow ol {
  display: flex;
  flex: 1;
  flex-wrap: wrap;
  align-items: center;
  gap: .5rem .7rem;
  margin: 0;
  padding: 0;
  list-style: none;
}
.df-flow li {
  display: inline-flex;
  align-items: center;
  gap: .45rem;
  color: var(--df-ink);
  font-size: 1rem;
  font-weight: 700;
}
.df-flow li span {
  color: var(--df-ink) !important;
}
.df-flow li + li::before {
  content: "→";
  margin-right: .2rem;
  color: var(--df-border-strong);
  font-weight: 800;
}
.df-flow b {
  display: inline-grid;
  width: 28px;
  height: 28px;
  place-items: center;
  border-radius: 50%;
  background: var(--df-primary-soft);
  color: var(--df-primary-strong);
  font-size: 1rem;
}
#df-workspace {
  gap: var(--df-space-md) !important;
  align-items: stretch !important;
  margin-top: 0 !important;
}
.df-panel {
  min-width: 0 !important;
  padding: 1.25rem 1.35rem !important;
  background: var(--df-surface) !important;
  border: 1px solid var(--df-border) !important;
  border-radius: var(--df-radius-lg) !important;
}
#df-result-panel {
  background: var(--df-result-soft) !important;
  border-color: var(--df-border-strong) !important;
}
.df-panel-heading {
  margin-bottom: .85rem;
}
.df-panel-heading h2 {
  margin: 0 0 .2rem;
  color: var(--df-ink);
  font-size: 1.5rem;
  font-weight: 800;
  line-height: 1.3;
}
.df-panel-heading p {
  margin: 0;
  color: var(--df-muted);
  font-size: 1.0625rem !important;
  line-height: 1.55 !important;
}
#df-object,
#df-upload,
#df-probabilities,
.df-output-image,
#df-advanced {
  border-color: var(--df-border) !important;
  border-radius: var(--df-radius-md) !important;
}
#df-input-grid {
  gap: var(--df-space-md) !important;
  align-items: stretch !important;
}
.df-setup {
  gap: .75rem !important;
  min-width: 270px !important;
  padding: 1rem !important;
  background: var(--df-surface-muted) !important;
  border: 1px solid var(--df-border) !important;
  border-radius: var(--df-radius-md) !important;
}
#df-object {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
#df-object > .wrap,
#df-advanced > .wrap {
  padding: 0 !important;
  background: transparent !important;
  border: 0 !important;
}
#df-object label,
#df-object span,
#df-upload label,
#df-probabilities label,
.df-output-image label {
  color: var(--df-ink) !important;
}
#df-object > .wrap > span {
  color: var(--df-ink) !important;
  font-size: 1.125rem !important;
  font-weight: 700 !important;
}
#df-object .info-text {
  color: var(--df-muted) !important;
  font-size: 1rem !important;
  line-height: 1.5 !important;
}
#df-object .wrap {
  gap: .65rem !important;
}
#df-object .wrap label {
  min-height: 48px !important;
  padding: .65rem .8rem !important;
  border-radius: var(--df-radius-sm) !important;
  background: oklch(93% .013 190) !important;
  border: 1px solid transparent !important;
}
#df-object .wrap label:has(input:checked) {
  background: var(--df-primary-soft) !important;
  border-color: var(--df-primary) !important;
  color: var(--df-primary-strong) !important;
}
.df-object-help {
  margin: 0;
  color: var(--df-muted);
  font-size: 1rem;
}
.df-privacy {
  display: flex;
  gap: .65rem;
  align-items: center;
  margin: .75rem 0 0;
  padding: .75rem .9rem;
  border-radius: var(--df-radius-sm);
  background: var(--df-surface-soft);
  color: var(--df-primary-strong);
  font-size: 1rem;
  line-height: 1.5;
}
.df-privacy strong {
  white-space: nowrap;
}
.df-privacy strong,
.df-privacy span {
  color: var(--df-primary-strong) !important;
}
#df-upload {
  min-height: 320px !important;
  background: oklch(98% .01 184) !important;
  border-style: dashed !important;
  border-color: var(--df-border-strong) !important;
}
#df-upload .upload-container,
#df-upload .wrap {
  min-height: 275px !important;
}
#df-upload .upload-container p,
#df-upload .upload-container span {
  color: var(--df-muted) !important;
  font-size: 1.125rem !important;
}
#df-advanced {
  margin-top: auto !important;
  background: var(--df-surface-muted) !important;
}
#df-advanced summary {
  min-height: 48px;
  color: var(--df-ink) !important;
  font-size: 1rem !important;
  font-weight: 700;
}
#df-advanced .label-wrap,
#df-advanced .label-wrap * {
  color: var(--df-ink) !important;
  font-size: 1rem !important;
}
#df-upload .label-wrap,
#df-upload .label-wrap *,
.df-output-image .label-wrap,
.df-output-image .label-wrap *,
#df-probabilities .label-wrap,
#df-probabilities .label-wrap * {
  background: var(--df-surface) !important;
  color: var(--df-ink) !important;
}
#df-run,
#df-run button,
button#df-run {
  min-height: 56px !important;
  margin-top: .75rem !important;
  border: 0 !important;
  border-radius: var(--df-radius-sm) !important;
  background: var(--df-primary) !important;
  color: #fff !important;
  font-size: 1.1875rem !important;
  font-weight: 800 !important;
  transition:
    background-color 180ms cubic-bezier(.16, 1, .3, 1),
    transform 180ms cubic-bezier(.16, 1, .3, 1) !important;
}
#df-run:hover,
#df-run button:hover,
button#df-run:hover {
  background: var(--df-primary-strong) !important;
  transform: translateY(-1px);
}
#df-run:active,
#df-run button:active,
button#df-run:active {
  transform: translateY(0);
}
#df-summary {
  min-height: 116px;
  padding: .9rem 1rem !important;
  background: var(--df-surface) !important;
  border: 1px solid var(--df-border-strong) !important;
  border-radius: var(--df-radius-sm) !important;
}
#df-summary h3 {
  margin: 0 0 .25rem !important;
  color: var(--df-primary-strong) !important;
  font-size: 1.375rem !important;
  font-weight: 800 !important;
}
#df-summary p {
  margin: 0 !important;
  color: var(--df-muted) !important;
  font-size: 1.0625rem !important;
}
#df-summary code {
  padding: .1rem .35rem;
  border-radius: 4px;
  background: var(--df-surface-soft);
  color: var(--df-primary-strong);
  font-size: 1rem;
}
#df-probabilities {
  min-height: 210px !important;
  margin-top: .75rem !important;
  background: var(--df-surface) !important;
}
#df-probabilities .label-wrap {
  background: var(--df-surface) !important;
}
.df-section-heading {
  margin: 1.15rem 0 .65rem;
}
.df-section-heading h2 {
  margin: 0 0 .15rem;
  color: var(--df-ink);
  font-size: 1.5rem;
  font-weight: 800;
}
.df-section-heading p {
  margin: 0;
  color: var(--df-muted);
  font-size: 1.0625rem !important;
}
#df-localization {
  gap: var(--df-space-md) !important;
}
.df-output-image {
  overflow: hidden !important;
  background: var(--df-surface) !important;
  border: 1px solid var(--df-border) !important;
}
.df-output-image .label-wrap {
  background: var(--df-surface) !important;
}
#df-evidence {
  margin-top: 1rem !important;
  border: 1px solid var(--df-border) !important;
  border-radius: var(--df-radius-md) !important;
  background: var(--df-surface) !important;
}
#df-evidence summary {
  min-height: 52px;
  color: var(--df-ink) !important;
  font-size: 1.0625rem !important;
  font-weight: 750;
}
#df-evidence .label-wrap,
#df-evidence .label-wrap * {
  color: var(--df-ink) !important;
  font-size: 1.0625rem !important;
}
.df-boundary {
  margin-top: 1rem;
  padding: 1rem 1.1rem;
  border: 1px solid oklch(83% .058 69);
  border-radius: var(--df-radius-md);
  background: oklch(97% .026 75);
  color: oklch(37% .07 65);
  font-size: 1rem;
  line-height: 1.6;
}
.df-boundary strong {
  color: oklch(31% .07 65);
}
.df-footer {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: .65rem 1.25rem;
  margin-top: 1rem;
  padding: .85rem .15rem 0;
  color: var(--df-muted);
  font-size: 1rem;
}
.df-footer-links {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
}
footer { display: none !important; }
@media (max-width: 900px) {
  #df-workspace,
  #df-input-grid {
    align-items: stretch !important;
    flex-direction: column !important;
  }
  #df-workspace > *,
  #df-input-grid > * {
    flex: 1 1 auto !important;
    width: 100% !important;
    min-width: 0 !important;
  }
}
@media (max-width: 760px) {
  :root {
    --size-8: .5rem;
  }
  .gradio-container.gradio-container {
    max-width: 100% !important;
    padding: .75rem !important;
  }
  .main.fillable {
    padding: 0 !important;
  }
  #df-header {
    padding: 1rem;
  }
  .df-brand-row {
    align-items: flex-start;
    flex-direction: column;
  }
  #df-header h1 {
    font-size: 2rem;
  }
  .df-flow {
    align-items: flex-start;
    flex-direction: column;
  }
  .df-flow ol {
    align-items: flex-start;
    flex-direction: column;
    gap: .4rem;
  }
  .df-flow li + li::before {
    content: "";
    margin: 0;
  }
  .df-panel {
    padding: 1rem !important;
  }
  #df-upload {
    min-height: 290px !important;
  }
  #df-upload .upload-container,
  #df-upload .wrap {
    min-height: 245px !important;
  }
  .df-privacy {
    align-items: flex-start;
    flex-direction: column;
  }
}
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    scroll-behavior: auto !important;
    transition-duration: .01ms !important;
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
  }
}
"""


def _run(image: object, object_name: str, threshold: float) -> tuple[object, ...]:
    try:
        probabilities, mask, heatmap, summary, evidence = predict(
            image,
            object_name,
            threshold,
        )
        return (
            gr.Column(visible=True),
            gr.Label(
                value=probabilities,
                visible=True,
                label="Classification confidence（分類信心）",
                num_top_classes=2,
            ),
            mask,
            heatmap,
            summary,
            evidence,
            gr.Column(visible=True),
            gr.Accordion(
                "查看模型證據與 checkpoint provenance",
                visible=True,
            ),
        )
    except SpaceContractError as error:
        raise gr.Error(str(error), duration=8) from error


def _clear_results() -> tuple[object, ...]:
    return (
        gr.Column(visible=False),
        gr.Label(visible=False),
        None,
        None,
        (
            "### 等待檢測\n"
            "重新按下「開始檢測」後，結果會顯示在這裡。"
        ),
        {},
        gr.Column(visible=False),
        gr.Accordion(visible=False),
    )


def build_app() -> gr.Blocks:
    manifest = load_manifest()
    device = preferred_device().type.upper()
    with gr.Blocks(
        title="DefectForge · VisA 瑕疵影像檢測",
        analytics_enabled=False,
        fill_width=True,
    ) as demo:
        gr.HTML(
            "<section id='df-header' lang='zh-Hant-TW'>"
            "<div class='df-brand-row'>"
            "<div class='df-brand'><span class='df-mark'>DF</span>"
            "<span>DefectForge · VisA</span></div>"
            f"<div class='df-runtime'>模型已就緒 · {device} · "
            f"{manifest['source_commit'][:10]}</div>"
            "</div>"
            "<h1>瑕疵影像檢測 Demo</h1>"
            "<p>上傳一張 <strong>pcb1</strong> 或 <strong>capsules</strong> 影像，"
            "即可同時取得 anomaly classification 與 defect-region segmentation "
            "結果。</p>"
            "<nav class='df-flow' aria-label='檢測流程'>"
            "<span class='df-flow-label'>操作流程</span><ol>"
            "<li><b>1</b><span>選擇物件</span></li>"
            "<li><b>2</b><span>上傳影像</span></li>"
            "<li><b>3</b><span>開始檢測</span></li>"
            "</ol></nav>"
            "</section>"
        )
        with gr.Row(equal_height=False, elem_id="df-workspace"):
            with gr.Column(
                scale=5,
                min_width=420,
                elem_id="df-input-panel",
                elem_classes=["df-panel"],
            ):
                gr.HTML(
                    "<div class='df-panel-heading' lang='zh-Hant-TW'>"
                    "<h2>選擇物件並上傳影像</h2>"
                    "<p>完成後按「開始檢測」，分類與瑕疵位置會一次顯示。</p>"
                    "</div>"
                )
                with gr.Row(equal_height=True, elem_id="df-input-grid"):
                    with gr.Column(scale=4, min_width=270, elem_classes=["df-setup"]):
                        object_name = gr.Radio(
                            choices=[
                                ("pcb1｜印刷電路板", "pcb1"),
                                ("capsules｜膠囊", "capsules"),
                            ],
                            value="pcb1",
                            label="選擇與影像相符的物件",
                            info="模型為 object-specific；選錯物件會讓結果失真。",
                            elem_id="df-object",
                        )
                        gr.HTML(
                            "<p class='df-object-help' lang='zh-Hant-TW'>"
                            "支援 JPEG、PNG；單張上限 20 MB。</p>"
                        )
                        with gr.Accordion(
                            "進階顯示設定",
                            open=False,
                            elem_id="df-advanced",
                        ):
                            threshold = gr.Slider(
                                minimum=0.05,
                                maximum=0.95,
                                value=0.50,
                                step=0.05,
                                label="Mask 顯示 threshold",
                                info="正式 preregistered threshold 為 0.50。此設定只改變畫面上的 "
                                "binary mask，不會改寫已發布指標。",
                            )
                    with gr.Column(scale=7, min_width=340):
                        image = gr.Image(
                            label="上傳影像",
                            type="pil",
                            sources=["upload", "clipboard"],
                            image_mode="RGB",
                            placeholder="將圖片拖放到這裡，或點擊上傳",
                            buttons=["fullscreen"],
                            height=330,
                            elem_id="df-upload",
                        )
                gr.HTML(
                    "<div class='df-privacy' lang='zh-Hant-TW'>"
                    "<strong>隱私說明</strong><span>影像只在記憶體中處理；"
                    "本程式不會儲存你上傳的檔案。</span></div>"
                )
                run = gr.Button("開始檢測", variant="primary", elem_id="df-run")
            with gr.Column(
                scale=4,
                min_width=340,
                visible=False,
                elem_id="df-result-panel",
                elem_classes=["df-panel"],
            ) as result_panel:
                gr.HTML(
                    "<div class='df-panel-heading' lang='zh-Hant-TW'>"
                    "<h2>檢測結果</h2>"
                    "</div>"
                )
                summary = gr.Markdown(
                    "### 等待檢測\n"
                    "按下「開始檢測」後，結果會顯示在這裡。",
                    elem_id="df-summary",
                )
                probabilities = gr.Label(
                    label="Classification confidence（分類信心）",
                    num_top_classes=2,
                    elem_id="df-probabilities",
                    visible=False,
                )
        with gr.Column(visible=False) as output_details:
            gr.HTML(
                "<div class='df-section-heading' lang='zh-Hant-TW'>"
                "<h2>瑕疵位置視覺化</h2>"
                "<p>Binary mask 顯示超過 threshold 的區域；"
                "heatmap 保留完整機率分布。</p>"
                "</div>"
            )
            with gr.Row(elem_id="df-localization"):
                mask = gr.Image(
                    label="Binary defect mask（二值瑕疵遮罩）",
                    image_mode="L",
                    buttons=["download", "fullscreen"],
                    elem_classes=["df-output-image"],
                )
                heatmap = gr.Image(
                    label="Defect probability heatmap（瑕疵機率熱圖）",
                    buttons=["download", "fullscreen"],
                    elem_classes=["df-output-image"],
                )
        with gr.Accordion(
            "查看模型證據與 checkpoint provenance",
            open=False,
            visible=False,
            elem_id="df-evidence",
        ) as evidence_panel:
            evidence = gr.JSON(
                value={},
                label="Immutable evidence（不可變證據）",
                open=True,
            )
        gr.HTML(
            "<aside class='df-boundary' lang='zh-Hant-TW'>"
            "<strong>研究用途聲明：</strong>這是公開 research / evaluation Demo，"
            "不是 production AOI、品質放行或安全系統。SegFormer 衍生 checkpoint "
            "僅限 non-commercial research / evaluation。v1 實驗的 synthetic-data "
            "平均增益為負，本專案如實保留該結果。"
            "</aside>"
            "<div class='df-footer' lang='zh-Hant-TW'>"
            "<span>DefectForge · Verified research artifact</span>"
            "<span class='df-footer-links'>"
            f"<a href='{GITHUB_URL}' target='_blank' rel='noopener'>GitHub 原始碼</a>"
            f"<a href='{DATASET_URL}' target='_blank' rel='noopener'>Hugging Face Dataset</a>"
            "</span></div>"
        )
        run.click(
            fn=_run,
            inputs=[image, object_name, threshold],
            outputs=[
                result_panel,
                probabilities,
                mask,
                heatmap,
                summary,
                evidence,
                output_details,
                evidence_panel,
            ],
            concurrency_limit=1,
            concurrency_id="model-inference",
            show_progress="full",
            api_visibility="private",
            scroll_to_output=True,
        )
        for component in (image, object_name):
            component.change(
                fn=_clear_results,
                inputs=None,
                outputs=[
                    result_panel,
                    probabilities,
                    mask,
                    heatmap,
                    summary,
                    evidence,
                    output_details,
                    evidence_panel,
                ],
                show_progress="hidden",
                api_visibility="private",
                trigger_mode="always_last",
            )
    return demo


demo = build_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.queue(max_size=8, default_concurrency_limit=1, api_open=False).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=False,
        max_threads=4,
        max_file_size="20mb",
        enable_monitoring=False,
        blocked_paths=[str(MODEL_ROOT)],
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.emerald,
            secondary_hue=gr.themes.colors.amber,
            neutral_hue=gr.themes.colors.slate,
        ),
        css=CSS,
    )
