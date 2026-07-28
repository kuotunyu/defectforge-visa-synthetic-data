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
  --df-canvas: #f3f7f6;
  --df-surface: #ffffff;
  --df-surface-soft: #eaf4f1;
  --df-surface-muted: #f7f9f9;
  --df-ink: #172b32;
  --df-muted: #52666b;
  --df-primary: #0f766e;
  --df-primary-strong: #0b5753;
  --df-primary-soft: #d9eeea;
  --df-accent: #d97706;
  --df-border: #cbdcd8;
  --df-border-strong: #9ebbb5;
  --df-danger: #b42318;
  --df-focus: #0b84a5;
  --df-radius-sm: 8px;
  --df-radius-md: 12px;
  --df-radius-lg: 16px;
  --df-font: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", Inter,
    system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
html, body, #root, .gradio-container {
  background: var(--df-canvas) !important;
  color: var(--df-ink) !important;
  font-family: var(--df-font) !important;
}
.gradio-container {
  --text-xs: .875rem;
  --text-sm: .9375rem;
  --text-md: 1rem;
  --text-lg: 1.125rem;
  max-width: 1280px !important;
  margin: 0 auto !important;
  min-height: 100vh !important;
  padding: 1.5rem 1.25rem 2.5rem !important;
  font-size: 1rem !important;
  line-height: 1.6 !important;
}
.gradio-container > .main,
.gradio-container > div:first-child {
  background: transparent !important;
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
  font-size: 1rem !important;
}
.gradio-container .prose p,
.gradio-container .prose li {
  line-height: 1.65 !important;
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
  padding: 1.75rem 2rem;
  background: var(--df-surface);
  border: 1px solid var(--df-border);
  border-radius: var(--df-radius-lg);
}
.df-brand-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.25rem;
}
.df-brand {
  display: inline-flex;
  align-items: center;
  gap: .75rem;
  color: var(--df-primary-strong);
  font-size: 1rem;
  font-weight: 750;
}
.df-mark {
  display: inline-grid;
  width: 40px;
  height: 40px;
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
  font-size: .875rem;
  font-weight: 700;
}
.df-runtime::before {
  content: "";
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--df-primary);
}
#df-header h1 {
  max-width: 24ch;
  margin: 0 0 .65rem;
  color: var(--df-ink);
  font-size: 2.25rem;
  font-weight: 800;
  line-height: 1.22;
  letter-spacing: -.025em;
  text-wrap: balance;
}
#df-header p {
  max-width: 68ch;
  margin: 0;
  color: var(--df-muted);
  font-size: 1.0625rem;
  line-height: 1.7;
  text-wrap: pretty;
}
#df-header strong {
  color: var(--df-primary-strong);
}
#df-steps {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  margin: 1rem 0 1.25rem;
  overflow: hidden;
  background: var(--df-surface);
  border: 1px solid var(--df-border);
  border-radius: var(--df-radius-md);
}
.df-step {
  display: grid;
  grid-template-columns: 38px 1fr;
  gap: .75rem;
  align-items: center;
  min-height: 76px;
  padding: .85rem 1rem;
}
.df-step + .df-step {
  border-left: 1px solid var(--df-border);
}
.df-step-number {
  display: grid;
  width: 38px;
  height: 38px;
  place-items: center;
  border-radius: 50%;
  background: var(--df-primary-soft);
  color: var(--df-primary-strong);
  font-weight: 800;
}
.df-step strong,
.df-step small {
  display: block;
}
.df-step strong {
  color: var(--df-ink);
  font-size: 1rem;
}
.df-step small {
  margin-top: .1rem;
  color: var(--df-muted);
  font-size: .875rem;
}
#df-workspace {
  gap: 1.25rem !important;
  align-items: flex-start !important;
}
.df-panel {
  padding: 1.25rem !important;
  background: var(--df-surface) !important;
  border: 1px solid var(--df-border) !important;
  border-radius: var(--df-radius-lg) !important;
}
.df-panel-heading {
  margin-bottom: 1rem;
}
.df-panel-heading h2 {
  margin: 0 0 .35rem;
  color: var(--df-ink);
  font-size: 1.375rem;
  font-weight: 800;
  line-height: 1.35;
}
.df-panel-heading p {
  margin: 0;
  color: var(--df-muted);
  font-size: .9375rem !important;
}
.df-panel-label {
  display: inline-block;
  margin-bottom: .35rem;
  color: var(--df-primary-strong);
  font-size: .875rem;
  font-weight: 800;
}
#df-object,
#df-upload,
#df-probabilities,
.df-output-image,
#df-advanced {
  border-color: var(--df-border) !important;
  border-radius: var(--df-radius-md) !important;
}
#df-object {
  background: var(--df-surface-muted) !important;
}
#df-object label,
#df-object span,
#df-upload label,
#df-probabilities label,
.df-output-image label {
  color: var(--df-ink) !important;
}
#df-object .wrap {
  gap: .6rem !important;
}
#df-object .wrap label {
  min-height: 44px !important;
  padding: .55rem .8rem !important;
  border-radius: var(--df-radius-sm) !important;
}
.df-object-help {
  margin: -.15rem 0 .85rem;
  color: var(--df-muted);
  font-size: .875rem;
}
.df-privacy {
  display: flex;
  gap: .55rem;
  align-items: flex-start;
  margin: .75rem 0 1rem;
  padding: .75rem .85rem;
  border-radius: var(--df-radius-sm);
  background: var(--df-surface-soft);
  color: var(--df-primary-strong);
  font-size: .875rem;
  line-height: 1.55;
}
.df-privacy strong {
  white-space: nowrap;
}
#df-upload {
  min-height: 390px !important;
  background: var(--df-surface-muted) !important;
}
#df-upload .upload-container,
#df-upload .wrap {
  min-height: 340px !important;
}
#df-upload .upload-container p,
#df-upload .upload-container span {
  color: var(--df-muted) !important;
  font-size: 1rem !important;
}
#df-advanced {
  margin-top: .85rem !important;
  background: var(--df-surface-muted) !important;
}
#df-advanced summary {
  min-height: 48px;
  color: var(--df-ink) !important;
  font-size: .9375rem !important;
  font-weight: 700;
}
#df-run,
#df-run button,
button#df-run {
  min-height: 52px !important;
  margin-top: .9rem !important;
  border: 0 !important;
  border-radius: var(--df-radius-sm) !important;
  background: var(--df-primary) !important;
  color: #fff !important;
  font-size: 1.0625rem !important;
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
  min-height: 132px;
  padding: 1rem 1.1rem !important;
  background: var(--df-surface-soft) !important;
  border: 1px solid var(--df-border) !important;
  border-radius: var(--df-radius-md) !important;
}
#df-summary h3 {
  margin: 0 0 .4rem !important;
  color: var(--df-primary-strong) !important;
  font-size: 1.1875rem !important;
  font-weight: 800 !important;
}
#df-summary p {
  margin: 0 !important;
  color: var(--df-muted) !important;
}
#df-summary code {
  padding: .1rem .35rem;
  border-radius: 4px;
  background: #fff;
  color: var(--df-primary-strong);
  font-size: .9375rem;
}
#df-probabilities {
  min-height: 230px !important;
  margin-top: .85rem !important;
  background: var(--df-surface-muted) !important;
}
#df-probabilities .label-wrap {
  background: var(--df-surface-muted) !important;
}
.df-section-heading {
  margin: 1.35rem 0 .8rem;
}
.df-section-heading h2 {
  margin: 0 0 .25rem;
  color: var(--df-ink);
  font-size: 1.375rem;
  font-weight: 800;
}
.df-section-heading p {
  margin: 0;
  color: var(--df-muted);
  font-size: .9375rem !important;
}
#df-localization {
  gap: 1.25rem !important;
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
  font-size: 1rem !important;
  font-weight: 750;
}
.df-boundary {
  margin-top: 1rem;
  padding: 1rem 1.1rem;
  border: 1px solid #ead3b0;
  border-radius: var(--df-radius-md);
  background: #fff8eb;
  color: #67410d;
  font-size: .9375rem;
  line-height: 1.65;
}
.df-boundary strong {
  color: #513108;
}
.df-footer {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: .65rem 1.25rem;
  margin-top: 1rem;
  padding: .85rem .15rem 0;
  color: var(--df-muted);
  font-size: .875rem;
}
.df-footer-links {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
}
footer { display: none !important; }
@media (max-width: 760px) {
  :root {
    --size-8: .5rem;
  }
  .gradio-container.gradio-container {
    padding: .75rem !important;
  }
  .main.fillable {
    padding: 0 !important;
  }
  #df-header {
    padding: 1.25rem;
  }
  .df-brand-row {
    align-items: flex-start;
    flex-direction: column;
  }
  #df-header h1 {
    font-size: 1.875rem;
  }
  #df-steps {
    grid-template-columns: 1fr;
  }
  .df-step {
    min-height: 68px;
  }
  .df-step + .df-step {
    border-top: 1px solid var(--df-border);
    border-left: 0;
  }
  .df-panel {
    padding: 1rem !important;
  }
  #df-upload {
    min-height: 320px !important;
  }
  #df-upload .upload-container,
  #df-upload .wrap {
    min-height: 275px !important;
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
        gr.Label(visible=False),
        None,
        None,
        (
            "### 尚未開始檢測\n"
            "請完成左側三個步驟。檢測後會顯示 classification confidence、"
            "binary mask 與 probability heatmap。"
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
            "結果。整個流程只需要三個步驟。</p>"
            "</section>"
        )
        gr.HTML(
            "<nav id='df-steps' aria-label='檢測流程' lang='zh-Hant-TW'>"
            "<div class='df-step'><span class='df-step-number'>1</span><span>"
            "<strong>選擇物件</strong><small>選擇 pcb1 或 capsules</small></span></div>"
            "<div class='df-step'><span class='df-step-number'>2</span><span>"
            "<strong>上傳影像</strong><small>支援拖放、貼上或點擊上傳</small></span></div>"
            "<div class='df-step'><span class='df-step-number'>3</span><span>"
            "<strong>開始檢測</strong><small>查看分類、mask 與 heatmap</small></span></div>"
            "</nav>"
        )
        with gr.Row(equal_height=False, elem_id="df-workspace"):
            with gr.Column(scale=5, min_width=340, elem_classes=["df-panel"]):
                gr.HTML(
                    "<div class='df-panel-heading' lang='zh-Hant-TW'>"
                    "<span class='df-panel-label'>檢測輸入</span>"
                    "<h2>準備一張待檢影像</h2>"
                    "<p>物件類型必須與影像內容相符，接著上傳影像即可。</p>"
                    "</div>"
                )
                object_name = gr.Radio(
                    choices=["pcb1", "capsules"],
                    value="pcb1",
                    label="1. 選擇物件類型",
                    info="模型是 object-specific，請勿跨物件使用。",
                    elem_id="df-object",
                )
                gr.HTML(
                    "<p class='df-object-help' lang='zh-Hant-TW'>"
                    "<strong>pcb1</strong>：印刷電路板　·　"
                    "<strong>capsules</strong>：膠囊</p>"
                )
                image = gr.Image(
                    label="2. 上傳待檢影像",
                    type="pil",
                    sources=["upload", "clipboard"],
                    image_mode="RGB",
                    buttons=["fullscreen"],
                    height=420,
                    elem_id="df-upload",
                )
                gr.HTML(
                    "<div class='df-privacy' lang='zh-Hant-TW'>"
                    "<strong>隱私說明</strong><span>影像只在記憶體中處理；"
                    "本程式不會儲存你上傳的檔案。</span></div>"
                )
                with gr.Accordion(
                    "進階顯示設定（通常不需調整）",
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
                run = gr.Button("3. 開始檢測", variant="primary", elem_id="df-run")
            with gr.Column(scale=4, min_width=320, elem_classes=["df-panel"]):
                gr.HTML(
                    "<div class='df-panel-heading' lang='zh-Hant-TW'>"
                    "<span class='df-panel-label'>檢測結果</span>"
                    "<h2>先看結果摘要</h2>"
                    "<p>完成檢測後，這裡會顯示判定、信心分數與執行時間。</p>"
                    "</div>"
                )
                summary = gr.Markdown(
                    "### 尚未開始檢測\n"
                    "請完成左側三個步驟。檢測後會顯示 classification confidence、"
                    "binary mask 與 probability heatmap。",
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
