"""Public DefectForge classification and segmentation demo."""

from __future__ import annotations

import os

import gradio as gr
from runtime import MODEL_ROOT, SpaceContractError, load_manifest, predict, preferred_device

GITHUB_URL = "https://github.com/kuotunyu/01-defectforge-visa"
DATASET_URL = "https://huggingface.co/datasets/steven0226/defectforge-visa-synthetic"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root {
  --df-night: #07131a;
  --df-plate: #0d222b;
  --df-plate-2: #112d37;
  --df-ink: #e8f1ed;
  --df-muted: #92aaa8;
  --df-mint: #46d7af;
  --df-amber: #f7bd3f;
  --df-danger: #ef533b;
  --df-rule: rgba(70, 215, 175, 0.24);
}
html, body, #root, .gradio-container {
  background:
    linear-gradient(rgba(70, 215, 175, .035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(70, 215, 175, .035) 1px, transparent 1px),
    radial-gradient(circle at 91% -10%, rgba(247, 189, 63, .15), transparent 32%),
    var(--df-night) !important;
  background-size: 32px 32px, 32px 32px, auto, auto !important;
  color: var(--df-ink) !important;
  font-family: "IBM Plex Mono", monospace !important;
}
.gradio-container {
  max-width: 1480px !important; margin: 0 auto !important;
  min-height: 100vh !important;
}
.gradio-container > .main,
.gradio-container > div:first-child {
  background: transparent !important;
}
#df-hero {
  position: relative; overflow: hidden; padding: 2.2rem 2.4rem 2rem;
  border: 1px solid var(--df-rule); border-left: 7px solid var(--df-amber);
  background: linear-gradient(120deg, rgba(13,34,43,.96), rgba(7,19,26,.88));
  box-shadow: 12px 12px 0 rgba(0,0,0,.22);
}
#df-hero::after {
  content: "DF // 01"; position: absolute; right: 1.2rem; top: .7rem;
  color: rgba(70,215,175,.13); font: 700 clamp(3rem, 9vw, 8rem)/1 "Chakra Petch";
  letter-spacing: -.08em; pointer-events: none;
}
#df-hero h1 {
  position: relative; z-index: 1; margin: .2rem 0 .65rem; color: var(--df-ink);
  font: 700 clamp(2.4rem, 6vw, 5.8rem)/.88 "Chakra Petch", sans-serif;
  letter-spacing: -.055em; text-transform: uppercase; max-width: 960px;
}
#df-hero p { position: relative; z-index: 1; color: var(--df-muted); max-width: 820px; }
#df-hero strong { color: var(--df-amber) !important; }
.df-kicker {
  color: var(--df-mint) !important; letter-spacing: .23em;
  text-transform: uppercase; font-size: .75rem;
}
.df-status {
  display: inline-flex; gap: .55rem; align-items: center; margin-top: 1rem;
  padding: .42rem .72rem; border: 1px solid var(--df-rule);
  color: var(--df-mint) !important;
  background: rgba(70,215,175,.06); text-transform: uppercase; letter-spacing: .11em;
  font-size: .72rem;
}
.df-status::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--df-mint); box-shadow: 0 0 16px var(--df-mint); }
.df-panel, .df-output {
  border: 1px solid var(--df-rule) !important; border-radius: 0 !important;
  background: rgba(13, 34, 43, .9) !important; box-shadow: 8px 8px 0 rgba(0,0,0,.18) !important;
}
.df-panel { padding: 1rem !important; }
.df-output { border-top: 3px solid var(--df-mint) !important; }
.df-output, .df-output * {
  color: var(--df-ink) !important;
}
.df-output > label.float,
.df-output .label-wrap {
  background: var(--df-plate) !important; border-color: var(--df-rule) !important;
}
.df-output code {
  background: rgba(70, 215, 175, .12) !important;
  color: var(--df-mint) !important;
}
.df-run button, button.df-run {
  min-height: 52px !important; border-radius: 0 !important; border: 0 !important;
  background: var(--df-amber) !important; color: #1b210f !important;
  font: 700 .9rem "Chakra Petch", sans-serif !important; letter-spacing: .12em;
  text-transform: uppercase; box-shadow: 5px 5px 0 #74561d !important;
  transition: transform .14s ease, box-shadow .14s ease !important;
}
.df-run button:hover, button.df-run:hover { transform: translate(2px,2px); box-shadow: 3px 3px 0 #74561d !important; }
.df-section {
  margin: .2rem 0 .9rem !important; padding: .5rem .75rem !important;
  border-left: 4px solid var(--df-mint) !important;
  background: var(--df-plate) !important;
}
.df-section h2, .df-section h3 {
  margin: 0 !important; color: var(--df-ink) !important;
  font-family: "Chakra Petch", sans-serif !important;
  text-transform: uppercase; letter-spacing: .04em;
}
.df-note {
  border-left: 3px solid var(--df-danger); padding-left: 1rem;
  color: var(--df-muted) !important;
}
.df-note p, .df-note h3 { color: var(--df-muted) !important; }
.df-note strong { color: var(--df-ink) !important; }
.df-note a { color: var(--df-mint) !important; }
footer { display: none !important; }
"""


def _run(image: object, object_name: str, threshold: float) -> tuple[object, ...]:
    try:
        return predict(image, object_name, threshold)
    except SpaceContractError as error:
        raise gr.Error(str(error), duration=8) from error


def build_app() -> gr.Blocks:
    manifest = load_manifest()
    device = preferred_device().type.upper()
    with gr.Blocks(
        title="DefectForge · VisA Inspection Console",
        analytics_enabled=False,
        fill_width=True,
    ) as demo:
        gr.HTML(
            "<section id='df-hero'>"
            "<div class='df-kicker'>Machine vision / verified research artifact</div>"
            "<h1>DefectForge<br>Inspection Console</h1>"
            "<p>Paired anomaly classification and defect-region segmentation for "
            "VisA <strong>pcb1</strong> and <strong>capsules</strong>. Every checkpoint "
            "is hash-bound to a formal training report.</p>"
            f"<div class='df-status'>{device} runtime · source "
            f"{manifest['source_commit'][:10]}</div>"
            "</section>"
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes=["df-panel"]):
                gr.Markdown("## 01 / Configure", elem_classes=["df-section"])
                object_name = gr.Radio(
                    choices=["pcb1", "capsules"],
                    value="pcb1",
                    label="Object-specific model",
                    info="Choose the object family before uploading an inspection frame.",
                )
                image = gr.Image(
                    label="Inspection frame",
                    type="pil",
                    sources=["upload", "clipboard"],
                    image_mode="RGB",
                    buttons=["fullscreen"],
                    height=420,
                )
                threshold = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=0.50,
                    step=0.05,
                    label="Visualization threshold",
                    info="The formal preregistered threshold is 0.50. Moving this slider "
                    "changes only the displayed mask, never reported metrics.",
                )
                run = gr.Button("Run verified inspection", elem_classes=["df-run"])
            with gr.Column(scale=4):
                gr.Markdown("## 02 / Readout", elem_classes=["df-section"])
                probabilities = gr.Label(
                    label="Classification confidence",
                    num_top_classes=2,
                    elem_classes=["df-output"],
                )
                summary = gr.Markdown(
                    "Awaiting an inspection frame.",
                    elem_classes=["df-output"],
                )
                with gr.Accordion("Evidence and checkpoint provenance", open=False):
                    evidence = gr.JSON(label="Immutable evidence", open=True)
        gr.Markdown("## 03 / Localize", elem_classes=["df-section"])
        with gr.Row():
            mask = gr.Image(
                label="Binary defect mask",
                image_mode="L",
                buttons=["download", "fullscreen"],
                elem_classes=["df-output"],
            )
            heatmap = gr.Image(
                label="Defect probability heatmap",
                buttons=["download", "fullscreen"],
                elem_classes=["df-output"],
            )
        gr.Markdown(
            "### Research-use boundary\n"
            "This is an open research and evaluation demo, **not a production AOI or "
            "safety system**. The app code does not save uploaded images. The original "
            "v1 study reported negative mean synthetic-data gains and preserves that "
            f"result publicly. [GitHub]({GITHUB_URL}) · [Dataset]({DATASET_URL})",
            elem_classes=["df-note"],
        )
        run.click(
            fn=_run,
            inputs=[image, object_name, threshold],
            outputs=[probabilities, mask, heatmap, summary, evidence],
            concurrency_limit=1,
            concurrency_id="model-inference",
            show_progress="full",
            api_visibility="private",
            scroll_to_output=True,
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
            primary_hue=gr.themes.colors.amber,
            secondary_hue=gr.themes.colors.emerald,
            neutral_hue=gr.themes.colors.slate,
        ),
        css=CSS,
    )
