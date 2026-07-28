"""Public DefectForge classification and segmentation demo."""

from __future__ import annotations

import html
import os
from pathlib import Path

import gradio as gr
from runtime import MODEL_ROOT, SpaceContractError, load_manifest, predict, preferred_device

GITHUB_URL = "https://github.com/kuotunyu/01-defectforge-visa"
DATASET_URL = "https://huggingface.co/datasets/steven0226/defectforge-visa-synthetic"
APP_ROOT = Path(__file__).resolve().parent
DEMO_EXAMPLES = [
    [str(APP_ROOT / "examples/pcb1_defect_a.JPG"), "pcb1"],
    [str(APP_ROOT / "examples/pcb1_defect_b.JPG"), "pcb1"],
    [str(APP_ROOT / "examples/pcb1_normal.JPG"), "pcb1"],
    [str(APP_ROOT / "examples/capsules_defect.JPG"), "capsules"],
    [str(APP_ROOT / "examples/capsules_normal.JPG"), "capsules"],
]
DEMO_GALLERY = [
    (DEMO_EXAMPLES[0][0], "PCB 瑕疵 A"),
    (DEMO_EXAMPLES[1][0], "PCB 瑕疵 B"),
    (DEMO_EXAMPLES[2][0], "PCB 正常"),
    (DEMO_EXAMPLES[3][0], "膠囊瑕疵"),
    (DEMO_EXAMPLES[4][0], "膠囊正常"),
]

CSS = """
:root {
  color-scheme: light;
  --df-canvas: oklch(97% .012 215);
  --df-surface: oklch(100% 0 0);
  --df-surface-soft: oklch(94% .035 171);
  --df-surface-muted: oklch(98% .008 215);
  --df-surface-blue: oklch(95% .029 234);
  --df-surface-peach: oklch(95% .041 72);
  --df-surface-lilac: oklch(95% .027 302);
  --df-result-soft: oklch(96% .024 302);
  --df-ink: oklch(25% .035 222);
  --df-muted: oklch(39% .029 214);
  --df-primary: oklch(44% .09 174);
  --df-primary-strong: oklch(32% .068 178);
  --df-primary-soft: oklch(91% .045 171);
  --df-accent: oklch(65% .105 67);
  --df-border: oklch(85% .023 212);
  --df-border-strong: oklch(69% .035 208);
  --df-danger: oklch(48% .17 28);
  --df-focus: oklch(55% .12 221);
  --df-radius-sm: 2px;
  --df-radius-md: 4px;
  --df-radius-lg: 4px;
  --df-shadow: 0 2px 8px oklch(35% .03 220 / .07);
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
  --text-xs: 1.1875rem;
  --text-sm: 1.1875rem;
  --text-md: 1.25rem;
  --text-lg: 1.375rem;
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
  font-size: 1.25rem !important;
  line-height: 1.55 !important;
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
  font-size: 1.25rem !important;
}
.gradio-container .prose p,
.gradio-container .prose li {
  line-height: 1.55 !important;
}
.gradio-container .info,
.gradio-container .secondary-wrap,
.gradio-container .secondary-wrap *,
.gradio-container .label-wrap,
.gradio-container .label-wrap * {
  font-size: 1.1875rem !important;
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
  border: 0;
  border-radius: var(--df-radius-lg);
  box-shadow: var(--df-shadow);
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
  font-size: 1.25rem;
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
  border-radius: 2px;
  background: var(--df-primary);
  color: #fff;
  font-weight: 800;
  letter-spacing: -.02em;
}
.df-runtime {
  display: inline-flex;
  align-items: center;
  gap: .5rem;
  min-height: 40px;
  padding: .4rem .8rem;
  border-radius: 3px;
  background: var(--df-primary-soft);
  color: var(--df-primary-strong);
  font-size: 1.1875rem;
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
  font-size: 2.5rem;
  font-weight: 800;
  line-height: 1.2;
  letter-spacing: -.025em;
  text-wrap: balance;
}
#df-header p {
  max-width: 100ch;
  margin: 0;
  color: var(--df-muted);
  font-size: 1.25rem;
  line-height: 1.5;
  text-wrap: pretty;
}
#df-header strong {
  color: var(--df-primary-strong);
}
.df-flow {
  display: block;
  margin: 1rem -1.5rem -1.25rem;
}
.df-flow ol {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  grid-auto-rows: 80px;
  align-items: stretch;
  gap: 0;
  margin: 0;
  padding: 0;
  list-style: none;
}
.df-flow li {
  display: grid;
  grid-template-columns: 42px 7.5rem;
  align-items: center;
  justify-content: center;
  gap: .8rem;
  width: 100%;
  height: 80px !important;
  min-height: 80px !important;
  padding: 0 1.5rem;
  box-sizing: border-box;
  color: var(--df-ink);
  font-size: 1.25rem;
  font-weight: 700;
}
.df-flow li:nth-child(1) {
  background: var(--df-surface-soft);
}
.df-flow li:nth-child(2) {
  background: var(--df-surface-blue);
}
.df-flow li:nth-child(3) {
  background: var(--df-surface-peach);
}
.df-flow li span {
  color: var(--df-ink) !important;
  text-align: left;
  white-space: nowrap;
}
.df-flow li + li::before {
  content: none;
}
.df-flow b {
  display: inline-grid;
  width: 42px;
  height: 42px;
  place-items: center;
  border-radius: 2px;
  background: var(--df-surface);
  color: var(--df-primary-strong);
  font-size: 1.25rem;
}
#df-workspace {
  flex-direction: column !important;
  gap: var(--df-space-md) !important;
  align-items: stretch !important;
  margin-top: 0 !important;
}
#df-workspace > * {
  width: 100% !important;
  min-width: 0 !important;
}
.df-panel {
  min-width: 0 !important;
  padding: 1.25rem 1.35rem !important;
  background: var(--df-surface) !important;
  border: 0 !important;
  border-radius: var(--df-radius-lg) !important;
  box-shadow: var(--df-shadow) !important;
}
#df-result-panel {
  margin-top: .25rem !important;
  padding-top: 1rem !important;
  border-top: 6px solid var(--df-primary) !important;
  background: var(--df-surface) !important;
}
.df-panel-heading {
  margin-bottom: .85rem;
}
.df-panel-heading h2 {
  margin: 0 0 .2rem;
  color: var(--df-ink);
  font-size: 1.75rem;
  font-weight: 800;
  line-height: 1.3;
}
.df-panel-heading p {
  margin: 0;
  color: var(--df-muted);
  font-size: 1.25rem !important;
  line-height: 1.55 !important;
}
#df-object,
#df-upload,
#df-probabilities,
.df-output-image,
#df-advanced {
  border-color: var(--df-border) !important;
  border-radius: var(--df-radius-sm) !important;
}
#df-input-grid {
  gap: var(--df-space-md) !important;
  align-items: stretch !important;
}
.df-setup {
  gap: .75rem !important;
  min-width: 270px !important;
  padding: 1rem !important;
  background: var(--df-surface-soft) !important;
  border: 0 !important;
  border-radius: var(--df-radius-sm) !important;
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
#df-object > span[data-testid="block-info"] {
  color: var(--df-ink) !important;
  font-size: 1.25rem !important;
  font-weight: 700 !important;
}
#df-object .info-text {
  color: var(--df-muted) !important;
  font-size: 1.1875rem !important;
  line-height: 1.5 !important;
}
.df-setup > .form {
  background: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}
#df-object .wrap {
  display: grid !important;
  grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  gap: .75rem !important;
}
#df-object .wrap label {
  position: relative;
  display: grid !important;
  grid-template-columns: 48px minmax(0, 1fr);
  align-items: center;
  gap: .7rem !important;
  min-height: 88px !important;
  padding: .8rem !important;
  border-radius: var(--df-radius-sm) !important;
  background: var(--df-surface) !important;
  border: 0 !important;
  cursor: pointer;
  transition:
    background-color 160ms cubic-bezier(.16, 1, .3, 1),
    transform 160ms cubic-bezier(.16, 1, .3, 1) !important;
}
#df-object .wrap label::before {
  content: "PCB";
  display: grid;
  width: 48px;
  height: 48px;
  place-items: center;
  background: var(--df-surface-blue);
  color: oklch(35% .07 230);
  font-size: 1rem;
  font-weight: 850;
  letter-spacing: .04em;
}
#df-object .wrap label:nth-of-type(2)::before {
  content: "CAP";
  background: var(--df-surface-peach);
  color: oklch(38% .08 60);
}
#df-object .wrap label input {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
}
#df-object .wrap label span {
  font-size: 1.1875rem !important;
  font-weight: 750 !important;
  line-height: 1.35 !important;
  text-wrap: balance;
}
#df-object .wrap label:hover {
  transform: translateY(-1px);
}
#df-object .wrap label:has(input:checked) {
  background: var(--df-primary-soft) !important;
  color: var(--df-primary-strong) !important;
  box-shadow: inset 0 0 0 2px var(--df-primary) !important;
}
#df-object .wrap label:has(input:focus-visible) {
  outline: 3px solid color-mix(in srgb, var(--df-focus) 55%, transparent) !important;
  outline-offset: 2px !important;
}
.df-object-help {
  margin: 0;
  color: var(--df-muted);
  font-size: 1.1875rem;
}
.df-privacy {
  display: flex;
  gap: .65rem;
  align-items: center;
  margin: .75rem 0 0;
  padding: .75rem .9rem;
  border-radius: 0;
  background: var(--df-surface-soft);
  color: var(--df-primary-strong);
  font-size: 1.1875rem;
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
  background: var(--df-surface-blue) !important;
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
  font-size: 1.25rem !important;
}
#df-advanced {
  margin-top: auto !important;
  background: transparent !important;
  border-top: 1px solid var(--df-border) !important;
  border-bottom: 1px solid var(--df-border) !important;
  border-right: 0 !important;
  border-left: 0 !important;
  border-radius: 0 !important;
}
#df-advanced summary {
  min-height: 52px;
  color: var(--df-ink) !important;
  font-size: 1.1875rem !important;
  font-weight: 700;
}
#df-advanced .label-wrap,
#df-advanced .label-wrap * {
  color: var(--df-ink) !important;
  font-size: 1.1875rem !important;
}
.df-examples-intro {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 1rem;
  margin: 1rem 0 .55rem;
}
.df-examples-intro h3 {
  margin: 0 0 .15rem;
  color: var(--df-ink);
  font-size: 1.5rem;
  font-weight: 800;
}
.df-examples-intro p {
  margin: 0;
  color: var(--df-muted);
  font-size: 1.1875rem;
  line-height: 1.5;
}
.df-examples-intro a {
  flex: 0 0 auto;
  font-size: 1.1875rem;
  font-weight: 700;
}
#df-examples {
  margin: 0 !important;
  padding: .85rem !important;
  background: var(--df-surface-muted) !important;
  border: 0 !important;
  border-radius: var(--df-radius-sm) !important;
}
#df-examples > label {
  color: var(--df-ink) !important;
  font-size: 1.1875rem !important;
  font-weight: 750 !important;
}
#df-examples .label-wrap,
#df-examples .label-wrap * {
  color: var(--df-ink) !important;
  font-size: 1.1875rem !important;
  font-weight: 750 !important;
}
#df-examples button {
  font-size: 1.1875rem !important;
}
#df-examples .thumbnail-item {
  overflow: hidden !important;
  border: 0 !important;
  border-radius: var(--df-radius-sm) !important;
  background: var(--df-surface) !important;
}
#df-examples .grid-wrap {
  overflow-y: hidden !important;
}
#df-examples .thumbnail-item:focus-visible {
  outline: 3px solid color-mix(in srgb, var(--df-focus) 55%, transparent) !important;
  outline-offset: 2px !important;
}
#df-examples .caption-label {
  padding: .55rem .65rem !important;
  background: oklch(23% .035 222 / .92) !important;
  color: #fff !important;
  font-size: 1.125rem !important;
  font-weight: 750 !important;
}
#df-examples img {
  transition: transform 180ms cubic-bezier(.16, 1, .3, 1);
}
#df-examples .thumbnail-item:hover img {
  transform: scale(1.018);
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
  min-height: 60px !important;
  margin-top: .75rem !important;
  border: 0 !important;
  border-radius: var(--df-radius-sm) !important;
  background: var(--df-primary) !important;
  color: #fff !important;
  font-size: 1.3125rem !important;
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
  padding: 0 !important;
  background: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
}
#df-result-overview {
  align-items: stretch !important;
  gap: var(--df-space-md) !important;
}
#df-result-overview > * {
  min-width: 0 !important;
}
#df-decision-card,
#df-confidence-card {
  min-height: 250px !important;
  padding: 1.35rem !important;
  border: 0 !important;
  border-radius: var(--df-radius-sm) !important;
}
#df-decision-card {
  justify-content: center !important;
  background: var(--df-primary-strong) !important;
  color: #fff !important;
}
#df-confidence-card {
  background: var(--df-surface-blue) !important;
}
.df-decision {
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 200px;
}
.df-decision-kicker,
.df-confidence-title span,
.df-result-section-kicker {
  color: oklch(86% .07 172);
  font-size: 1.0625rem;
  font-weight: 800;
  letter-spacing: .08em;
}
.df-decision h3 {
  display: flex;
  align-items: center;
  gap: .7rem;
  margin: .55rem 0 .25rem;
  color: #fff;
  font-size: 2rem;
  font-weight: 850;
  line-height: 1.25;
}
.df-decision-dot {
  width: 12px;
  height: 12px;
  flex: 0 0 auto;
  background: var(--df-accent);
  box-shadow: 0 0 0 6px oklch(78% .12 72 / .16);
}
.df-decision-label {
  margin: 0 0 1.15rem;
  color: oklch(94% .025 180);
  font-size: 1.25rem;
  font-weight: 700;
}
.df-result-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: .65rem;
  margin: 0;
}
.df-result-metrics div {
  padding-top: .65rem;
  border-top: 1px solid oklch(92% .04 177 / .28);
}
.df-result-metrics dt {
  color: oklch(84% .035 180);
  font-size: 1.0625rem;
}
.df-result-metrics dd {
  margin: .2rem 0 0;
  color: #fff;
  font-size: 1.25rem;
  font-weight: 800;
  font-variant-numeric: tabular-nums;
}
.df-decision-threshold {
  margin: .75rem 0 0;
  color: oklch(84% .035 180);
  font-size: 1.0625rem;
}
.df-confidence-title {
  margin-bottom: 1rem;
}
.df-confidence-title span {
  color: var(--df-primary);
}
.df-confidence-title h3 {
  margin: .25rem 0 .15rem;
  color: var(--df-ink);
  font-size: 1.625rem;
  font-weight: 800;
}
.df-confidence-title p {
  margin: 0;
  color: var(--df-muted);
  font-size: 1.125rem;
}
#df-probabilities {
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
.df-confidence-bars {
  display: grid;
  gap: 1rem;
}
.df-confidence-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  color: var(--df-ink);
  font-size: 1.1875rem;
}
.df-confidence-row strong {
  color: var(--df-primary-strong);
  font-size: 1.25rem;
  font-variant-numeric: tabular-nums;
}
.df-confidence-track {
  overflow: hidden;
  height: 10px;
  margin-top: .4rem;
  background: oklch(88% .025 220);
}
.df-confidence-track span {
  display: block;
  width: 0;
  height: 100%;
  background: var(--df-primary);
}
.df-confidence-item:nth-child(2) .df-confidence-track span {
  background: oklch(70% .065 230);
}
.df-result-heading .df-result-section-kicker,
.df-section-heading .df-result-section-kicker {
  color: var(--df-primary);
}
.df-section-heading {
  margin: 1.15rem 0 .65rem;
}
.df-section-heading h2 {
  margin: 0 0 .15rem;
  color: var(--df-ink);
  font-size: 1.75rem;
  font-weight: 800;
}
.df-section-heading p {
  margin: 0;
  color: var(--df-muted);
  font-size: 1.1875rem !important;
}
#df-localization {
  gap: var(--df-space-md) !important;
  padding: 1rem !important;
  background: var(--df-canvas) !important;
}
.df-output-image {
  overflow: hidden !important;
  background: var(--df-surface) !important;
  border: 0 !important;
  border-radius: var(--df-radius-sm) !important;
  box-shadow: var(--df-shadow) !important;
}
.df-output-image .label-wrap {
  background: var(--df-surface) !important;
}
#df-evidence {
  margin-top: 1rem !important;
  border-top: 1px solid var(--df-border) !important;
  border-bottom: 1px solid var(--df-border) !important;
  border-right: 0 !important;
  border-left: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
}
#df-evidence summary {
  min-height: 56px;
  color: var(--df-ink) !important;
  font-size: 1.1875rem !important;
  font-weight: 750;
}
#df-evidence .label-wrap,
#df-evidence .label-wrap * {
  color: var(--df-ink) !important;
  font-size: 1.1875rem !important;
}
.df-boundary {
  margin-top: 1rem;
  padding: 1rem 1.1rem;
  border: 0;
  border-radius: 0;
  background: var(--df-surface-peach);
  color: oklch(37% .07 65);
  font-size: 1.1875rem;
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
  font-size: 1.1875rem;
}
.df-footer-links {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
}
footer { display: none !important; }
@media (max-width: 900px) {
  #df-input-grid,
  #df-result-overview {
    align-items: stretch !important;
    flex-direction: column !important;
  }
  #df-input-grid > *,
  #df-result-overview > * {
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
  .df-runtime {
    justify-content: center;
    width: 100%;
    padding: .35rem .5rem;
    font-size: 1.0625rem;
    white-space: nowrap;
  }
  #df-header h1 {
    font-size: 2.125rem;
  }
  .df-flow {
    margin: 1rem -1rem -1rem;
  }
  .df-flow ol {
    grid-template-columns: 1fr;
  }
  .df-flow li {
    justify-content: center;
  }
  #df-object .wrap {
    grid-template-columns: 1fr !important;
  }
  #df-examples {
    height: 380px !important;
  }
  #df-examples .grid-wrap {
    height: 380px !important;
  }
  #df-examples .grid-container {
    --grid-cols: 2 !important;
    --grid-rows: 3 !important;
  }
  #df-examples .caption-label {
    max-width: 100% !important;
    padding: .35rem .45rem !important;
  }
  .df-result-metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  #df-decision-card,
  #df-confidence-card {
    min-height: 0 !important;
  }
  #df-localization {
    flex-direction: column !important;
  }
  .df-examples-intro {
    align-items: flex-start;
    flex-direction: column;
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


def _result_summary_html(
    probabilities: dict[str, float],
    object_name: str,
    evidence: dict[str, object],
) -> str:
    anomaly_probability = float(probabilities.get("Defect（異常）", 0.0))
    inference = evidence.get("inference", {})
    if not isinstance(inference, dict):
        inference = {}
    is_defect = anomaly_probability >= 0.5
    title = "偵測到瑕疵" if is_defect else "判定為正常"
    decision = "Defect" if is_defect else "Normal"
    device = html.escape(str(evidence.get("device", "unknown")).upper())
    escaped_object = html.escape(object_name)
    elapsed_ms = float(inference.get("elapsed_ms", 0.0))
    threshold = float(inference.get("visualization_threshold", 0.5))
    coverage = float(inference.get("mask_coverage_percent", 0.0))
    return (
        "<section class='df-decision' lang='zh-Hant-TW'>"
        "<span class='df-decision-kicker'>模型判定</span>"
        f"<h3><span class='df-decision-dot' aria-hidden='true'></span>{title}</h3>"
        f"<p class='df-decision-label'>{decision} · 分類信心 "
        f"{anomaly_probability:.0%}</p>"
        "<dl class='df-result-metrics'>"
        f"<div><dt>物件</dt><dd>{escaped_object}</dd></div>"
        f"<div><dt>執行裝置</dt><dd>{device}</dd></div>"
        f"<div><dt>耗時</dt><dd>{elapsed_ms:.0f} ms</dd></div>"
        f"<div><dt>Mask coverage</dt><dd>{coverage:.2f}%</dd></div>"
        "</dl>"
        f"<p class='df-decision-threshold'>顯示 threshold：{threshold:.2f}</p>"
        "</section>"
    )


def _confidence_html(probabilities: dict[str, float]) -> str:
    defect = min(1.0, max(0.0, float(probabilities.get("Defect（異常）", 0.0))))
    normal = min(1.0, max(0.0, float(probabilities.get("Normal（正常）", 0.0))))
    rows = (
        ("Defect｜異常", defect),
        ("Normal｜正常", normal),
    )
    items = "".join(
        "<div class='df-confidence-item' role='listitem'>"
        "<div class='df-confidence-row'>"
        f"<span>{label}</span><strong>{value:.0%}</strong>"
        "</div>"
        "<div class='df-confidence-track' aria-hidden='true'>"
        f"<span style='width:{value * 100:.1f}%'></span>"
        "</div></div>"
        for label, value in rows
    )
    return (
        "<div class='df-confidence-bars' role='list' "
        "aria-label='Classification confidence'>"
        f"{items}</div>"
    )


def _run(image: object, object_name: str, threshold: float) -> tuple[object, ...]:
    try:
        probabilities, mask, heatmap, _summary, evidence = predict(
            image,
            object_name,
            threshold,
        )
        summary = _result_summary_html(probabilities, object_name, evidence)
        confidence = _confidence_html(probabilities)
        return (
            gr.Column(visible=True),
            gr.HTML(
                value=confidence,
                visible=True,
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
        gr.HTML(visible=False),
        None,
        None,
        "<p>重新按下「開始檢測」後，結果會顯示在這裡。</p>",
        {},
        gr.Column(visible=False),
        gr.Accordion(visible=False),
    )


def _select_example(evt: gr.SelectData) -> tuple[str, str]:
    index = evt.index[0] if isinstance(evt.index, tuple) else evt.index
    if not isinstance(index, int) or not 0 <= index < len(DEMO_EXAMPLES):
        raise gr.Error("無法載入這張範例影像，請改選其他範例。", duration=6)
    image_path, object_name = DEMO_EXAMPLES[index]
    return image_path, object_name


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
            "<ol>"
            "<li><b>01</b><span>選擇物件</span></li>"
            "<li><b>02</b><span>上傳影像</span></li>"
            "<li><b>03</b><span>開始檢測</span></li>"
            "</ol></nav>"
            "</section>"
        )
        with gr.Column(elem_id="df-workspace"):
            with gr.Column(
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
                    "<div class='df-examples-intro' lang='zh-Hant-TW'>"
                    "<div><h3>手上沒有影像？直接選一張範例</h3>"
                    "<p>點一下會自動帶入正確物件與影像，再按「開始檢測」即可。</p></div>"
                    "<a href='https://registry.opendata.aws/visa/' target='_blank' "
                    "rel='noopener'>VisA · CC BY 4.0</a></div>"
                )
                examples = gr.Gallery(
                    value=DEMO_GALLERY,
                    label="5 張可直接試玩的範例影像",
                    columns=5,
                    rows=1,
                    height=290,
                    allow_preview=False,
                    object_fit="cover",
                    buttons=[],
                    elem_id="df-examples",
                )
                gr.HTML(
                    "<div class='df-privacy' lang='zh-Hant-TW'>"
                    "<strong>隱私說明</strong><span>影像只在記憶體中處理；"
                    "本程式不會儲存你上傳的檔案。</span></div>"
                )
                run = gr.Button("開始檢測", variant="primary", elem_id="df-run")
            with gr.Column(
                visible=False,
                elem_id="df-result-panel",
                elem_classes=["df-panel"],
            ) as result_panel:
                gr.HTML(
                    "<div class='df-panel-heading df-result-heading' lang='zh-Hant-TW'>"
                    "<span class='df-result-section-kicker'>03 · 檢測輸出</span>"
                    "<h2>檢測結果</h2>"
                    "<p>先確認模型判定與分類信心，再查看瑕疵位置。</p>"
                    "</div>"
                )
                with gr.Row(equal_height=True, elem_id="df-result-overview"):
                    with gr.Column(scale=5, elem_id="df-decision-card"):
                        summary = gr.HTML(
                            "<p>按下「開始檢測」後，結果會顯示在這裡。</p>",
                            elem_id="df-summary",
                        )
                    with gr.Column(scale=4, elem_id="df-confidence-card"):
                        gr.HTML(
                            "<div class='df-confidence-title' lang='zh-Hant-TW'>"
                            "<span>分類信心</span>"
                            "<h3>Confidence distribution</h3>"
                            "<p>數值越高，代表模型越偏向該分類。</p>"
                            "</div>"
                        )
                        probabilities = gr.HTML(
                            "<div class='df-confidence-bars'></div>",
                            elem_id="df-probabilities",
                            visible=False,
                        )
                with gr.Column(visible=False) as output_details:
                    gr.HTML(
                        "<div class='df-section-heading' lang='zh-Hant-TW'>"
                        "<span class='df-result-section-kicker'>瑕疵定位</span>"
                        "<h2>Mask 與 heatmap</h2>"
                        "<p>Binary mask 顯示超過 threshold 的區域；"
                        "heatmap 保留完整機率分布。</p>"
                        "</div>"
                    )
                    with gr.Row(elem_id="df-localization"):
                        mask = gr.Image(
                            label="Binary mask",
                            image_mode="L",
                            buttons=["download", "fullscreen"],
                            height=390,
                            elem_classes=["df-output-image"],
                        )
                        heatmap = gr.Image(
                            label="Probability heatmap",
                            buttons=["download", "fullscreen"],
                            height=390,
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
        examples.select(
            fn=_select_example,
            inputs=None,
            outputs=[image, object_name],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
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
