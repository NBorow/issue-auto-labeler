#!/usr/bin/env python3
import os, json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# ── Config ──────────────────────────────────────────────────────────────────────
# Point to your Hub repo, and the subfolder that contains the model files.
# Example Space setting: HF_MODEL_ID=NBorow/issue-auto-labeler
MODEL_ID = os.getenv("HF_MODEL_ID", "NBorow/issue-auto-labeler")
SUBFOLDER = os.getenv("HF_MODEL_SUBFOLDER", "models/NEWBEST")

# ── Load model & tokenizer ─────────────────────────────────────────────────────
def load_tok_and_model(model_id: str, subfolder: str):
    # Try Hub (repo id + subfolder)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, subfolder=subfolder)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id, subfolder=subfolder)
    return tok, mdl

tokenizer, model = load_tok_and_model(MODEL_ID, SUBFOLDER)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# Label mapping from saved config
id2label = {int(k): v for k, v in model.config.id2label.items()}
labels = [id2label[i] for i in range(len(id2label))]

# ── Thresholds ──────────────────────────────────────────────────────────────────
def load_thresholds(repo_or_dir: str, subfolder: str, n_labels: int) -> np.ndarray:
    # 1) Local folder (if running locally with a cloned repo)
    local_path = Path(repo_or_dir) / subfolder / "thresholds.json"
    if local_path.exists():
        try:
            arr = np.array(json.loads(local_path.read_text(encoding="utf-8")), dtype=np.float32)
            if arr.shape[0] == n_labels:
                return arr
        except Exception:
            pass
    # 2) Hub (Space)
    try:
        thr_path = hf_hub_download(repo_id=repo_or_dir, filename="thresholds.json", subfolder=subfolder)
        arr = np.array(json.loads(Path(thr_path).read_text(encoding="utf-8")), dtype=np.float32)
        if arr.shape[0] == n_labels:
            return arr
    except Exception:
        pass
    # 3) Fallback to 0.5 for all labels
    return np.full(n_labels, 0.5, dtype=np.float32)

per_class_thresholds = load_thresholds(MODEL_ID, SUBFOLDER, len(labels))
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

# ── Helpers ─────────────────────────────────────────────────────────────────────
def join_title_body(title: str, body: str) -> str:
    title = (title or "").strip()
    body = (body or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body

def predict_one(
    text: str,
    mode: str,                 # "Validation-tuned per-label" | "Global threshold"
    global_thr: float,         # used only when mode == "Global threshold"
    min_prob: float = 0.0,
    max_labels: int = 5,
    cap_length: int = 384
) -> Tuple[List[str], List[List]]:
    if not text.strip():
        return [], []

    enc = tokenizer(text, truncation=True, max_length=cap_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits.detach().cpu().numpy()[0]
    probs = sigmoid(logits)

    # threshold selection
    if mode == "Global threshold":
        thrs = np.full(len(labels), float(global_thr), dtype=np.float32)
    else:
        thrs = per_class_thresholds

    # Multi-label decision
    keep = probs >= thrs
    chosen = [
        (labels[i], float(probs[i]), float(thrs[i]))
        for i in range(len(labels))
        if keep[i] and probs[i] >= min_prob
    ]
    chosen.sort(key=lambda x: x[1], reverse=True)
    chosen = chosen[:max_labels]

    # Top scores table (rounded for display)
    top_order = np.argsort(-probs)[: max(10, max_labels)]
    table = []
    for i in top_order:
        table.append([
            labels[i],
            round(float(probs[i]), 3),
            round(float(thrs[i]), 3),
            bool(probs[i] >= thrs[i])
        ])

    predicted_labels = [c[0] for c in chosen]
    return predicted_labels, table

def predict_batch(texts: List[str], mode: str, global_thr: float, min_prob: float, max_labels: int, cap_length: int):
    results = []
    for idx, t in enumerate(texts):
        preds, table = predict_one(t, mode, global_thr, min_prob, max_labels, cap_length)
        results.append({"index": idx, "text": t, "predicted": preds, "labels_table": table})
    return results

# ── Gradio UI ───────────────────────────────────────────────────────────────────
def ui_single(title, body, mode, global_thr, min_prob, max_labels, cap_len):
    text = join_title_body(title, body)
    preds, table = predict_one(text, mode, global_thr, min_prob, max_labels, cap_len)
    return ", ".join(preds) if preds else "(no labels above threshold)", table

def ui_batch(text_blob, mode, global_thr, min_prob, max_labels, cap_len):
    lines = [s for s in (text_blob or "").splitlines() if s.strip()]
    results = predict_batch(lines, mode, global_thr, min_prob, max_labels, cap_len)
    jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in results)
    preview = [[r["index"], r["text"][:80], ", ".join(r["predicted"])] for r in results[:20]]
    return jsonl, preview

with gr.Blocks(title="Issue Auto-Labeler (NEWBEST)") as demo:
    gr.Markdown(
        "### Issue Auto-Labeler · RoBERTa-base (multi-label)\n"
        "Probabilities **don’t** sum to 1 — each label is independent (sigmoid + per-class thresholds)."
    )

    # General controls
    with gr.Row():
        min_prob = gr.Slider(0.0, 0.9, value=0.0, step=0.01, label="Min probability")
        max_labels = gr.Slider(1, 10, value=5, step=1, label="Max labels")
        cap_len = gr.Slider(64, 512, value=384, step=16, label="Token cap")

    # Threshold mode (clean layout)
    threshold_mode = gr.Radio(
        ["Validation-tuned per-label", "Global threshold"],
        value="Validation-tuned per-label",
        label="Threshold mode", interactive=True
    )
    global_thr = gr.Slider(0.0, 0.9, value=0.5, step=0.01,
                           label="Threshold value", visible=False)

    threshold_mode.change(
        lambda mode: gr.update(visible=(mode == "Global threshold")),
        inputs=[threshold_mode],
        outputs=[global_thr],
    )

    with gr.Tab("Single"):
        title_in = gr.Textbox(lines=1, label="Title")
        body_in = gr.Textbox(lines=8, label="Body / description")
        out_labels = gr.Textbox(label="Predicted labels", interactive=False)
        out_table = gr.Dataframe(
            headers=["label", "prob", "threshold", "≥ threshold"],
            datatype=["str", "number", "number", "bool"],
            wrap=True
        )
        gr.Button("Predict").click(
            ui_single,
            [title_in, body_in, threshold_mode, global_thr, min_prob, max_labels, cap_len],
            [out_labels, out_table]
        )

    with gr.Tab("Batch"):
        gr.Markdown(
            "One issue per line (already concatenated as `Title\\n\\nBody`). "
            "Output is JSONL with: index, text, predicted, labels_table."
        )
        txt_batch = gr.Textbox(lines=10, label="Issues (one per line)")
        out_jsonl = gr.Textbox(label="JSONL output", lines=10)
        out_preview = gr.Dataframe(headers=["index", "text (preview)", "predicted"], wrap=True)
        gr.Button("Predict batch").click(
            ui_batch,
            [txt_batch, threshold_mode, global_thr, min_prob, max_labels, cap_len],
            [out_jsonl, out_preview]
        )

    gr.Examples(
        [
            ["Smogifier fails to start", "Fix the uhhhhh smogifier"],
            ["Add CSV export", "A toggle to export reports as CSV would help ops."],
            ["Dataset viewer timeout", "Why is the dataset viewer timing out on large parquet files?"],
        ],
        inputs=[title_in, body_in],
    )

if __name__ == "__main__":
    # Works locally; on Spaces, launch is managed by the platform.
    demo.launch()
