#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def load_thresholds(model_dir: Path, num_labels: int):
    th = model_dir / "thresholds.json"
    if th.exists():
        with open(th, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if isinstance(arr, list) and len(arr) == num_labels:
            return np.array(arr, dtype=np.float32)
    return np.full(num_labels, 0.5, dtype=np.float32)

def join_title_body(title: str, body: str) -> str:
    title = (title or "").strip()
    body = (body or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body

def iter_texts(args):
    # Prefer explicit title/body if provided
    if getattr(args, "title", None) or getattr(args, "body", None):
        combined = join_title_body(args.title, args.body)
        if combined:
            yield combined
        return

    # Back-compat: --text, --infile, or stdin
    if args.text:
        yield args.text
    elif args.infile:
        with open(args.infile, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s: yield s
    else:
        for line in sys.stdin:
            s = line.strip()
            if s: yield s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="models/NEWBEST")
    # NEW: explicit fields (preferred)
    ap.add_argument("--title", type=str, help="Issue title")
    ap.add_argument("--body", type=str, help="Issue body/description")
    # Back-compat flags
    ap.add_argument("--text", type=str, help="Single input text")
    ap.add_argument("--infile", type=str, help="File with one text per line")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_length", type=int, default=384)
    args = ap.parse_args()

    mdir = Path(args.model_dir)
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir)
    model.to(args.device).eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    num_labels = model.config.num_labels
    thresholds = load_thresholds(mdir, num_labels)

    def predict_batch(texts):
        enc = tok(texts, truncation=True, max_length=args.max_length,
                  padding=True, return_tensors="pt")
        enc = {k: v.to(args.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits.detach().cpu().numpy()
        return sigmoid(logits)

    # stream in small batches
    B, buf, indices, i = 16, [], [], 0
    for t in iter_texts(args):
        buf.append(t); indices.append(i); i += 1
        if len(buf) == B:
            probs = predict_batch(buf)
            for j, p in enumerate(probs):
                emit(indices[j], buf[j], p, thresholds, id2label, args.topk)
            buf, indices = [], []
    if buf:
        probs = predict_batch(buf)
        for j, p in enumerate(probs):
            emit(indices[j], buf[j], p, thresholds, id2label, args.topk)

def emit(i, text, probs, thresholds, id2label, topk):
    order = np.argsort(-probs)
    chosen = [id2label[k] for k in range(len(probs)) if probs[k] >= thresholds[k]]
    top = [{"label": id2label[k], "prob": float(probs[k])} for k in order[:topk]]
    print(json.dumps({"index": i, "text": text, "predicted_labels": chosen, "topk": top},
                     ensure_ascii=False))

if __name__ == "__main__":
    main()
