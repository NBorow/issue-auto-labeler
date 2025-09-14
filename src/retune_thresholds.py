# src/retune_thresholds.py (fixed)
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import f1_score, precision_recall_fscore_support

def load_label_maps(data_dir: Path):
    with open(data_dir / "label_to_id.json", "r", encoding="utf-8") as f:
        l2i = json.load(f)
    return l2i, {int(v): k for k, v in l2i.items()}

def to_multi_hot(batch, l2i, C):
    out = []
    for names in batch["labels"]:
        v = np.zeros(C, dtype=np.float32)
        for nm in names:
            v[l2i[nm]] = 1.0
        out.append(v.tolist())
    return {"labels": out}

def tokenize(batch, tok):
    return tok(batch["text"], truncation=True)

def logits_labels(model, ds, tok, bs=64):
    coll = DataCollatorWithPadding(tokenizer=tok)
    keep = {"input_ids","attention_mask","labels"}
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=coll)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    L, Y = [], []
    with torch.no_grad():
        for b in dl:
            logits = model(input_ids=b["input_ids"].to(device),
                           attention_mask=b["attention_mask"].to(device)).logits
            L.append(logits.cpu().numpy()); Y.append(b["labels"].numpy())
    return np.concatenate(L), np.concatenate(Y)

def sweep_per_class(val_logits, val_y, grid=None):
    if grid is None: grid = np.linspace(0.1, 0.9, 33)
    probs = 1/(1+np.exp(-val_logits)); C = probs.shape[1]
    th = np.full(C, 0.5, dtype=np.float32)
    for c in range(C):
        best, tbest = -1.0, 0.5
        pc, yc = probs[:,c], val_y[:,c]
        for t in grid:
            f1 = f1_score(yc, (pc>=t).astype(int), zero_division=0)
            if f1 > best: best, tbest = f1, t
        th[c] = tbest
    return th

def report(name, logits, y, th):
    p = 1/(1+np.exp(-logits)); yhat = (p>=th).astype(int)
    f1_micro = f1_score(y, yhat, average="micro", zero_division=0)
    f1_macro = f1_score(y, yhat, average="macro", zero_division=0)
    f1_samples = f1_score(y, yhat, average="samples", zero_division=0)
    pr, rc, f1_w, _ = precision_recall_fscore_support(y, yhat, average="weighted", zero_division=0)
    subset = (yhat == y).all(axis=1).mean()
    print(f"{name} micro-F1={f1_micro:.3f}  macro-F1={f1_macro:.3f}  samples-F1={f1_samples:.3f}  weighted-F1={f1_w:.3f}  subset-acc={subset:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    data_dir, model_dir = Path(args.data_dir), Path(args.model_dir)
    l2i, _ = load_label_maps(data_dir); C = len(l2i)

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)

    ds = load_dataset(
        "json",
        data_files={
            "validation": str(data_dir / ("val.jsonl" if (data_dir / "val.jsonl").exists() else "valid.jsonl")),
            "test": str(data_dir / "test.jsonl"),
        },
        split={"validation": "validation", "test": "test"},
    )

    # multi-hot labels then tokenize text, for both splits
    for s in ["validation","test"]:
        ds[s] = ds[s].map(lambda b: to_multi_hot(b, l2i, C), batched=True)
        ds[s] = ds[s].map(lambda b: tokenize(b, tok), batched=True)

    # get logits/labels
    vL, vY = logits_labels(mdl, ds["validation"], tok, args.batch_size)
    tL, tY = logits_labels(mdl, ds["test"], tok, args.batch_size)

    # per-class thresholds from validation
    base = sweep_per_class(vL, vY)

    # hybrid shrink toward 0.5 to reduce overfit
    alphas = [0.25, 0.5, 0.75, 1.0]
    def macro_f1(L, Y, th):
        p = 1/(1+np.exp(-L)); yhat = (p>=th).astype(int)
        return f1_score(Y, yhat, average="macro", zero_division=0)
    best_a, best_m = 1.0, -1.0
    for a in alphas:
        th = a*base + (1-a)*0.5
        m = macro_f1(vL, vY, th)
        if m > best_m: best_m, best_a = m, a
    th_final = best_a*base + (1-best_a)*0.5
    print(f"[retune] selected alpha={best_a} (VAL macro-F1={best_m:.4f})")

    # save thresholds and report
    with open(model_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump([float(x) for x in th_final], f, indent=2)
    report("VAL ", vL, vY, th_final)
    report("TEST", tL, tY, th_final)

if __name__ == "__main__":
    main()
