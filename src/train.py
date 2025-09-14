# src/train.py
import argparse
import json
import inspect
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
    get_linear_schedule_with_warmup,
)

# ============= Utilities =============

def ensure_splits(data_dir: Path) -> Dict[str, Path]:
    files = {
        "train": data_dir / "train.jsonl",
        "validation": data_dir / "val.jsonl",
        "test": data_dir / "test.jsonl",
    }
    if not files["validation"].exists():
        alt = data_dir / "valid.jsonl"
        if alt.exists():
            files["validation"] = alt
    for k, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {k} split at {p}")
    return files

def load_label_maps(data_dir: Path) -> Dict[str, Dict]:
    l2i = data_dir / "label_to_id.json"
    if not l2i.exists():
        raise FileNotFoundError(f"Expected label map at {l2i}.")
    with open(l2i, "r", encoding="utf-8") as f:
        label_to_id: Dict[str, int] = json.load(f)
    id_to_label = {int(v): k for k, v in label_to_id.items()}
    return {"label_to_id": label_to_id, "id_to_label": id_to_label}

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def build_compute_metrics(threshold: float = 0.5):
    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_prob = _sigmoid(logits)
        y_pred = (y_prob >= threshold).astype(int)
        f1_micro   = f1_score(y_true, y_pred, average="micro",  zero_division=0)
        f1_macro   = f1_score(y_true, y_pred, average="macro",  zero_division=0)
        f1_samples = f1_score(y_true, y_pred, average="samples", zero_division=0)
        pr, rc, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        subset_acc = (y_pred == y_true).all(axis=1).mean()
        return {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_w,
            "f1_samples": f1_samples,
            "subset_accuracy": subset_acc,
        }
    return compute_metrics

def compute_pos_weight(train_multi_hot: np.ndarray, clip_max: Optional[float] = None) -> torch.Tensor:
    pos = train_multi_hot.sum(axis=0)  # [C]
    N = train_multi_hot.shape[0]
    pw = np.zeros_like(pos, dtype=np.float32)
    has_pos = pos > 0
    pw[has_pos] = (N - pos[has_pos]) / pos[has_pos]
    t = torch.tensor(pw, dtype=torch.float32)
    if clip_max is not None:
        t = torch.clamp(t, max=float(clip_max))
    return t

class MultiLabelTrainer(Trainer):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight

    # accept extra kwargs across transformers versions
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def trainer_supported() -> bool:
    try:
        import accelerate  # noqa: F401
    except Exception:
        return False
    try:
        inspect.signature(TrainingArguments.__init__)
        return True
    except Exception:
        return False

def make_training_args(out_dir: Path, epochs: float, batch_size: int, lr: float, seed: int,
                       grad_accum: int, warmup_ratio: float, fp16: bool):
    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())
    base = dict(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size if "per_device_eval_batch_size" in supported else None,
        learning_rate=lr,
        seed=seed,
        gradient_accumulation_steps=grad_accum if "gradient_accumulation_steps" in supported else None,
        warmup_ratio=warmup_ratio if "warmup_ratio" in supported else None,
        fp16=fp16 if "fp16" in supported else None,
        logging_steps=50 if "logging_steps" in supported else None,
        save_total_limit=2 if "save_total_limit" in supported else None,
        report_to=["none"] if "report_to" in supported else None,
    )
    base = {k: v for k, v in base.items() if v is not None}
    if "evaluation_strategy" in supported and "save_strategy" in supported:
        base["evaluation_strategy"] = "epoch"
        base["save_strategy"] = "epoch"
        if "load_best_model_at_end" in supported: base["load_best_model_at_end"] = True
        if "metric_for_best_model" in supported:  base["metric_for_best_model"] = "f1_macro"
        if "greater_is_better" in supported:      base["greater_is_better"] = True
    else:
        if "save_strategy" in supported:          base["save_strategy"] = "no"
        if "load_best_model_at_end" in supported: base["load_best_model_at_end"] = False
        base.pop("metric_for_best_model", None)
        base.pop("greater_is_better", None)
    return TrainingArguments(**base)

def dataloaders_for_no_trainer(ds_enc, data_collator, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(ds_enc["train"], batch_size=batch_size, shuffle=True,  collate_fn=data_collator)
    val_loader   = DataLoader(ds_enc["validation"], batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    test_loader  = DataLoader(ds_enc["test"], batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    return train_loader, val_loader, test_loader

def evaluate_no_trainer(model, loader, device, threshold: float) -> Dict[str, float]:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device).float()
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_prob = 1 / (1 + np.exp(-logits))
    y_pred = (y_prob >= threshold).astype(int)
    f1_micro   = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    f1_macro   = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    f1_samples = f1_score(y_true, y_pred, average="samples", zero_division=0)
    pr, rc, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    subset_acc = (y_pred == y_true).all(axis=1).mean()
    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_w,
        "f1_samples": f1_samples,
        "subset_accuracy": subset_acc,
    }

# ---- Threshold utilities ----
def logits_labels(model, ds_split, tokenizer, batch_size=32):
    collate = DataCollatorWithPadding(tokenizer=tokenizer)
    keep = {"input_ids", "attention_mask", "labels"}
    ds_split = ds_split.remove_columns([c for c in ds_split.column_names if c not in keep])
    loader = DataLoader(ds_split, batch_size=batch_size, shuffle=False, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device).float()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            ).logits
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)

def sweep_thresholds_per_class(val_logits, val_y, grid=None):
    if grid is None:
        grid = np.linspace(0.1, 0.9, 33)
    probs = 1/(1+np.exp(-val_logits))
    C = probs.shape[1]
    best = np.full(C, 0.5, dtype=np.float32)
    for c in range(C):
        best_f1, best_t = -1.0, 0.5
        pc, yc = probs[:, c], val_y[:, c]
        for t in grid:
            f1 = f1_score(yc, (pc >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best[c] = best_t
    return best

def write_metrics_txt(path: Path, header: str, d: Dict[str, float]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(header + "\n")
        for k, v in d.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

# ============= Main =============

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="")  # auto-name under models\ if blank
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weights", action="store_true", help="Use pos_weight for BCE (imbalance)")
    ap.add_argument("--posw_clip", type=float, default=None, help="Clip pos_weight to this max (e.g., 30.0)")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--warmup_ratio", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=0.5, help="baseline decision threshold (pre-tuning)")
    ap.add_argument("--tune_batch_size", type=int, default=32, help="batch size for threshold tuning")
    return ap.parse_args()

def auto_name_dir(base_models: Path, model_name: str, epochs: float, bsz: int, lr: float,
                  use_posw: bool, posw_clip: Optional[float]) -> Path:
    tag = f"{model_name}-e{int(epochs)}-b{bsz}-lr{lr:.0e}".replace("+0", "")
    if use_posw:
        tag += "-posw"
        if posw_clip is not None:
            tag += f"-clip{int(posw_clip)}"
    tag += "-" + datetime.now().strftime("%Y%m%d-%H%M")
    return base_models / tag

def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    base_models = Path("models")
    base_models.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out_dir) if args.out_dir else auto_name_dir(
        base_models, args.model_name, args.epochs, args.batch_size, args.lr, args.class_weights, args.posw_clip
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    with open(out_dir / "CLI.txt", "w", encoding="utf-8") as f:
        f.write(
            f"python src\\train.py --data_dir {args.data_dir} --out_dir {out_dir} "
            f"--model_name {args.model_name} --epochs {args.epochs} --batch_size {args.batch_size} "
            f"--lr {args.lr} --seed {args.seed} {'--class_weights' if args.class_weights else ''} "
            f"{'--fp16' if args.fp16 else ''} --grad_accum {args.grad_accum} --warmup_ratio {args.warmup_ratio} "
            f"--threshold {args.threshold} --tune_batch_size {args.tune_batch_size} "
            f"{f'--posw_clip {args.posw_clip}' if args.posw_clip is not None else ''}\n"
        )

    files = ensure_splits(data_dir)
    maps = load_label_maps(data_dir)
    label_to_id: Dict[str, int] = maps["label_to_id"]
    id_to_label: Dict[int, str] = maps["id_to_label"]
    num_labels = len(label_to_id)
    print(f"Detected {num_labels} labels.")

    # Load JSONL
    ds = load_dataset(
        "json",
        data_files={
            "train": str(files["train"]),
            "validation": str(files["validation"]),
            "test": str(files["test"]),
        },
        split={"train": "train", "validation": "validation", "test": "test"},
    )
    for split in ["train", "validation", "test"]:
        cols = set(ds[split].column_names)
        if not {"text", "labels"}.issubset(cols):
            raise ValueError(f"{split} split must contain 'text' and 'labels'. Found: {sorted(cols)}")

    # list-of-label-names -> multi-hot
    def to_multi_hot(batch):
        mh = []
        for names in batch["labels"]:
            vec = np.zeros(num_labels, dtype=np.float32)
            for name in names:
                if name not in label_to_id:
                    raise ValueError(f"Unknown label '{name}'.")
                vec[label_to_id[name]] = 1.0
            mh.append(vec.tolist())
        return {"labels": mh}
    ds = ds.map(to_multi_hot, batched=True)

    # pos_weight
    pos_weight = None
    if args.class_weights:
        train_y = np.array(ds["train"]["labels"], dtype=np.float32)
        pos_weight = compute_pos_weight(train_y, clip_max=args.posw_clip)
        print(f"pos_weight (per class): {pos_weight.tolist()}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True)
    ds_enc = ds.map(_tok, batched=True)

    # Keep only fields we need; float labels
    for split in ["train", "validation", "test"]:
        keep = {"input_ids", "attention_mask", "labels"}
        drop = [c for c in ds_enc[split].column_names if c not in keep]
        if drop:
            ds_enc[split] = ds_enc[split].remove_columns(drop)
    def cast_labels(batch):
        return {"labels": [[float(z) for z in row] for row in batch["labels"]]}
    ds_enc = ds_enc.map(cast_labels, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id,
        problem_type="multi_label_classification",
    )

    # ---- Train with Trainer ----
    if trainer_supported():
        training_args = make_training_args(
            out_dir=out_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            seed=args.seed, grad_accum=args.grad_accum, warmup_ratio=args.warmup_ratio, fp16=args.fp16
        )
        kw = dict(
            model=model,
            args=training_args,
            train_dataset=ds_enc["train"],
            eval_dataset=ds_enc["validation"],
            data_collator=data_collator,
            compute_metrics=build_compute_metrics(args.threshold),
            pos_weight=pos_weight,
        )
        init_sig = inspect.signature(MultiLabelTrainer.__init__)
        if "tokenizer" in init_sig.parameters:
            kw["tokenizer"] = tokenizer
        elif "processing_class" in init_sig.parameters:
            kw["processing_class"] = tokenizer

        trainer = MultiLabelTrainer(**kw)
        trainer.train()
        val_metrics = trainer.evaluate(eval_dataset=ds_enc["validation"])
        test_metrics = trainer.evaluate(eval_dataset=ds_enc["test"])
        trainer.save_model(str(out_dir))
    else:
        raise RuntimeError("Trainer/accelerate required for this script.")

    # ----- Threshold tuning -----
    tuned_model = AutoModelForSequenceClassification.from_pretrained(out_dir)
    val_logits, val_y = logits_labels(tuned_model, ds_enc["validation"], tokenizer, args.tune_batch_size)
    test_logits, test_y = logits_labels(tuned_model, ds_enc["test"], tokenizer, args.tune_batch_size)
    thresholds = sweep_thresholds_per_class(val_logits, val_y)

    with open(out_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump([float(x) for x in thresholds], f, indent=2)

    def summarize(logits, y):
        probs = 1/(1+np.exp(-logits))
        y_pred = (probs >= thresholds).astype(int)
        return {
            "micro-F1":   f1_score(y, y_pred, average="micro",  zero_division=0),
            "macro-F1":   f1_score(y, y_pred, average="macro",  zero_division=0),
            "samples-F1": f1_score(y, y_pred, average="samples", zero_division=0),
            "weighted-F1": precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=0)[2],
            "subset-acc": (y_pred == y).all(axis=1).mean(),
        }

    tuned_val  = summarize(val_logits,  val_y)
    tuned_test = summarize(test_logits, test_y)

    # Write metrics
    def write_metrics_txt(path: Path, header: str, d: Dict[str, float]):
        with open(path, "a", encoding="utf-8") as f:
            f.write(header + "\n")
            for k, v in d.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

    write_metrics_txt(out_dir / "METRICS.txt", "== Baseline (threshold=%.2f) VAL ==" % args.threshold, val_metrics)
    write_metrics_txt(out_dir / "METRICS.txt", "== Baseline (threshold=%.2f) TEST ==" % args.threshold, test_metrics)
    write_metrics_txt(out_dir / "METRICS.txt", "== Tuned thresholds (per-class) VAL ==", tuned_val)
    write_metrics_txt(out_dir / "METRICS.txt", "== Tuned thresholds (per-class) TEST ==", tuned_test)

    # Console summary
    print("\n== Saved model to:", out_dir)
    print("Thresholds saved to:", out_dir / "thresholds.json")
    print("\n== Baseline VAL ==", val_metrics)
    print("== Baseline TEST ==", test_metrics)
    print("\n== Tuned VAL ==", tuned_val)
    print("== Tuned TEST ==", tuned_test)
    print("Done.")

if __name__ == "__main__":
    main()
