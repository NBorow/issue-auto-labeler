#!/usr/bin/env python3
"""
Prepare GitHub issue JSONL files for multi-label classification.

- Reads data/raw/*.jsonl (configurable)
- Builds `text` = title + "\n\n" + body
- Extracts labels from `labels[].name`, normalizes and removes noise labels
- Filters labels by minimum frequency
- Splits into train/val/test
- Writes JSONL splits + label maps + stats

Usage:
  python prepare_jsonl.py \
    --input_glob "data/raw/*.jsonl" \
    --out_dir "data/processed" \
    --min_label_freq 50 \
    --seed 42
"""

import argparse
import glob
import json
import os
import random
import re
from collections import Counter

# ---- label normalization / noise ----

NOISE_PREFIXES = [
    "status:", "state:", "type:", "domain:", "area:", "component:", "pkg:",
    "package:", "topic:", "module:", "priority:", "ci:", "team:", "kind:",
    "os:", "platform:", "size:", "difficulty:", "affects:", "needs:",
    "good first issue:"
]

NOISE_LABELS = set([
    "good first issue", "good-first-issue", "help wanted", "needs triage",
    "triage", "stale", "invalid", "duplicate", "wontfix", "wont fix",
    "automerge", "do not merge", "do-not-merge"
])

PREFIX_RE = re.compile(
    r"^(?:{})(.*)$".format("|".join([re.escape(p) for p in NOISE_PREFIXES])),
    re.IGNORECASE
)

def normalize_label(name: str) -> str:
    if not name:
        return ""
    s = name.strip()
    # remove some emoji / odd symbols (quick & dirty)
    s = re.sub(r"[\u2600-\u27BF]", "", s)
    s = s.replace("_", " ").replace("-", " ").strip()
    # strip common prefixes (type:, domain:, etc.)
    m = PREFIX_RE.match(s)
    if m:
        s = m.group(1).strip()
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s

def extract_labels(obj: dict) -> list[str]:
    labs = obj.get("labels", [])
    names: list[str] = []
    if isinstance(labs, list):
        for it in labs:
            if isinstance(it, dict):
                nm = it.get("name") or it.get("label") or ""
                nm = normalize_label(nm)
                if nm:
                    names.append(nm)
            elif isinstance(it, str):
                nm = normalize_label(it)
                if nm:
                    names.append(nm)
    # unique per item & drop noise labels
    names = list(dict.fromkeys(names))
    names = [n for n in names if n not in NOISE_LABELS]
    return names

def item_text(obj: dict) -> tuple[str, str]:
    title = obj.get("title") or obj.get("issue_title") or ""
    body = obj.get("body") or ""
    url  = obj.get("html_url") or obj.get("issue_url") or ""
    text = (title.strip() + "\n\n" + body.strip()).strip()
    return text, url

def is_pull_request(obj: dict) -> bool:
    # GitHub issues that are PRs contain a "pull_request" object
    return isinstance(obj.get("pull_request"), dict)

# ---- io helpers ----

def iter_jsonl(paths: list[str]):
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # skip malformed line
                    continue

def write_jsonl(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", type=str, default="data/raw/*.jsonl")
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--min_label_freq", type=int, default=50,
                    help="Drop labels occurring fewer than this many times.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--drop_pull_requests", action="store_true",
                    help="Skip items that are PRs (not plain issues).")
    ap.add_argument("--max_items", type=int, default=0,
                    help="Optional cap for quick experimentation (0=all).")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.input_glob}")

    # 1) load + basic cleaning
    rng = random.Random(args.seed)
    items = []
    seen_urls = set()

    for obj in iter_jsonl(paths):
        if args.drop_pull_requests and is_pull_request(obj):
            continue
        text, url = item_text(obj)
        if not text:
            continue
        labels = extract_labels(obj)
        if not labels:
            continue

        # dedupe by URL if available, else by (title+body) fingerprint
        key = url or (text[:200])
        if key in seen_urls:
            continue
        seen_urls.add(key)

        items.append({"text": text, "labels": labels, "url": url})

        if args.max_items and len(items) >= args.max_items:
            break

    if not items:
        raise SystemExit("No usable items found. Check your inputs.")

    # 2) label frequency filter
    freq = Counter()
    for it in items:
        for l in it["labels"]:
            freq[l] += 1

    keep = {l for l, c in freq.items() if c >= args.min_label_freq}
    if not keep:
        raise SystemExit(
            f"No labels meet min_label_freq={args.min_label_freq}. "
            f"Try lowering it."
        )

    filtered = []
    for it in items:
        kept = [l for l in it["labels"] if l in keep]
        if kept:
            filtered.append({"text": it["text"], "labels": kept, "url": it["url"]})

    items = filtered
    if not items:
        raise SystemExit("All items dropped after label filtering.")

    # rebuild label set & maps
    labels_sorted = sorted({l for it in items for l in it["labels"]})
    label_to_id = {l: i for i, l in enumerate(labels_sorted)}

    # 3) split
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)
    train = items[:n_train]
    val   = items[n_train:n_train + n_val]
    test  = items[n_train + n_val:]

    # ensure each label appears in train (best-effort fix)
    present = {l for it in train for l in it["labels"]}
    missing = [l for l in labels_sorted if l not in present]
    if missing:
        pool = val + test
        rng.shuffle(pool)
        for l in list(missing):
            for i, it in enumerate(pool):
                if l in it["labels"]:
                    train.append(it)
                    pool.pop(i)
                    present.update(it["labels"])
                    break
        # re-split pool back into val/test (keep sizes approx)
        n_pool = len(pool)
        val = pool[:n_val]
        test = pool[n_val:]

    # 4) write outputs
    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "val.jsonl"),   val)
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"),  test)

    with open(os.path.join(args.out_dir, "label_to_id.json"), "w", encoding="utf-8") as f:
        json.dump(label_to_id, f, ensure_ascii=False, indent=2)

    # quick stats
    def counts(rows):
        c = Counter()
        for it in rows:
            c.update(it["labels"])
        return dict(sorted(c.items(), key=lambda x: -x[1]))

    stats = {
        "num_items_total": len(items),
        "num_train": len(train),
        "num_val": len(val),
        "num_test": len(test),
        "num_labels": len(labels_sorted),
        "labels": labels_sorted,
        "label_counts_total": counts(items),
        "label_counts_train": counts(train),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[ok] Wrote splits to: {args.out_dir}")
    print(f"  labels ({len(labels_sorted)}): {labels_sorted}")
    print(f"  train/val/test: {len(train)}/{len(val)}/{len(test)}")

if __name__ == "__main__":
    main()
