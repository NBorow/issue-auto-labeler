# Issue Auto-Labeler
Auto-label issues

---
Multi-label text classification with a fine-tuned [RoBERTa-base](https://huggingface.co/roberta-base).  
Trained on issue titles + bodies to predict multiple labels (bug, enhancement, question, etc.).
---

▶ Try it live: [Hugging Face Space Demo](https://huggingface.co/spaces/NBorow/issue-auto-labeler-demo)

## Features
- **Multi-label outputs** (probabilities per label, not softmax — scores do not sum to 1).
- **Threshold tuning** (`thresholds.json`) for better precision/recall trade-offs.
- **Inference script** (`src/predict.py`) for CLI usage.
- **Training script** (`src/train.py`) for full retraining with class weights.
- **Hugging Face Space** ready (`app.py` + Gradio UI).


## Model Performance (NEWBEST)

Trained model: RoBERTa-base (multi-label, issue titles + bodies)  
Epochs: 12 · Batch size: 32 · LR: 1.2e-5 · Seed: 4 · Class weights: enabled  

```
| Setting                         | Split       | Micro-F1 | Macro-F1 | Weighted-F1 | Samples-F1 | Subset Acc |
|---------------------------------|-------------|----------|----------|-------------|------------|------------|
| **Baseline (global = 0.5)**     | Validation  | 0.645    | 0.456    | 0.749       | 0.702      | 0.375      |
|                                 | Test        | 0.701    | 0.440    | 0.814       | 0.789      | 0.584      |
| **Validation-tuned per-label**  | Validation  | 0.802    | 0.717    | 0.849       | 0.817      | 0.682      |
|                                 | Test        | 0.823    | 0.609    | 0.887       | 0.858      | 0.764      |
```

## Project Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/issue-auto-labeler.git
cd issue-auto-labeler
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
 ```

 ### 3. Install dependencies
```bash
pip install -U pip
pip install -r requirements.txt
 ```

 ## Repository structure
 ```
├─ src/
│  ├─ train.py          # Training entrypoint
│  └─ predict.py        # CLI inference
├─ app.py               # Gradio UI for Hugging Face Space
├─ requirements.txt     # Inference deps for Spaces
├─ preapre_jsonl.py     # utility to preapre JSONL files for multi-label classification
└─ README.md
```

Fine-tuned model artifacts are uploaded in the Hugging Face repo:
https://huggingface.co/NBorow/issue-auto-labeler

## Usage

### Local inference
```bash
python src/predict.py --model_dir models/NEWBEST --title "Title here" --body "Body here"
```

 #### License
 Code: MIT License (open for reuse).