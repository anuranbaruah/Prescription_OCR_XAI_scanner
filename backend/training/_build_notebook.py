"""Builds finetune_trocr_colab.ipynb from cell definitions (valid JSON guaranteed)."""
import json
from pathlib import Path

C = []  # (kind, source)


def md(s): C.append(("markdown", s))
def code(s): C.append(("code", s))


md("""# Fine-tuning TrOCR for Handwritten Prescription Recognition

Fine-tunes `microsoft/trocr-base-handwritten` on the project's two word-level
handwritten medicine-name datasets (**Handwritten Rx** and **Doctor's
Handwritten Prescription BD**). The pretrained baseline measured by the eval
harness is ~10–13 % exact word accuracy; this notebook produces a fine-tuned
checkpoint to beat it.

**Runtime:** Colab → *Runtime → Change runtime type → GPU* (T4 is enough).

**Output:** a `trocr-rx-finetuned/` folder (model + processor). Download it,
then point the app/eval harness at it:
```
set RXAI_TROCR_MODEL=C:\\path\\to\\trocr-rx-finetuned
python -m app.eval.run_ocr_eval        # re-measure with the same metrics
```
""")

md("""## 1. GPU check + install dependencies""")
code("""import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
!pip -q install "transformers>=4.40" "datasets>=2.18" evaluate jiwer accelerate sentencepiece""")

md("""## 2. Get the data into Colab

Upload the project's `dataset/` folder so `DATA_ROOT` contains the two dataset
folders (`Handwritten Rx` and `Doctor's Handwritten Prescription BD dataset`).

**Option A — Google Drive (recommended).** Zip your local `dataset/` folder,
put it in Drive, then mount and unzip:
```python
from google.colab import drive; drive.mount('/content/drive')
!unzip -q "/content/drive/MyDrive/dataset.zip" -d /content/
```

**Option B — direct upload.** Run the cell below and pick your `dataset.zip`.""")
code("""# Option B: upload dataset.zip from your machine (skip if you used Drive)
import os
if not os.path.exists('/content/dataset'):
    try:
        from google.colab import files
        up = files.upload()                      # choose dataset.zip
        zname = next(iter(up))
        !unzip -q "$zname" -d /content/
    except Exception as e:
        print("Upload skipped / failed:", e)
!ls /content/dataset""")

md("""## 3. Configuration""")
code('''# ============================ CONFIG ============================
DATA_ROOT    = "/content/dataset"     # folder holding the dataset folders
DATASET      = "both"                 # "rx" | "bd" | "both"
BASE_MODEL   = "microsoft/trocr-base-handwritten"
OUTPUT_DIR   = "/content/trocr-rx-finetuned"
MAX_LEN      = 32                     # medicine names are short
EPOCHS       = 8
BATCH        = 8                      # lower to 4 if you hit OOM
GRAD_ACCUM   = 2
LR           = 4e-5
VAL_FRACTION = 0.1                    # val split for datasets lacking one (Rx)
SEED         = 42
# ===============================================================''')

md("""## 4. Load the two datasets

Both are word-level (one handwritten medicine name per image). The loaders
mirror the project's `prepare_ocr_manifests.py` so the splits match the eval
harness. Handwritten Rx has no validation split, so a fraction of its train set
is held out for validation.""")
code('''import csv, random
from pathlib import Path

def _rx_split(root, split):
    base = Path(root) / "Handwritten Rx"
    label = "Train_Label.csv" if split == "train" else "Test_Label.csv"
    img_dir = "Train_Set" if split == "train" else "Test_Set"
    rows = []
    fp = base / label
    if not fp.exists():
        return rows
    with open(fp, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = (r.get("Images") or "").strip()
            text = (r.get("Text") or "").strip()
            p = base / img_dir / name
            if name and text and p.exists():
                rows.append((str(p), text))
    return rows

def _bd_split(root, split):
    folder = {"train": "Training", "val": "Validation", "test": "Testing"}[split]
    words  = {"train": "training_words", "val": "validation_words", "test": "testing_words"}[split]
    labels = {"train": "training_labels.csv", "val": "validation_labels.csv", "test": "testing_labels.csv"}[split]
    matches = list(Path(root).glob("*Handwritten Prescription BD*"))
    if not matches:
        return []
    base = matches[0] / folder
    rows = []
    fp = base / labels
    if not fp.exists():
        return rows
    with open(fp, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = (r.get("IMAGE") or "").strip()
            text = (r.get("MEDICINE_NAME") or "").strip()
            p = base / words / name
            if name and text and p.exists():
                rows.append((str(p), text))
    return rows

def build_splits():
    rng = random.Random(SEED)
    train, val, test = [], [], []
    if DATASET in ("rx", "both"):
        rx = _rx_split(DATA_ROOT, "train"); rng.shuffle(rx)
        k = int(len(rx) * VAL_FRACTION)
        val += rx[:k]; train += rx[k:]
        test += _rx_split(DATA_ROOT, "test")
    if DATASET in ("bd", "both"):
        train += _bd_split(DATA_ROOT, "train")
        val   += _bd_split(DATA_ROOT, "val")
        test  += _bd_split(DATA_ROOT, "test")
    return train, val, test

train_rows, val_rows, test_rows = build_splits()
print(f"{len(train_rows)} train | {len(val_rows)} val | {len(test_rows)} test")
assert train_rows, "No training data found — check DATA_ROOT / unzip step."
print("examples:", train_rows[:3])''')

md("""## 5. Dataset + processor""")
code('''import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained(BASE_MODEL)

class RxWordDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        path, text = self.rows[i]
        image = Image.open(path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = processor.tokenizer(
            text, padding="max_length", max_length=MAX_LEN, truncation=True
        ).input_ids
        labels = [t if t != processor.tokenizer.pad_token_id else -100 for t in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

train_ds = RxWordDataset(train_rows)
val_ds   = RxWordDataset(val_rows)
test_ds  = RxWordDataset(test_rows)''')

md("""## 6. Model + generation config""")
code('''from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)
# Required generation settings for fine-tuning a TrOCR encoder-decoder.
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.max_length = MAX_LEN
model.config.num_beams = 4
model.config.early_stopping = True''')

md("""## 7. Metrics — CER, WER, exact word accuracy

Same metrics the eval harness reports, so before/after numbers are comparable.""")
code('''import evaluate
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = [[t if t != -100 else processor.tokenizer.pad_token_id for t in seq]
                 for seq in pred.label_ids]
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    pred_str = [p.strip() for p in pred_str]
    label_str = [l.strip() for l in label_str]
    safe_p = [p or " " for p in pred_str]
    safe_l = [l or " " for l in label_str]
    cer = cer_metric.compute(predictions=safe_p, references=safe_l)
    wer = wer_metric.compute(predictions=safe_p, references=safe_l)
    acc = sum(p.lower() == l.lower() for p, l in zip(pred_str, label_str)) / len(pred_str)
    return {"cer": cer, "wer": wer, "word_acc": acc}''')

md("""## 8. Train

`load_best_model_at_end` keeps the lowest-CER checkpoint. Handles the
`eval_strategy`/`evaluation_strategy` rename across transformers versions.""")
code('''from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

common = dict(
    output_dir=OUTPUT_DIR,
    predict_with_generate=True,
    generation_max_length=MAX_LEN,
    generation_num_beams=4,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    warmup_ratio=0.1,
    fp16=True,
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    report_to="none",
    seed=SEED,
)
try:
    args = Seq2SeqTrainingArguments(eval_strategy="epoch", save_strategy="epoch", **common)
except TypeError:
    args = Seq2SeqTrainingArguments(evaluation_strategy="epoch", save_strategy="epoch", **common)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()''')

md("""## 9. Evaluate on the held-out test split(s)""")
code('''val_metrics = trainer.evaluate(metric_key_prefix="val")
test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
print("VALIDATION:", {k: round(v, 4) for k, v in val_metrics.items() if isinstance(v, float)})
print("TEST      :", {k: round(v, 4) for k, v in test_metrics.items() if isinstance(v, float)})''')

md("""## 10. Save + download the fine-tuned checkpoint""")
code('''import shutil
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Saved model + processor to", OUTPUT_DIR)

shutil.make_archive(OUTPUT_DIR, "zip", OUTPUT_DIR)
print("Zipped:", OUTPUT_DIR + ".zip")
try:
    from google.colab import files
    files.download(OUTPUT_DIR + ".zip")
except Exception as e:
    print("Download manually from the Files panel:", e)''')

md("""## 11. Plug it into the project

1. Unzip `trocr-rx-finetuned.zip` somewhere on your machine.
2. Point the app + eval harness at it and re-measure:
   ```bat
   cd backend
   set RXAI_TROCR_MODEL=C:\\path\\to\\trocr-rx-finetuned
   set RXAI_EVAL_OCR_DIR=../dataset/Handwritten Rx
   python -m app.eval.run_ocr_eval
   ```
   The OCR table now reflects the fine-tuned model — compare word accuracy / CER
   against the pretrained baseline already in `results/benchmarks.json`.
3. To make it the app default, set `RXAI_TROCR_MODEL` in `backend/.env`.

This before/after comparison (pretrained vs fine-tuned, on both datasets) is the
core OCR result for the paper.""")

nb = {
    "cells": [
        {"cell_type": k, "metadata": {}, "source": s.splitlines(keepends=True)}
        | ({"outputs": [], "execution_count": None} if k == "code" else {})
        for k, s in C
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).parent / "finetune_trocr_colab.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", out, "with", len(C), "cells")
