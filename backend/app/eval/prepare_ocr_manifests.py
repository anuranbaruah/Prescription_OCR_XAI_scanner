"""Turn the bundled handwritten datasets into OCR-eval manifests.

The project ships two real **word-level handwritten** medicine-name datasets
under ``<project>/dataset``. This script writes a ``manifest.csv`` (image,text)
for each one's **test split**, so the OCR evaluator can run on genuine
handwriting:

    python -m app.eval.prepare_ocr_manifests

Then evaluate either dataset (optionally subsampled for speed):

    set RXAI_EVAL_OCR_DIR=../dataset/Handwritten Rx
    set RXAI_EVAL_OCR_LIMIT=200
    python -m app.eval.run_ocr_eval
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..config import BACKEND_DIR

PROJECT_ROOT = BACKEND_DIR.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"


def _write_manifest(out_dir: Path, rows: list[dict]) -> Path:
    out = out_dir / "manifest.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "text"])
        writer.writeheader()
        writer.writerows(rows)
    return out


def prepare_handwritten_rx() -> tuple[Path, int] | None:
    """Handwritten Rx: Test_Label.csv (Images,Text) + Test_Set/*.jpg."""
    root = DATASET_ROOT / "Handwritten Rx"
    label_csv = root / "Test_Label.csv"
    img_dir = root / "Test_Set"
    if not (label_csv.exists() and img_dir.exists()):
        return None
    rows = []
    with label_csv.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            name = (r.get("Images") or "").strip()
            text = (r.get("Text") or "").strip()
            if name and text and (img_dir / name).exists():
                rows.append({"image": f"Test_Set/{name}", "text": text})
    return _write_manifest(root, rows), len(rows)


def prepare_bd() -> tuple[Path, int] | None:
    """Doctor's Handwritten Prescription BD: testing_labels.csv + testing_words/."""
    # The folder name uses a typographic apostrophe (U+2019); match by glob.
    matches = list(DATASET_ROOT.glob("*Handwritten Prescription BD*"))
    if not matches:
        return None
    test_dir = matches[0] / "Testing"
    label_csv = test_dir / "testing_labels.csv"
    img_dir = test_dir / "testing_words"
    if not (label_csv.exists() and img_dir.exists()):
        return None
    rows = []
    with label_csv.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            name = (r.get("IMAGE") or "").strip()
            text = (r.get("MEDICINE_NAME") or "").strip()
            if name and text and (img_dir / name).exists():
                rows.append({"image": f"testing_words/{name}", "text": text})
    return _write_manifest(test_dir, rows), len(rows)


def main() -> None:
    print(f"Dataset root: {DATASET_ROOT}")
    for label, fn in [
        ("Handwritten Rx", prepare_handwritten_rx),
        ("Doctor's Handwritten Prescription BD", prepare_bd),
    ]:
        result = fn()
        if result is None:
            print(f"  [skip] {label}: expected files not found")
            continue
        path, n = result
        print(f"  [ok]   {label}: {n} test images -> {path}")
        print(f"         RXAI_EVAL_OCR_DIR={path.parent}")


if __name__ == "__main__":
    main()
