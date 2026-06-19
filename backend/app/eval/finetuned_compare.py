"""Add fine-tuned-TrOCR rows to the measured benchmarks (baseline vs fine-tuned).

Run this in a process where ``RXAI_TROCR_MODEL`` points at the fine-tuned
checkpoint (``results/trocr-rx-finetuned/best``). It:

  1. evaluates the fine-tuned TrOCR on each handwritten test split, using the
     SAME manifest + metrics as the baseline OCR table, and appends a
     "TrOCR (fine-tuned)" row next to the pretrained one (re-ranking by CER);
  2. re-runs the end-to-end recognition->generic metric (fine-tuned TrOCR now
     drives the OCR), adding a fine-tuned row to ``recognition_to_generic``;
  3. writes everything back to ``results/benchmarks.json``.

This keeps the baseline rows intact so the paper shows a clean before/after.

    cd backend
    set RXAI_TROCR_MODEL=%CD%\results\trocr-rx-finetuned\best
    python -m app.eval.finetuned_compare
"""
from __future__ import annotations

import os
from pathlib import Path

from ..config import BACKEND_DIR, settings
from ..utils import imread_unicode
from ..pipeline import ocr, preprocessing
from ..pipeline.detection import Region
from .datasets import load_ocr_samples
from .metrics import corpus_wer_cer
from .results_store import load_benchmarks, save_benchmarks
from .run_ocr_eval import _word_accuracy, _source_label
from . import bd_truth

FT_LABEL = "TrOCR (fine-tuned, 3ep)"
DATASET_ROOT = BACKEND_DIR.parent / "dataset"


def _dataset_dirs() -> list[str]:
    """The two handwritten test-split dirs that carry a manifest.csv."""
    dirs: list[str] = []
    rx = DATASET_ROOT / "Handwritten Rx"
    if (rx / "manifest.csv").exists():
        dirs.append(str(rx))
    for bd in DATASET_ROOT.glob("*Handwritten Prescription BD*"):
        test_dir = bd / "Testing"
        if (test_dir / "manifest.csv").exists():
            dirs.append(str(test_dir))
    return dirs


def _eval_finetuned_trocr(dataset_dir: str) -> tuple[str, list]:
    """Return (source_label, row) for the fine-tuned TrOCR on one dataset."""
    os.environ["RXAI_EVAL_OCR_DIR"] = dataset_dir
    os.environ.pop("RXAI_EVAL_OCR_LIMIT", None)
    samples = load_ocr_samples()
    source = _source_label()
    pairs: list[tuple[str, str]] = []
    latencies: list[float] = []
    for s in samples:
        bgr = imread_unicode(s.image_path)
        if bgr is None:
            continue
        out = ocr.run_trocr([Region((0, 0, bgr.shape[1], bgr.shape[0]), 1.0, bgr)])
        pairs.append((s.text, out.text))
        latencies.append(out.inference_ms)
    if not pairs:
        return source, []
    wer, cer = corpus_wer_cer(pairs)
    acc = _word_accuracy(pairs)
    mean_ms = round(sum(latencies) / len(latencies), 1)
    row = [FT_LABEL, round(acc * 100, 2), round(wer * 100, 2), round(cer * 100, 2),
           mean_ms, f"fine-tuned on {len(pairs)} test imgs (3 epochs)"]
    return source, row


def _merge_ocr_row(table: dict, row: list) -> None:
    """Replace any existing fine-tuned row, append the new one, re-rank by CER."""
    if not table or "rows" not in table:
        return
    table["rows"] = [r for r in table["rows"] if r and r[0] != FT_LABEL]
    table["rows"].append(row)
    # CER is column index 3; lowest CER wins.
    best_i, best_cer = None, float("inf")
    for i, r in enumerate(table["rows"]):
        try:
            if float(r[3]) < best_cer:
                best_cer, best_i = float(r[3]), i
        except (TypeError, ValueError, IndexError):
            continue
    table["best_row"] = best_i


def main() -> None:
    ckpt = settings.trocr_model
    if Path(ckpt) == Path("microsoft/trocr-base-handwritten") or not Path(ckpt).exists():
        raise SystemExit(
            "RXAI_TROCR_MODEL must point at the fine-tuned checkpoint dir "
            "(results/trocr-rx-finetuned/best). Currently: " + str(ckpt)
        )
    print(f"Using fine-tuned checkpoint: {ckpt}")

    b = load_benchmarks() or {}
    ocr_tables = b.get("ocr_datasets") or ([b["ocr"]] if b.get("ocr") else [])
    by_source = {t.get("source"): t for t in ocr_tables}

    print("== Per-dataset fine-tuned TrOCR ==")
    for d in _dataset_dirs():
        source, row = _eval_finetuned_trocr(d)
        if not row:
            print(f"  [{source}] no samples")
            continue
        print(f"  [{source}] {FT_LABEL}: Acc={row[1]}%  WER={row[2]}%  CER={row[3]}%")
        if source in by_source:
            _merge_ocr_row(by_source[source], row)
        # keep the primary `ocr` table in sync if it is this dataset
        if b.get("ocr", {}).get("source") == source:
            _merge_ocr_row(b["ocr"], row)

    print("== End-to-end recognition -> generic (fine-tuned TrOCR) ==")
    r2g = bd_truth.evaluate()
    if r2g.get("measured"):
        existing = b.get("recognition_to_generic", {})
        rows = [r for r in existing.get("rows", []) if r.get("engine") != FT_LABEL]
        for r in r2g["rows"]:
            if r["engine"].startswith("TrOCR"):
                rows.append({**r, "engine": FT_LABEL})
        existing["rows"] = rows
        existing["measured"] = True
        b["recognition_to_generic"] = existing
        for r in r2g["rows"]:
            if r["engine"].startswith("TrOCR"):
                print(f"  {FT_LABEL}: word={r['word_accuracy']}%  generic={r['generic_accuracy']}%")

    p = save_benchmarks(b)
    print(f"\nMerged fine-tuned rows -> {p}")


if __name__ == "__main__":
    main()
