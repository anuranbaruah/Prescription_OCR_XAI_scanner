"""OCR evaluation — measured WER/CER/latency per engine on a labeled image set.

Runs every OCR engine that is actually installed (TrOCR, EasyOCR, Tesseract)
over the OCR manifest and reports corpus WER/CER and mean per-image latency.
No engine is reported unless it ran; nothing is hard-coded.

Run:  python -m app.eval.run_ocr_eval
"""
from __future__ import annotations

import os

from ..utils import imread_unicode
from ..pipeline import ocr, preprocessing
from ..pipeline.capabilities import probe_capabilities
from ..pipeline.detection import Region
from .datasets import load_ocr_samples
from .metrics import corpus_wer_cer


def _word_accuracy(pairs: list[tuple[str, str]]) -> float:
    """Exact-match accuracy (case/space-insensitive) — the standard metric for
    word-level handwritten recognition datasets."""
    if not pairs:
        return 0.0
    hits = sum(
        1 for ref, hyp in pairs
        if " ".join(ref.lower().split()) == " ".join(hyp.lower().split())
    )
    return hits / len(pairs)

# (engine label, capability key, runner) — runner takes (bgr, gray) -> (text, ms)
def _trocr_run(bgr, gray):
    out = ocr.run_trocr([Region((0, 0, bgr.shape[1], bgr.shape[0]), 1.0, bgr)])
    return out.text, out.inference_ms


def _easyocr_run(bgr, gray):
    out = ocr.run_easyocr(bgr)
    return out.text, out.inference_ms


def _tesseract_run(bgr, gray):
    out = ocr.run_tesseract(gray)
    return out.text, out.inference_ms


_ENGINES = [
    ("TrOCR (pretrained)", "trocr", _trocr_run, "ViT+RoBERTa, handwritten checkpoint"),
    ("EasyOCR", "easyocr", _easyocr_run, "CRNN, multi-language"),
    ("Tesseract v5", "tesseract", _tesseract_run, "Open-source baseline"),
]


def evaluate() -> dict:
    samples = load_ocr_samples()
    caps = probe_capabilities()

    rows: list[list] = []
    best_idx = None
    best_wer = float("inf")

    if samples:
        # Preprocess each image once and reuse across engines.
        prepared = []
        for s in samples:
            bgr = imread_unicode(s.image_path)
            if bgr is None:
                continue
            gray = preprocessing.preprocess(bgr).gray
            prepared.append((s.text, bgr, gray))

        for label, cap_key, runner, note in _ENGINES:
            if not caps.get(cap_key):
                continue
            pairs: list[tuple[str, str]] = []
            latencies: list[float] = []
            errors = 0
            for ref_text, bgr, gray in prepared:
                try:
                    hyp, ms = runner(bgr, gray)
                except Exception as exc:  # noqa: BLE001 — record as failure, keep going
                    hyp, ms = "", 0.0
                    errors += 1
                    note = f"unavailable: {exc}"[:60]
                pairs.append((ref_text, hyp))
                latencies.append(ms)
            # An engine that hard-failed on every image isn't really installed
            # (e.g. pytesseract present but no system binary) — don't report a
            # misleading 100% WER for it.
            if not pairs or errors == len(pairs):
                continue
            wer, cer = corpus_wer_cer(pairs)
            acc = _word_accuracy(pairs)
            mean_ms = round(sum(latencies) / len(latencies), 1)
            rows.append([
                label, round(acc * 100, 2), round(wer * 100, 2),
                round(cer * 100, 2), mean_ms, note,
            ])
            # Rank by CER — more stable than WER on single-word references.
            if cer < best_wer:
                best_wer = cer
                best_idx = len(rows) - 1

    return {
        "title": "OCR Model Performance (measured)",
        "columns": ["Model", "Word Acc %", "WER %", "CER %", "Inference (ms)", "Notes"],
        "rows": rows,
        "best_row": best_idx,
        "measured": bool(rows),
        "n_samples": len(samples),
        "source": _source_label(),
    }


def _source_label() -> str:
    """Human-readable name of the OCR dataset being evaluated."""
    src = os.environ.get("RXAI_EVAL_OCR_DIR")
    if not src:
        return "demo (printed)"
    parts = [p for p in src.replace("\\", "/").rstrip("/").split("/") if p]
    # Generic split folders aren't descriptive on their own — use the parent.
    generic = {"testing", "test", "test_set", "validation", "val"}
    if parts and parts[-1].lower() in generic and len(parts) >= 2:
        return f"{parts[-2]} ({parts[-1]})"
    return parts[-1] if parts else "demo (printed)"


def main() -> None:
    res = evaluate()
    print(res["title"], f"— n={res['n_samples']} source={res.get('source')}")
    if not res["rows"]:
        print("  (no OCR samples or no engines available)")
        return
    for i, row in enumerate(res["rows"]):
        mark = " *" if i == res["best_row"] else "  "
        print(f"{mark} {row[0]:<22} Acc={row[1]:>6}%  WER={row[2]:>7}%  CER={row[3]:>6}%  {row[4]:>7}ms")


if __name__ == "__main__":
    main()
