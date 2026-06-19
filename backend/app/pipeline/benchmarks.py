"""Benchmark tables served to the UI — backed by *measured* results only.

Earlier versions of this file hard-coded the synopsis target figures. Those
were projections, not measurements, and must never be presented as
experimental results in a paper. This module now reads whatever the evaluation
harness actually measured (``backend/results/benchmarks.json``). If the
harness has not been run, it returns an explicit "not measured" structure so
the UI shows honest empty tables rather than invented numbers.

To populate it:  python -m app.eval.run_all
"""
from __future__ import annotations

from ..eval.results_store import load_benchmarks

_NOT_MEASURED_TABLE = {
    "rows": [],
    "best_row": None,
    "measured": False,
}


def _empty_payload() -> dict:
    return {
        "status": "not_measured",
        "disclaimer": (
            "No measured results found. Run `python -m app.eval.run_all` to "
            "populate backend/results/benchmarks.json. The UI will not display "
            "any figure that has not been measured."
        ),
        "ocr": {**_NOT_MEASURED_TABLE, "title": "OCR Model Performance (not measured)",
                "columns": ["Model", "WER %", "CER %", "Inference (ms)", "Notes"]},
        "detection": {**_NOT_MEASURED_TABLE, "title": "Text Region Detection (not measured)",
                      "columns": ["Model", "mAP@0.5", "mAP@0.5:0.95", "Inference (ms)"]},
        "ner": {**_NOT_MEASURED_TABLE, "title": "NER Model Comparison (not measured)",
                "columns": ["Model", "Precision", "Recall", "F1", "Notes"]},
        "system": {**_NOT_MEASURED_TABLE, "title": "End-to-End System Performance (not measured)",
                   "columns": ["Metric", "Value", "Notes"]},
    }


def get_benchmarks() -> dict:
    """Return measured benchmarks, or an honest 'not measured' payload."""
    measured = load_benchmarks()
    return measured if measured is not None else _empty_payload()
