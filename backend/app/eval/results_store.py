"""Where measured benchmark results live, and how they are read back.

The evaluators write ``backend/results/benchmarks.json``. The API reads it via
``load_benchmarks()``. If the file is absent (evaluators never run), callers
get ``None`` and must report the metrics as *not measured* — never a fake
number.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import BACKEND_DIR

RESULTS_DIR = BACKEND_DIR / "results"
BENCHMARKS_PATH = RESULTS_DIR / "benchmarks.json"


def save_benchmarks(data: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARKS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return BENCHMARKS_PATH


def load_benchmarks() -> dict | None:
    if not BENCHMARKS_PATH.exists():
        return None
    try:
        return json.loads(BENCHMARKS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
