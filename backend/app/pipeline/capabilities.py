"""Detect which optional heavy dependencies are installed at runtime.

Every stage degrades gracefully: if a dependency is missing the pipeline
still runs and the report flags the capability as unavailable. This keeps the
app usable while the (large) ML wheels download/install.
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

from ..config import settings


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@lru_cache
def probe_capabilities() -> dict[str, bool]:
    have_torch = _have("torch")
    return {
        "torch": have_torch,
        # Vision LLM (Groq/OpenRouter/OpenAI/...) needs the SDK + an API key.
        "vision": (
            settings.enable_vision
            and _have("openai")
            and bool(settings.resolved_vision_key)
        ),
        "trocr": have_torch and _have("transformers") and settings.enable_trocr,
        "easyocr": _have("easyocr") and settings.enable_easyocr,
        "tesseract": _have("pytesseract") and settings.enable_tesseract,
        "yolo": _have("ultralytics") and Path(settings.yolo_weights).exists(),
        "ner": have_torch and _have("transformers"),
        "shap": _have("shap"),
        "lime": _have("lime"),
        "symspell": _have("symspellpy"),
        "rapidfuzz": _have("rapidfuzz"),
    }
