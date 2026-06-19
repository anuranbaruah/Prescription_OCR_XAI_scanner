"""Dataset manifest loaders for the evaluation harness.

Each evaluator reads a small, explicit manifest format so you can point it at
either the bundled demo gold set (``backend/data/eval/...``) or the full
public datasets once downloaded (see ``EVALUATION.md``). Override the location
of any set with an environment variable (``RXAI_EVAL_*``); otherwise the
bundled demo set under ``data/eval`` is used.

Manifest formats
----------------
OCR          ``<dir>/manifest.csv`` with columns ``image,text``
             (``image`` is a path relative to the manifest's directory;
             ``text`` is the ground-truth transcription).
NER          ``<dir>/labeled.jsonl`` — one JSON object per line:
             ``{"text": "...", "drugs": ["Augmentin 625", ...]}``
Recommend    ``<dir>/cases.jsonl`` — ``{"brand": "...", "expected_generic": "..."}``
Interactions ``<dir>/pairs.jsonl`` — ``{"drugs": ["A", "B"], "interacts": true}``
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..config import DATA_DIR

EVAL_DIR = DATA_DIR / "eval"


def _resolve(env_var: str, default_subdir: str) -> Path:
    override = os.environ.get(env_var)
    return Path(override) if override else EVAL_DIR / default_subdir


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
@dataclass
class OCRSample:
    image_path: Path
    text: str


def load_ocr_samples() -> list[OCRSample]:
    """Read ``manifest.csv`` (image,text). Skips rows whose image is missing.

    Set ``RXAI_EVAL_OCR_LIMIT=N`` to evaluate on a fixed-seed random subsample
    of N images (useful for a quick but representative run on large test sets).
    """
    import csv
    import random

    root = _resolve("RXAI_EVAL_OCR_DIR", "ocr")
    manifest = root / "manifest.csv"
    if not manifest.exists():
        return []
    samples: list[OCRSample] = []
    with manifest.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            img = (root / row["image"]).resolve()
            if img.exists():
                samples.append(OCRSample(img, row["text"].strip()))

    limit = os.environ.get("RXAI_EVAL_OCR_LIMIT")
    if limit and limit.isdigit() and 0 < int(limit) < len(samples):
        samples = random.Random(42).sample(samples, int(limit))
    return samples


# --------------------------------------------------------------------------- #
# NER
# --------------------------------------------------------------------------- #
@dataclass
class NERSample:
    text: str
    drugs: list[str]


def load_ner_samples() -> list[NERSample]:
    root = _resolve("RXAI_EVAL_NER_DIR", "ner")
    path = root / "labeled.jsonl"
    return [
        NERSample(obj["text"], [d.strip() for d in obj.get("drugs", [])])
        for obj in _read_jsonl(path)
    ]


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
@dataclass
class RecommendSample:
    brand: str
    expected_generic: str


def load_recommend_samples() -> list[RecommendSample]:
    root = _resolve("RXAI_EVAL_RECOMMEND_DIR", "recommend")
    path = root / "cases.jsonl"
    return [
        RecommendSample(obj["brand"].strip(), obj["expected_generic"].strip())
        for obj in _read_jsonl(path)
    ]


# --------------------------------------------------------------------------- #
# Interactions
# --------------------------------------------------------------------------- #
@dataclass
class InteractionSample:
    drugs: list[str]
    interacts: bool


def load_interaction_samples() -> list[InteractionSample]:
    root = _resolve("RXAI_EVAL_INTERACTION_DIR", "interactions")
    path = root / "pairs.jsonl"
    return [
        InteractionSample([d.strip() for d in obj["drugs"]], bool(obj["interacts"]))
        for obj in _read_jsonl(path)
    ]


# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
