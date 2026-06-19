"""Stage 4 — Medicine NER + DB matching (synopsis 7.4).

Two complementary strategies, merged:
  1. BioBERT token-classification (transformers) — context-aware drug spans.
  2. Dictionary fuzzy-matcher over OCR text against the medicine DB — robust to
     NER misses and OCR noise (RapidFuzz, threshold from settings).

Each candidate is fuzzy-matched to a canonical DB record so downstream stages
(recommendation, interactions) operate on real medicines.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..config import settings
from ..data.loader import MedicineDB

logger = logging.getLogger("rxai.ner")

_ner_pipe = None  # lazy singleton
_DRUG_LABEL_HINTS = ("MEDIC", "DRUG", "CHEMICAL", "PHARM")


@dataclass
class Entity:
    text: str
    score: float
    start: int
    end: int
    matched_name: str | None = None
    match_score: float | None = None


def _load_ner():
    global _ner_pipe
    if _ner_pipe is not None:
        return _ner_pipe
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline,
    )

    logger.info("Loading NER model: %s", settings.ner_model)
    tok = AutoTokenizer.from_pretrained(settings.ner_model)
    model = AutoModelForTokenClassification.from_pretrained(settings.ner_model)
    device = 0 if settings.resolved_device == "cuda" else -1
    _ner_pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tok,
        aggregation_strategy="simple",
        device=device,
    )
    return _ner_pipe


def _biobert_entities(text: str) -> list[Entity]:
    pipe = _load_ner()
    raw = pipe(text)
    ents: list[Entity] = []
    for e in raw:
        label = str(e.get("entity_group", "")).upper()
        if any(h in label for h in _DRUG_LABEL_HINTS):
            ents.append(
                Entity(
                    text=e["word"].strip(),
                    score=float(e["score"]),
                    start=int(e["start"]),
                    end=int(e["end"]),
                )
            )
    return ents


def _candidate_strings(text: str) -> list[tuple[str, int]]:
    """Words + bigrams with their char offset, for dictionary matching."""
    candidates: list[tuple[str, int]] = []
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9\-]{2,}", text):
        candidates.append((m.group(0), m.start()))
    # bigrams (e.g. "Augmentin 625")
    tokens = list(re.finditer(r"\S+", text))
    for i in range(len(tokens) - 1):
        phrase = tokens[i].group(0) + " " + tokens[i + 1].group(0)
        candidates.append((phrase, tokens[i].start()))
    return candidates


def _dictionary_entities(
    text: str, db: MedicineDB, letter_guard: bool = True
) -> list[Entity]:
    if not db.ok:
        return []
    found: dict[str, Entity] = {}
    for cand, offset in _candidate_strings(text):
        row = db.find(cand, letter_guard=letter_guard)
        if not row:
            continue
        name = row["name"]
        score = row.get("match_score", 0.0)
        if name not in found or score > (found[name].match_score or 0):
            found[name] = Entity(
                text=cand,
                score=score / 100.0,
                start=offset,
                end=offset + len(cand),
                matched_name=name,
                match_score=score,
            )
    return list(found.values())


def extract_entities(
    text: str, db: MedicineDB, caps: dict[str, bool]
) -> tuple[list[Entity], list[dict]]:
    """Return (entities, matched_db_rows). matched_db_rows feed downstream."""
    if not text.strip():
        return [], []

    entities: list[Entity] = []
    if caps.get("ner"):
        try:
            entities = _biobert_entities(text)
        except Exception as exc:
            logger.warning("BioBERT NER unavailable: %s", exc)

    # fuzzy-match BioBERT spans to DB
    for e in entities:
        row = db.find(e.text)
        if row:
            e.matched_name = row["name"]
            e.match_score = row.get("match_score")

    # augment with dictionary matches (catches NER misses)
    dict_ents = _dictionary_entities(text, db)
    have = {e.matched_name for e in entities if e.matched_name}
    for de in dict_ents:
        if de.matched_name not in have:
            entities.append(de)
            have.add(de.matched_name)

    # collect unique matched DB rows
    matched_rows: list[dict] = []
    seen: set[str] = set()
    for e in entities:
        if e.matched_name and e.matched_name not in seen:
            seen.add(e.matched_name)
            row = db.df[db.df["name"] == e.matched_name]
            if not row.empty:
                matched_rows.append(row.iloc[0].to_dict())

    # Drop wordpiece fragments / noise for display; always keep matched drugs.
    cleaned = [
        e
        for e in entities
        if e.matched_name or ("##" not in e.text and len(e.text.strip()) >= 4)
    ]

    cleaned.sort(key=lambda x: x.start)
    return cleaned, matched_rows
