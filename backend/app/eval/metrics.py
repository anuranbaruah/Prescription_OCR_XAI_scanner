"""Metric primitives for the evaluation harness.

Implemented with the standard edit-distance / set-overlap definitions used in
the OCR and information-extraction literature, with no heavy dependencies so
the evaluators run anywhere the pipeline runs.

References for the definitions used here:
  * WER / CER — Levenshtein edit distance normalized by reference length
    (standard ASR/OCR metric; see e.g. the ``jiwer`` package docs).
  * Precision / Recall / F1 — standard set-based information-extraction metrics.
"""
from __future__ import annotations

from dataclasses import dataclass


def _levenshtein(ref: list[str], hyp: list[str]) -> int:
    """Minimum single-token edit distance (insert/delete/substitute = 1)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost, # substitution
            )
        prev = cur
    return prev[m]


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = edit_distance(words) / max(1, num_reference_words)."""
    ref = _normalize(reference).split()
    hyp = _normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER = edit_distance(chars) / max(1, num_reference_chars)."""
    ref = list(_normalize(reference))
    hyp = list(_normalize(hypothesis))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def corpus_wer_cer(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    """Aggregate WER/CER over (reference, hypothesis) pairs.

    Aggregated the standard way: total edit distance over total reference
    length (a length-weighted micro-average), not the mean of per-sample rates.
    """
    ref_words = hyp_word_edits = 0
    ref_chars = hyp_char_edits = 0
    for reference, hypothesis in pairs:
        r_w = _normalize(reference).split()
        h_w = _normalize(hypothesis).split()
        ref_words += len(r_w)
        hyp_word_edits += _levenshtein(r_w, h_w)

        r_c = list(_normalize(reference))
        h_c = list(_normalize(hypothesis))
        ref_chars += len(r_c)
        hyp_char_edits += _levenshtein(r_c, h_c)

    wer = hyp_word_edits / ref_words if ref_words else 0.0
    cer = hyp_char_edits / ref_chars if ref_chars else 0.0
    return wer, cer


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def prf_from_sets(predicted: set[str], gold: set[str]) -> PRF:
    """Precision/Recall/F1 for one example given predicted vs gold item sets."""
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)


def micro_prf(examples: list[tuple[set[str], set[str]]]) -> PRF:
    """Micro-averaged P/R/F1 over many (predicted, gold) set pairs.

    Micro-averaging pools TP/FP/FN across all examples before computing the
    rates — the convention for entity-level NER scoring (cf. CoNLL / seqeval).
    """
    tp = fp = fn = 0
    for predicted, gold in examples:
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)
