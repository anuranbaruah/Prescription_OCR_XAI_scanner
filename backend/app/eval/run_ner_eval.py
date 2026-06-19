"""NER evaluation — measured precision/recall/F1 for medicine-name extraction.

Compares the NER approaches on the labeled set, scoring at the level of matched
canonical medicine names (micro-averaged over examples):

  * Lexicon linker, greedy           — fuzzy DB match, no letter guard
  * Lexicon linker + letter guard    — rejects digit-driven false positives
  * BioBERT NER -> DB match           — general biomedical transformer
  * Combined (deployed system)        — BioBERT + guarded lexicon linker

The greedy-vs-guarded rows are an ablation of the precision fix.

Run:  python -m app.eval.run_ner_eval
"""
from __future__ import annotations

import os
from functools import lru_cache

from ..config import DATA_DIR
from ..data.loader import MedicineDB
from ..pipeline import ner
from ..pipeline.capabilities import probe_capabilities
from .datasets import load_ner_samples
from .metrics import micro_prf


@lru_cache
def _ner_reference_db() -> MedicineDB:
    """Reference lexicon for the NER entity-*linking* ablation.

    This is deliberately the curated sample lexicon, not the full deployed
    catalog: the 12-example NER gold set is authored against these canonical
    names, so the greedy-vs-letter-guard precision ablation is only meaningful
    when gold labels and the linker's lexicon use the same naming. Recommendation
    and interaction, by contrast, run against the full real Kaggle/DrugBank data.
    Override with ``RXAI_EVAL_NER_DB`` if you author a real-catalog gold set.
    """
    path = os.environ.get("RXAI_EVAL_NER_DB", str(DATA_DIR / "sample_medicines.csv"))
    return MedicineDB(path)


def _dict_pred_greedy(text: str, db) -> set[str]:
    return {
        e.matched_name
        for e in ner._dictionary_entities(text, db, letter_guard=False)
        if e.matched_name
    }


def _dict_pred(text: str, db) -> set[str]:
    return {
        e.matched_name
        for e in ner._dictionary_entities(text, db, letter_guard=True)
        if e.matched_name
    }


def _biobert_pred(text: str, db) -> set[str]:
    preds: set[str] = set()
    for e in ner._biobert_entities(text):
        row = db.find(e.text)
        if row:
            preds.add(row["name"])
    return preds


def _combined_pred(text: str, db, caps) -> set[str]:
    _, matched_rows = ner.extract_entities(text, db, caps)
    return {r["name"] for r in matched_rows if r.get("name")}


def evaluate() -> dict:
    samples = load_ner_samples()
    db = _ner_reference_db()
    caps = probe_capabilities()

    rows: list[list] = []
    best_idx = None
    best_f1 = -1.0

    if samples and db.ok:
        # Each method: (label, predict_fn(text), note)
        methods = [
            ("Lexicon linker (greedy)", lambda t: _dict_pred_greedy(t, db),
             "fuzzy DB match, no letter guard"),
            ("Lexicon linker (+letter guard)", lambda t: _dict_pred(t, db),
             "rejects digit-driven false positives"),
        ]
        if caps.get("ner"):
            methods.append(("BioBERT NER -> DB", lambda t: _biobert_pred(t, db),
                            "general biomedical transformer"))
            methods.append(("Combined (deployed)", lambda t: _combined_pred(t, db, caps),
                            "BioBERT + guarded linker"))

        for label, predict, note in methods:
            examples = [(predict(s.text), set(s.drugs)) for s in samples]
            prf = micro_prf(examples)
            rows.append([
                label,
                round(prf.precision, 3),
                round(prf.recall, 3),
                round(prf.f1, 3),
                note,
            ])
            if prf.f1 > best_f1:
                best_f1 = prf.f1
                best_idx = len(rows) - 1

    return {
        "title": "NER Model Comparison - DRUG entity (measured)",
        "columns": ["Model", "Precision", "Recall", "F1", "Notes"],
        "rows": rows,
        "best_row": best_idx,
        "measured": bool(rows),
        "n_samples": len(samples),
    }


def main() -> None:
    res = evaluate()
    print(res["title"], f"— n={res['n_samples']}")
    for i, row in enumerate(res["rows"]):
        mark = " *" if i == res["best_row"] else "  "
        print(f"{mark} {row[0]:<24} P={row[1]:.3f} R={row[2]:.3f} F1={row[3]:.3f}")


if __name__ == "__main__":
    main()
