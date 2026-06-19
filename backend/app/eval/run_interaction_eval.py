"""Drug-interaction-detection evaluation — measured recall/precision/F1.

Positives/negatives are drawn from the real DrugBank DDI list intersected with
catalog ingredients and mapped to real brand drugs (see ``realsets``). Each case
runs the *deployed* detector (brand -> composition -> active ingredients ->
interaction lookup), so the score reflects our extraction/normalization fidelity
against a known knowledge base. Falls back to the bundled demo ``pairs.jsonl``
when no real DDI sample can be built.

Set ``RXAI_EVAL_INTERACTION_N`` to change the real-sample size (default 200).

Run:  python -m app.eval.run_interaction_eval
"""
from __future__ import annotations

import os

from ..data.loader import get_interaction_db, get_medicine_db
from ..pipeline import interactions
from .datasets import load_interaction_samples
from .realsets import build_interaction_sample


def evaluate() -> dict:
    db = get_medicine_db()
    idb = get_interaction_db()

    n = int(os.environ.get("RXAI_EVAL_INTERACTION_N", "200"))
    samples = build_interaction_sample(db, idb, n=n)
    mode = "real_ddi"
    if not samples:  # e.g. running on the tiny sample DB with no overlap
        samples = load_interaction_samples()
        mode = "labeled_demo"

    tp = fp = fn = tn = 0
    skipped = 0

    for s in samples:
        rows = []
        for name in s.drugs:
            row = db.find(name)
            if row:
                rows.append(row)
        if len(rows) < 2:
            skipped += 1
            continue
        detected = len(interactions.detect_interactions(rows, idb)) > 0
        if s.interacts and detected:
            tp += 1
        elif s.interacts and not detected:
            fn += 1
        elif (not s.interacts) and detected:
            fp += 1
        else:
            tn += 1

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n_eval = tp + fp + fn + tn
    accuracy = (tp + tn) / n_eval if n_eval else 0.0
    presented = len(samples)
    resolution = (presented - skipped) / presented if presented else 0.0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "skipped": skipped,
        "brand_resolution_rate": round(resolution, 4),
        "n_samples": n_eval,
        "mode": mode,
        "source": (
            f"DrugBank DDI x A-Z catalog ({presented} brand pairs)"
            if mode == "real_ddi" else "demo gold pairs"
        ),
        "measured": bool(n_eval),
    }


def main() -> None:
    res = evaluate()
    print(f"Drug interaction detection (measured, {res['mode']})")
    print(f"  recall    : {res['recall'] * 100:.1f}%")
    print(f"  precision : {res['precision'] * 100:.1f}%")
    print(f"  F1        : {res['f1']:.3f}")
    print(f"  TP={res['tp']} FP={res['fp']} FN={res['fn']} TN={res['tn']} "
          f"(n={res['n_samples']}, skipped {res['skipped']}, "
          f"brand-resolution {res['brand_resolution_rate'] * 100:.1f}%)")


if __name__ == "__main__":
    main()
