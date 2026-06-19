"""Generic-recommendation evaluation.

On the real A-Z India catalog there is no brand->generic ground-truth label, so
we measure what the data objectively supports (see ``realsets`` for the full
rationale):

  * **coverage**            — fraction of prescribed brands for which the system
                              finds >=1 cheaper same-composition generic;
  * **cost saving**         — mean/median/max % saving on those (real MRP prices);
  * **equivalence precision** — returned alternatives genuinely share the
                              normalized composition (validation; ~1.0 by design).

If a labelled ``cases.jsonl`` is present *and* its expected generics resolve in
the active DB, a secondary exact-match ``labeled_accuracy`` is also reported
(used by the bundled demo gold set).

Run:  python -m app.eval.run_recommend_eval
"""
from __future__ import annotations

import os
import statistics

from ..data.loader import get_medicine_db, strength_composition
from ..pipeline import recommend
from .datasets import load_recommend_samples
from .realsets import build_recommend_brands


def _labeled_accuracy(db) -> dict | None:
    """Exact expected-generic match on a labelled gold set, if it resolves."""
    samples = load_recommend_samples()
    if not samples:
        return None
    total = correct = resolved = 0
    for s in samples:
        row = db.find(s.brand)
        total += 1
        if not row:
            continue
        recs = recommend.recommend_for_entities([row], db)
        alt_names = {a["name"] for r in recs for a in r.alternatives}
        if any(s.expected_generic.lower() == n.lower() for n in alt_names):
            correct += 1
        # A label "resolves" only if it is an *exact* catalog name. Fuzzy
        # resolution would falsely mark demo labels (tailored to the sample DB)
        # as present in the real catalog and report a misleading 0% accuracy.
        if s.expected_generic.lower() in db._by_lower:
            resolved += 1
    # Only meaningful if the gold labels actually exist in the active catalog.
    if resolved == 0:
        return {"labeled_accuracy": None, "labeled_n": total, "labels_resolve": False}
    return {
        "labeled_accuracy": round(correct / total, 4) if total else 0.0,
        "labeled_n": total,
        "labels_resolve": True,
    }


def evaluate() -> dict:
    db = get_medicine_db()
    n = int(os.environ.get("RXAI_EVAL_RECOMMEND_N", "500"))
    brands = build_recommend_brands(db, n=n)

    covered = 0
    savings: list[float] = []
    alt_total = alt_equiv = 0  # equivalence-precision accounting

    for name in brands:
        row = db.find(name)
        if not row:
            continue
        ref_key = strength_composition(row.get("composition", ""))
        recs = recommend.recommend_for_entities([row], db)
        best_saving = 0.0
        has_alt = False
        for r in recs:
            for a in r.alternatives:
                has_alt = True
                alt_total += 1
                if strength_composition(a.get("composition", "")) == ref_key:
                    alt_equiv += 1
                best_saving = max(best_saving, a.get("saving_pct") or 0.0)
        if has_alt:
            covered += 1
        if best_saving > 0:
            savings.append(best_saving)

    n_eval = len(brands)
    coverage = covered / n_eval if n_eval else 0.0
    result = {
        "measured": bool(n_eval),
        "mode": "catalog",
        "n_samples": n_eval,
        "coverage": round(coverage, 4),
        "equivalence_precision": round(alt_equiv / alt_total, 4) if alt_total else None,
        "mean_cost_saving_pct": round(statistics.mean(savings), 1) if savings else 0.0,
        "median_cost_saving_pct": round(statistics.median(savings), 1) if savings else 0.0,
        "max_cost_saving_pct": round(max(savings), 1) if savings else 0.0,
        "n_with_saving": len(savings),
        "source": f"A-Z India catalog (n={n_eval} sampled brands)",
        # backward-compatible key: coverage is the closest analogue to the old
        # "accuracy" (fraction of brands the recommender serves a generic for).
        "accuracy": round(coverage, 4),
    }
    labeled = _labeled_accuracy(db)
    if labeled:
        result.update(labeled)
    return result


def main() -> None:
    res = evaluate()
    print("Generic recommendation (measured, real catalog)")
    print(f"  brands sampled       : {res['n_samples']}")
    print(f"  substitution coverage: {res['coverage'] * 100:.1f}%")
    print(f"  mean cost saving     : {res['mean_cost_saving_pct']}%  (median {res['median_cost_saving_pct']}%, max {res['max_cost_saving_pct']}%, n={res['n_with_saving']})")
    if res.get("equivalence_precision") is not None:
        print(f"  equivalence precision: {res['equivalence_precision'] * 100:.1f}%  (validation)")
    if res.get("labeled_accuracy") is not None:
        print(f"  labeled accuracy     : {res['labeled_accuracy'] * 100:.1f}%  (n={res['labeled_n']} gold cases)")
    elif res.get("labels_resolve") is False:
        print("  labeled accuracy     : n/a (demo gold labels don't exist in the real catalog)")


if __name__ == "__main__":
    main()
