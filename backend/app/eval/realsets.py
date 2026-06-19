"""Real-scale evaluation sets built from the wired catalog + DDI databases.

These let the recommendation and interaction stages be measured on the full
public datasets (A-Z India catalog, DrugBank DDI list) instead of the tiny
curated demo gold sets. Sets are built deterministically (fixed seed) so runs
are reproducible.

Honesty notes (important — see EVALUATION.md)
---------------------------------------------
* The A-Z catalog has **no brand->generic ground-truth mapping**, so the
  recommendation stage cannot have a labelled "is this the right generic"
  accuracy on this data. Instead we measure what *is* objectively checkable:
    - substitution **coverage**: does a cheaper same-composition generic exist
      for a prescribed brand?
    - the real **cost-saving** distribution (from real MRP prices);
    - **equivalence precision**: returned alternatives genuinely share the
      normalized composition (a validation check on the equivalence logic — it
      is ~1.0 by construction and is reported as such, not as a headline).
* Interaction positives/negatives are grounded in the real DDI list intersected
  with catalog ingredients, then mapped back to real single-ingredient brand
  drugs. Recall therefore measures our brand->composition->ingredient extraction
  fidelity against a known knowledge base; it is near-ceiling by construction
  and is reported honestly as a plumbing/coverage check, not as evidence the
  DDI list itself is complete.
"""
from __future__ import annotations

import random

from ..data.loader import MedicineDB, InteractionDB, active_ingredients
from .datasets import InteractionSample


def ingredient_to_brand(db: MedicineDB) -> dict[str, str]:
    """Map each single active ingredient -> a representative brand drug name.

    Only non-discontinued, priced, *single-ingredient* drugs are used so a
    mapped brand expands to exactly the intended ingredient (no confounds from
    multi-ingredient combinations).
    """
    mapping: dict[str, str] = {}
    df = db.df
    for name, comp, disc, price in zip(
        df["name"], df["composition"], df["is_discontinued"], df["price"]
    ):
        if disc or price != price:  # price != price filters NaN
            continue
        ings = active_ingredients(comp)
        if len(ings) != 1:
            continue
        mapping.setdefault(ings[0], name)
    return mapping


def _ddi_names(idb: InteractionDB) -> set[str]:
    names: set[str] = set()
    for key in idb.pairs:
        names |= set(key)
    return names


def build_recommend_brands(db: MedicineDB, n: int = 500, seed: int = 42) -> list[str]:
    """Fixed-seed sample of prescribable brand names (non-discontinued, priced,
    with a parsed composition) to measure substitution coverage + savings on."""
    df = db.df
    candidates = [
        nm
        for nm, price, disc, nc in zip(
            df["name"], df["price"], df["is_discontinued"], df["norm_comp"]
        )
        if price == price and not disc and nc
    ]
    rng = random.Random(seed)
    if len(candidates) > n:
        candidates = rng.sample(candidates, n)
    return candidates


def build_interaction_sample(
    db: MedicineDB, idb: InteractionDB, n: int = 200, seed: int = 42
) -> list[InteractionSample]:
    """Balanced positive/negative interaction cases drawn from real DDI pairs.

    Positives are real DrugBank interacting ingredient pairs (both ingredients
    mappable to a real single-ingredient brand). Negatives are random mappable
    ingredient pairs absent from the DDI list. Each ingredient is replaced by a
    real brand name so the *deployed* brand->ingredient pipeline is exercised.
    """
    rng = random.Random(seed)
    ing2brand = ingredient_to_brand(db)
    usable = set(ing2brand) & _ddi_names(idb)

    half = n // 2
    positives: list[tuple[str, str]] = []
    for key in idb.pairs:
        a, b = tuple(key)
        if a in usable and b in usable:
            positives.append((a, b))
    rng.shuffle(positives)
    positives = positives[:half]

    usable_list = sorted(usable)
    negatives: list[tuple[str, str]] = []
    attempts = 0
    while len(negatives) < half and attempts < half * 100 and len(usable_list) > 2:
        a, b = rng.sample(usable_list, 2)
        attempts += 1
        if frozenset((a, b)) in idb.pairs:
            continue
        negatives.append((a, b))

    samples = [
        InteractionSample([ing2brand[a], ing2brand[b]], True) for a, b in positives
    ] + [
        InteractionSample([ing2brand[a], ing2brand[b]], False) for a, b in negatives
    ]
    rng.shuffle(samples)
    return samples
