"""Stage 6 — Rule-based drug interaction detection (synopsis 7.6).

For every pair of identified medicines we expand them to their active
ingredients and query the interaction knowledge base for any matching pair,
returning a severity-tagged, plain-language warning.
"""
from __future__ import annotations

from itertools import combinations

from ..data.loader import InteractionDB, active_ingredients

_SEVERITY_RANK = {"major": 0, "moderate": 1, "minor": 2}


def detect_interactions(matched: list[dict], idb: InteractionDB) -> list[dict]:
    """``matched`` is a list of DB rows; returns interaction dicts."""
    if not idb.ok or len(matched) < 2:
        return []

    # map each medicine -> its active ingredients
    meds = []
    for row in matched:
        name = row.get("name")
        comp = row.get("composition", "")
        if name:
            meds.append((name, active_ingredients(comp)))

    found: dict[frozenset, dict] = {}
    for (name_a, ings_a), (name_b, ings_b) in combinations(meds, 2):
        for ia in ings_a:
            for ib in ings_b:
                hit = idb.check(ia, ib)
                if hit:
                    key = frozenset((name_a, name_b))
                    # keep the most severe interaction per medicine pair
                    if key not in found or _SEVERITY_RANK.get(
                        hit["severity"], 1
                    ) < _SEVERITY_RANK.get(found[key]["severity"], 1):
                        found[key] = {
                            "drug_a": name_a,
                            "drug_b": name_b,
                            "severity": hit["severity"],
                            "description": hit["description"],
                        }

    return sorted(
        found.values(), key=lambda d: _SEVERITY_RANK.get(d["severity"], 1)
    )
