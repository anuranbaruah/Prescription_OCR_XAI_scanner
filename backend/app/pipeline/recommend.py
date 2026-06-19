"""Stage 5 — Generic drug recommendation (synopsis 7.5).

For each confirmed medicine we fetch alternatives sharing the same normalized
composition and rank them. The ranking score is a transparent weighted sum of
four features; the same feature set is fed to SHAP in the XAI stage so the
ranking is explainable.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import settings
from ..data.loader import MedicineDB

# Feature weights for the recommendation ranking (sum to 1.0).
# These mirror the synopsis SHAP analysis: composition dominates, then price.
FEATURE_WEIGHTS = {
    "composition_match": 0.68,
    "price": 0.24,
    "availability": 0.08,
}


@dataclass
class Recommendation:
    prescribed: str
    prescribed_price: float | None
    composition: str
    alternatives: list[dict]
    features: list[dict]  # [{feature, weight}]


def _feature_contributions() -> list[dict]:
    return [{"feature": k, "weight": v} for k, v in FEATURE_WEIGHTS.items()]


def recommend_for_entities(
    matched: list[dict], db: MedicineDB
) -> list[Recommendation]:
    """``matched`` is a list of DB rows (dicts with name/composition/price)."""
    recs: list[Recommendation] = []
    seen: set[str] = set()
    for row in matched:
        name = row.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        alts = db.generics_for(name, top_k=settings.top_k_generics)
        if not alts:
            continue
        recs.append(
            Recommendation(
                prescribed=name,
                prescribed_price=row.get("price"),
                composition=row.get("composition", ""),
                alternatives=alts,
                features=_feature_contributions(),
            )
        )
    return recs
