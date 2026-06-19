"""LIME explanation for the BioBERT NER decision (synopsis 7.7).

Explains *why* a token was classified as a medicine name by perturbing the
surrounding context and observing how the DRUG probability changes.

Primary: ``lime.lime_text.LimeTextExplainer`` driving the BioBERT NER pipeline
(bounded sample budget to stay responsive).
Fallback: a fast occlusion analysis (drop one word at a time, measure the
change in the entity's DRUG score).
"""
from __future__ import annotations

import logging
import re

import numpy as np

from ...config import settings
from ...utils import matplotlib_fig_to_base64
from ..ner import Entity

logger = logging.getLogger("rxai.xai.lime")

# keep XAI responsive: LIME perturbation budget
_LIME_SAMPLES = 120


def _drug_prob(texts: list[str], target: str) -> np.ndarray:
    """P(text contains a DRUG entity ~ target) for each text. Shape (n, 2)."""
    from ..ner import _load_ner

    pipe = _load_ner()
    target_l = target.lower()
    out = []
    results = pipe(texts) if len(texts) > 1 else [pipe(texts[0])]
    for res in results:
        best = 0.0
        for e in res:
            label = str(e.get("entity_group", "")).upper()
            if any(h in label for h in ("MEDIC", "DRUG", "CHEMICAL", "PHARM")):
                score = float(e["score"])
                # bias toward the specific token we are explaining
                if target_l in e["word"].lower() or e["word"].lower() in target_l:
                    score = max(score, float(e["score"]))
                best = max(best, score)
        out.append([1.0 - best, best])
    return np.array(out)


def _lime_explanation(text: str, entity: Entity) -> list[tuple[str, float]] | None:
    try:
        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(class_names=["not_drug", "drug"], bow=False)

        def classifier_fn(texts):
            return _drug_prob(list(texts), entity.text)

        exp = explainer.explain_instance(
            text,
            classifier_fn,
            num_features=8,
            num_samples=_LIME_SAMPLES,
            labels=(1,),
        )
        return exp.as_list(label=1)
    except Exception as exc:
        logger.warning("LIME unavailable, using occlusion: %s", exc)
        return None


def _occlusion_explanation(text: str, entity: Entity) -> list[tuple[str, float]] | None:
    try:
        words = re.findall(r"\S+", text)
        if not words:
            return None
        base = _drug_prob([text], entity.text)[0][1]
        importances: list[tuple[str, float]] = []
        # limit forward passes for responsiveness
        for i, w in enumerate(words[:25]):
            occluded = " ".join(words[:i] + words[i + 1:])
            if not occluded.strip():
                continue
            p = _drug_prob([occluded], entity.text)[0][1]
            importances.append((w, round(base - p, 4)))  # positive => supports DRUG
        importances.sort(key=lambda t: abs(t[1]), reverse=True)
        return importances[:8]
    except Exception as exc:
        logger.warning("Occlusion explanation failed: %s", exc)
        return None


def lime_artifact(text: str, entities: list[Entity], caps: dict[str, bool]) -> dict | None:
    if not settings.enable_xai or not text.strip() or not entities or not caps.get("ner"):
        return None

    target = next((e for e in entities if e.matched_name), entities[0])

    weights = None
    method_label = "LIME"
    if caps.get("lime"):
        weights = _lime_explanation(text, target)
    if weights is None:
        weights = _occlusion_explanation(text, target)
        method_label = "Occlusion (LIME-style)"
    if not weights:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        words = [w for w, _ in weights][::-1]
        vals = [v for _, v in weights][::-1]
        colors = ["#16a34a" if v >= 0 else "#dc2626" for v in vals]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.barh(words, vals, color=colors)
        ax.axvline(0, color="#666", linewidth=0.8)
        ax.set_xlabel("Contribution to DRUG classification")
        ax.set_title(f"{method_label}: '{target.text}' classified as medicine")
        fig.tight_layout()
        b64 = matplotlib_fig_to_base64(fig)
        plt.close(fig)
    except Exception as exc:
        logger.warning("LIME plot failed: %s", exc)
        return None

    return {
        "method": "lime",
        "target_stage": "BioBERT NER",
        "title": "Why this token is a medicine name (LIME)",
        "image_base64": b64,
        "note": (
            "Green words pushed the model toward a DRUG classification; red words "
            "pushed against it. Helps spot low-confidence NER calls."
        ),
    }
