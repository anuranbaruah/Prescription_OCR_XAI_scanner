"""End-to-end recognition->generic evaluation on the Doctor's Handwritten
Prescription BD dataset (the only set that ships a brand->generic ground truth).

Why this exists
---------------
The A-Z India catalog has *no* brand->generic label, so on that data the
recommendation stage can only report coverage/savings (see ``realsets``). The BD
dataset, by contrast, labels a ``GENERIC_NAME`` for every ``MEDICINE_NAME``
(``Aceta -> Paracetamol``). Those brands are Bangladeshi and do **not** appear in
the Indian catalog, so they cannot be scored against it — instead BD is used as
its *own* brand->generic knowledge base, which is exactly what a deployed system
would carry for that market.

What it measures (per OCR engine, on the held-out handwriting images)
--------------------------------------------------------------------
  * **word accuracy**     — raw OCR brand transcription (exact match);
  * **linking accuracy**  — the fuzzy entity linker (``MedicineDB.find``) maps the
                            (possibly garbled) OCR text back to the correct brand;
  * **generic accuracy**  — the END-TO-END clinical endpoint: handwritten image
                            -> predicted generic ingredient matches the labelled
                            generic.

The headline result is *generic accuracy*: it is typically **higher than raw word
accuracy** because the entity-linking layer recovers OCR spelling errors
(``Aceta``/``Aceto`` both link to Paracetamol). That recovery is the
contribution — no prior handwritten-Rx paper reports a downstream generic
accuracy, they stop at CER.

Run:  python -m app.eval.bd_truth            (all installed engines)
      RXAI_EVAL_BD_LIMIT=150 python -m app.eval.bd_truth   (quick subsample)
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import pandas as pd

from ..config import BACKEND_DIR, DATA_DIR
from ..data.loader import MedicineDB, active_ingredients, normalize_composition
from ..utils import imread_unicode
from ..pipeline import preprocessing
from .run_ocr_eval import _ENGINES, _word_accuracy  # reuse engine runners + metric

PROJECT_ROOT = BACKEND_DIR.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
BD_CATALOG_CSV = DATA_DIR / "eval" / "bd_catalog.csv"

# A few well-known international-vs-South-Asian generic synonyms so a correct
# link is not marked wrong purely on naming convention. Kept tiny and explicit
# — this is ground-truth scoring, not a place to launder fuzzy matches.
_SYNONYMS = {
    "acetaminophen": "paracetamol",
    "salbutamol": "albuterol",
    "albuterol": "salbutamol",
    "frusemide": "furosemide",
    "amoxycillin": "amoxicillin",
}


def _canon(name: str) -> str:
    c = normalize_composition(name)
    return _SYNONYMS.get(c, c)


# --------------------------------------------------------------------------- #
# BD label loading
# --------------------------------------------------------------------------- #
def _bd_root() -> Path | None:
    matches = list(DATASET_ROOT.glob("*Handwritten Prescription BD*"))
    return matches[0] if matches else None


def find_bd_label_files() -> list[Path]:
    """All BD ``*_labels.csv`` files (honours ``RXAI_EVAL_BD_LABELS``)."""
    override = os.environ.get("RXAI_EVAL_BD_LABELS")
    if override:
        p = Path(override)
        if p.is_file():
            return [p]
        return sorted(p.rglob("*_labels.csv")) if p.is_dir() else []
    root = _bd_root()
    return sorted(root.rglob("*_labels.csv")) if root else []


def load_bd_brand_generic() -> dict[str, str]:
    """``brand (lowercased) -> generic`` from all BD label splits.

    The mapping is a fixed lookup (not OCR), so aggregating splits only widens
    brand coverage. On conflicts the most frequent generic wins.
    """
    counts: dict[str, dict[str, int]] = {}
    for path in find_bd_label_files():
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = {c.lower().strip(): c for c in df.columns}
        bcol = cols.get("medicine_name") or cols.get("medicine name")
        gcol = cols.get("generic_name") or cols.get("generic name")
        if not (bcol and gcol):
            continue
        for brand, generic in zip(df[bcol], df[gcol]):
            b, g = str(brand).strip(), str(generic).strip()
            if not b or not g or b.lower() == "nan" or g.lower() == "nan":
                continue
            counts.setdefault(b.lower(), {}).setdefault(g, 0)
            counts[b.lower()][g] += 1
    # preserve the original (cased) brand spelling for the catalog
    cased: dict[str, str] = {}
    for path in find_bd_label_files():
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = {c.lower().strip(): c for c in df.columns}
        bcol = cols.get("medicine_name") or cols.get("medicine name")
        if not bcol:
            continue
        for brand in df[bcol]:
            b = str(brand).strip()
            if b and b.lower() not in cased:
                cased[b.lower()] = b
    return {
        cased.get(b, b): max(g.items(), key=lambda kv: kv[1])[0]
        for b, g in counts.items()
    }


def build_bd_catalog() -> MedicineDB | None:
    """Materialise BD's brand->generic map as a MedicineDB the linker can search.

    ``composition`` is set to the labelled generic so a linked brand expands to
    the correct ingredient. No price column (BD has none) — that's fine; this
    evaluator scores correctness, not savings.
    """
    truth = load_bd_brand_generic()
    if not truth:
        return None
    BD_CATALOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with BD_CATALOG_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "composition"])
        w.writeheader()
        for brand, generic in sorted(truth.items()):
            w.writerow({"name": brand, "composition": generic})
    return MedicineDB(str(BD_CATALOG_CSV))


def _test_samples() -> list[tuple[Path, str, str]]:
    """(image_path, truth_brand, truth_generic) for the BD **test** split."""
    root = _bd_root()
    if not root:
        return []
    test_dir = root / "Testing"
    label_csv = test_dir / "testing_labels.csv"
    img_dir = test_dir / "testing_words"
    if not (label_csv.exists() and img_dir.exists()):
        return []
    out: list[tuple[Path, str, str]] = []
    with label_csv.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            img = (img_dir / (r.get("IMAGE") or "").strip())
            brand = (r.get("MEDICINE_NAME") or "").strip()
            generic = (r.get("GENERIC_NAME") or "").strip()
            if brand and generic and img.exists():
                out.append((img, brand, generic))
    return out


def _generic_matches(label_generic: str, predicted_generic: str) -> bool:
    want = {_canon(p) for p in label_generic.replace("+", ",").split(",")}
    want = {w for w in want if w}
    have = {_canon(i) for i in active_ingredients(predicted_generic)}
    have = {h for h in have if h}
    if not want or not have:
        return False
    for w in want:
        if w in have or any(w == h or f" {w} " in f" {h} " for h in have):
            continue
        return False
    return True


# --------------------------------------------------------------------------- #
# End-to-end evaluation
# --------------------------------------------------------------------------- #
def evaluate() -> dict:
    catalog = build_bd_catalog()
    samples = _test_samples()
    if not catalog or not samples:
        return {"measured": False, "reason": "BD dataset not found"}

    limit = os.environ.get("RXAI_EVAL_BD_LIMIT")
    if limit and limit.isdigit() and 0 < int(limit) < len(samples):
        samples = random.Random(42).sample(samples, int(limit))

    # Preprocess each image once and reuse across engines.
    from ..pipeline.capabilities import probe_capabilities

    caps = probe_capabilities()
    prepared = []
    for img, brand, generic in samples:
        bgr = imread_unicode(img)
        if bgr is None:
            continue
        gray = preprocessing.preprocess(bgr).gray
        prepared.append((brand, generic, bgr, gray))

    rows: list[dict] = []
    for label, cap_key, runner, _note in _ENGINES:
        if not caps.get(cap_key):
            continue
        word_pairs: list[tuple[str, str]] = []
        link_hits = generic_hits = errors = 0
        for brand, generic, bgr, gray in prepared:
            try:
                hyp, _ms = runner(bgr, gray)
            except Exception:
                hyp, errors = "", errors + 1
            word_pairs.append((brand, hyp))
            row = catalog.find(hyp)
            if row:
                if _canon(row.get("name", "")) == _canon(brand):
                    link_hits += 1
                if _generic_matches(generic, row.get("composition", "")):
                    generic_hits += 1
        if not word_pairs or errors == len(word_pairs):
            continue
        n = len(word_pairs)
        rows.append({
            "engine": label,
            "word_accuracy": round(_word_accuracy(word_pairs) * 100, 2),
            "linking_accuracy": round(link_hits / n * 100, 2),
            "generic_accuracy": round(generic_hits / n * 100, 2),
        })

    return {
        "measured": bool(rows),
        "title": "End-to-end handwriting -> generic (BD ground truth)",
        "source": "Doctor's Handwritten Prescription BD (test split, GENERIC_NAME labels)",
        "n_samples": len(prepared),
        "n_brands_in_catalog": len(catalog.df),
        "rows": rows,
        "note": (
            "generic_accuracy is the end-to-end clinical endpoint; it exceeds "
            "word_accuracy because entity linking recovers OCR spelling errors."
        ),
    }


def main() -> None:
    res = evaluate()
    if not res.get("measured"):
        print(f"Not measured: {res.get('reason', 'no engines/images')}")
        return
    print(res["title"], f"— n={res['n_samples']} images, {res['n_brands_in_catalog']} BD brands")
    print(f"  source: {res['source']}")
    print(f"  {'Engine':<22} {'WordAcc%':>9} {'LinkAcc%':>9} {'GenericAcc%':>12}")
    for r in res["rows"]:
        print(f"  {r['engine']:<22} {r['word_accuracy']:>9} {r['linking_accuracy']:>9} {r['generic_accuracy']:>12}")
    print(f"  note: {res['note']}")


if __name__ == "__main__":
    main()
