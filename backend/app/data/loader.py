"""Medicine knowledge base + drug-drug interaction store.

Designed to work with both the bundled sample CSVs and the real Kaggle
"11000 Medicine Details" dataset (column names are mapped case-insensitively).
Missing prices are tolerated — savings are simply omitted in that case.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

import pandas as pd

from ..config import settings

logger = logging.getLogger("rxai.data")

# common column aliases -> canonical name
_MED_ALIASES = {
    "name": {"name", "medicine name", "medicine_name", "drug", "drug name", "brand"},
    "composition": {"composition", "salt composition", "salt", "active ingredients", "generic name"},
    "manufacturer": {"manufacturer", "manufacturer_name", "marketer", "company"},
    "price": {"price", "price(₹)", "price (₹)", "mrp", "cost", "price_inr"},
}

# Split-composition columns (e.g. the A-Z India catalog stores active
# ingredients across ``short_composition1`` / ``short_composition2``). When no
# single ``composition`` column is present we synthesize one by joining these.
_COMP_PART_RE = re.compile(r"(short_)?composition[\s_]*\d+", re.I)
_DISCONTINUED_COLS = {"is_discontinued", "discontinued"}

_DOSAGE_RE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|ml|g|iu|%|units?)\b", re.I)
_NONWORD_RE = re.compile(r"[^a-z0-9+ ]")
_ALPHA_TOKEN_RE = re.compile(r"[a-z]{3,}")


def alpha_stem(text: str) -> str:
    """Space-joined alphabetic tokens (length >= 3) of a string, lowercased.

    Used to make fuzzy drug matching letter-driven: 'Crocin 500' -> 'crocin',
    '500 1-0-1' -> '' (all digits). This lets us reject matches that are driven
    only by a shared dosage number rather than the brand name itself.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(_ALPHA_TOKEN_RE.findall(text.lower()))


def normalize_composition(comp: str) -> str:
    """Canonicalize a composition string for equivalence matching.

    'Amoxicillin 500mg + Clavulanic Acid 125mg' -> 'amoxicillin + clavulanic acid'
    Sorting the components makes order-independent matching possible.
    """
    if not isinstance(comp, str):
        return ""
    s = comp.lower()
    s = _DOSAGE_RE.sub("", s)
    s = _NONWORD_RE.sub(" ", s)
    parts = [re.sub(r"\s+", " ", p).strip() for p in s.split("+")]
    parts = [p for p in parts if p]
    return " + ".join(sorted(parts))


def active_ingredients(comp: str) -> list[str]:
    """Return the list of base ingredient names (no dosage) from a composition."""
    norm = normalize_composition(comp)
    return [p.strip() for p in norm.split("+") if p.strip()]


def _ingredient_with_strength(part: str) -> str:
    """'Amoxycillin (500mg)' -> 'amoxycillin 500mg' (name + normalized dose)."""
    p = part.lower()
    m = _DOSAGE_RE.search(p)
    dose = re.sub(r"\s+", "", m.group(0)) if m else ""
    name = _NONWORD_RE.sub(" ", _DOSAGE_RE.sub(" ", p))
    name = re.sub(r"\s+", " ", name).strip()
    return f"{name} {dose}".strip()


def strength_composition(comp: str) -> str:
    """Dosage-aware composition key for *generic substitution*.

    Unlike ``normalize_composition`` (molecule-level, used for interactions),
    this keeps each ingredient's strength so a substitute must be bioequivalent:
    'Azithromycin (500mg)' -> 'azithromycin 500mg' will NOT match a 250mg drug.
    """
    if not isinstance(comp, str):
        return ""
    parts = [_ingredient_with_strength(p) for p in comp.split("+")]
    parts = [p for p in parts if p]
    return " + ".join(sorted(parts))


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for canonical, aliases in _MED_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                rename[lower[alias]] = canonical
                break
    return df.rename(columns=rename)


def _synthesize_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Build a single ``composition`` column from split-composition columns.

    The A-Z India catalog has no ``composition`` column — active ingredients
    live in ``short_composition1`` / ``short_composition2``. We join the
    non-empty parts with ``+`` so ``normalize_composition`` can treat them like
    any other multi-ingredient string. No-op when a usable ``composition``
    column already exists (e.g. the bundled sample CSV).
    """
    existing = df["composition"].astype(str).str.strip() if "composition" in df.columns else None
    if existing is not None and existing.str.lower().replace("nan", "").str.len().gt(0).any():
        return df
    parts = [c for c in df.columns if _COMP_PART_RE.fullmatch(str(c).strip())]
    if not parts:
        return df
    cols = []
    for c in parts:
        s = df[c].fillna("").astype(str).str.strip()
        cols.append(s.where(s.str.lower() != "nan", "").to_numpy())
    rows = zip(*cols)
    df = df.copy()
    df["composition"] = [" + ".join(p for p in row if p) for row in rows]
    return df


def _discontinued_flag(df: pd.DataFrame) -> pd.Series:
    col = next((c for c in df.columns if str(c).lower().strip() in _DISCONTINUED_COLS), None)
    if col is None:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


class MedicineDB:
    def __init__(self, csv_path: str):
        self.ok = False
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning("Could not load medicines CSV %s: %s", csv_path, exc)
            self.df = pd.DataFrame(columns=["name", "composition", "manufacturer", "price"])
            return

        df = _map_columns(df)
        df["is_discontinued"] = _discontinued_flag(df)
        df = _synthesize_composition(df)
        for col in ("name", "composition", "manufacturer", "price"):
            if col not in df.columns:
                df[col] = None
        df["name"] = df["name"].astype(str).str.strip()
        df["composition"] = df["composition"].astype(str).str.strip()
        df["manufacturer"] = df["manufacturer"].fillna("Unknown").astype(str).str.strip()
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df[df["name"].str.len() > 0].drop_duplicates(subset=["name"]).reset_index(drop=True)
        df["norm_comp"] = df["composition"].map(normalize_composition)
        # Dosage-aware key for generic substitution (must be same strength).
        df["strength_key"] = df["composition"].map(strength_composition)

        self.df = df
        self.has_price = df["price"].notna().any()
        self.names = df["name"].tolist()
        # case-insensitive exact-name index -> row position (fast path for find())
        self._by_lower = {n.lower(): i for i, n in enumerate(self.names)}
        # Blocking index: maps the first 2 letters of *every* word's stem in a
        # name to that name. Fuzzy matching then searches only the query's block
        # (~hundreds of names) instead of the whole catalog (~250k) — a ~750 ms
        # scan becomes ~1 ms. Indexing every word (not just the leading one)
        # keeps a molecule buried inside a combo name (e.g. "...+Metformin")
        # findable. Essential for interactive latency on the real catalog.
        self._blocks: dict[str, list[str]] = {}
        for n in self.names:
            seen: set[str] = set()
            for tok in alpha_stem(n).split():
                key = tok[:2]
                if len(key) == 2 and key not in seen:
                    self._blocks.setdefault(key, []).append(n)
                    seen.add(key)
        self.ok = len(df) > 0
        logger.info(
            "MedicineDB loaded: %d rows (price=%s, discontinued=%d)",
            len(df), self.has_price, int(df["is_discontinued"].sum()),
        )

    def find(self, query: str, threshold: int | None = None, letter_guard: bool = True):
        """Fuzzy-match a (possibly OCR-garbled) name to a DB record.

        ``letter_guard`` (default on) requires the matched drug's alphabetic
        stem to appear in the query, rejecting digit-driven false positives.
        Pass ``False`` to get the raw greedy fuzzy behaviour (used for ablation).
        """
        if not self.ok or not query:
            return None
        query = query.strip()
        # Reject subword fragments / too-short tokens that fuzzy-match spuriously
        # (e.g. BioBERT emits "as", "##ocin" which would mis-match real names).
        if len(query.replace("#", "")) < 4 or query.startswith("##"):
            return None
        # Fast path: exact (case-insensitive) name hit. Avoids a full fuzzy scan
        # over the whole catalog (~250k names) when the query is already exact.
        exact = self._by_lower.get(query.lower())
        if exact is not None:
            row = self.df.iloc[exact].to_dict()
            row["match_score"] = 100.0
            return row
        threshold = threshold or settings.fuzzy_threshold
        try:
            from rapidfuzz import fuzz, process

            # Restrict the fuzzy search to the query's blocking bucket (names
            # containing a word that shares the query's first two stem letters).
            # A query with no 2-letter alphabetic stem (pure digits/dosage junk
            # like "1-0-1") cannot match a real drug — return early instead of
            # paying a full ~250k-name scan on every such OCR fragment.
            qstem = alpha_stem(query)
            qkey = qstem.split(" ", 1)[0][:2] if qstem else ""
            if len(qkey) < 2:
                return None
            choices = self._blocks.get(qkey)
            if not choices:
                return None
            match = process.extractOne(
                query, choices, scorer=fuzz.WRatio, score_cutoff=threshold
            )
        except Exception:
            # exact (case-insensitive) fallback
            ql = query.lower()
            for n in self.names:
                if n.lower() == ql:
                    return self.df[self.df["name"] == n].iloc[0].to_dict() | {"match_score": 100.0}
            return None
        if not match:
            return None
        name, score, _ = match
        # Letter-driven guard: the matched drug's alphabetic stem must actually
        # appear in the query. Rejects digit-driven false positives such as a
        # '500 1-0-1' fragment fuzzy-matching 'Glucophage 500' on the shared
        # dosage number. Names with no alphabetic stem (e.g. 'P-500') fall back
        # to the raw fuzzy score.
        name_alpha = alpha_stem(name)
        if letter_guard and name_alpha:
            query_alpha = alpha_stem(query)
            if not query_alpha:
                return None
            from rapidfuzz import fuzz as _fuzz

            if _fuzz.token_set_ratio(name_alpha, query_alpha) < 80:
                return None
        row = self.df[self.df["name"] == name].iloc[0].to_dict()
        row["match_score"] = float(score)
        return row

    def generics_for(self, name: str, top_k: int | None = None) -> list[dict]:
        """Return cheaper alternatives sharing the same normalized composition."""
        if not self.ok:
            return []
        top_k = top_k or settings.top_k_generics
        rec = self.df[self.df["name"] == name]
        if rec.empty:
            return []
        key = rec.iloc[0]["strength_key"]
        ref_price = rec.iloc[0]["price"]
        # Empty composition is not an equivalence class — otherwise every drug
        # with an unparsed composition would be a "generic" of every other.
        # Match on the dosage-aware key so substitutes are same molecule AND
        # strength (a true bioequivalent generic), not merely the same molecule.
        if not key:
            return []
        same = self.df[
            (self.df["strength_key"] == key)
            & (self.df["name"] != name)
            & (~self.df["is_discontinued"])
        ].copy()
        if same.empty:
            return []
        # rank by price ascending (cheaper first); unknown prices go last
        same["_sort_price"] = same["price"].fillna(float("inf"))
        same = same.sort_values("_sort_price")
        out = []
        for _, r in same.head(top_k).iterrows():
            price = r["price"]
            saving = None
            if pd.notna(price) and pd.notna(ref_price) and ref_price > 0:
                saving = round(max(0.0, (ref_price - price) / ref_price * 100), 1)
            out.append(
                {
                    "name": r["name"],
                    "composition": r["composition"],
                    "manufacturer": r["manufacturer"],
                    "price": float(price) if pd.notna(price) else 0.0,
                    "saving_pct": saving if saving is not None else 0.0,
                }
            )
        return out


class InteractionDB:
    def __init__(self, csv_path: str):
        self.ok = False
        self.pairs: dict[frozenset[str], dict] = {}
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning("Could not load interactions CSV %s: %s", csv_path, exc)
            return
        lower = {c.lower().strip(): c for c in df.columns}
        a = lower.get("drug_a") or lower.get("drug a") or lower.get("drug1") or lower.get("drug 1")
        b = lower.get("drug_b") or lower.get("drug b") or lower.get("drug2") or lower.get("drug 2")
        sev = lower.get("severity") or lower.get("level")
        desc = (
            lower.get("description")
            or lower.get("interaction description")
            or lower.get("interaction")
            or lower.get("effect")
        )
        if not (a and b):
            logger.warning("Interactions CSV missing drug columns")
            return
        for _, row in df.iterrows():
            da = str(row[a]).strip().lower()
            dbn = str(row[b]).strip().lower()
            if not da or not dbn or da == "nan" or dbn == "nan":
                continue
            self.pairs[frozenset((da, dbn))] = {
                "drug_a": str(row[a]).strip(),
                "drug_b": str(row[b]).strip(),
                "severity": (str(row[sev]).strip().lower() if sev else "moderate"),
                "description": (str(row[desc]).strip() if desc else "Potential interaction."),
            }
        self.ok = len(self.pairs) > 0
        logger.info("InteractionDB loaded: %d pairs", len(self.pairs))

    def check(self, ingredient_a: str, ingredient_b: str) -> dict | None:
        return self.pairs.get(frozenset((ingredient_a.lower(), ingredient_b.lower())))


@lru_cache
def get_medicine_db() -> MedicineDB:
    return MedicineDB(settings.medicines_csv_path)


@lru_cache
def get_interaction_db() -> InteractionDB:
    return InteractionDB(settings.interactions_csv_path)
