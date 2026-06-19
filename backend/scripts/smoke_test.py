"""Offline smoke test — verifies the pipeline wiring without needing the
heavy ML models or a running server. Run:  python -m scripts.smoke_test
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from app.data.loader import get_interaction_db, get_medicine_db
from app.pipeline import interactions, ner, recommend
from app.pipeline.capabilities import probe_capabilities
from app.pipeline.orchestrator import run_pipeline


def make_prescription_png() -> bytes:
    img = Image.new("RGB", (700, 360), "white")
    d = ImageDraw.Draw(img)
    lines = [
        "Dr. A. Sharma Clinic",
        "Rx",
        "Crocin 500mg  1-0-1",
        "Warfarin 5mg  0-0-1",
        "Aspirin 75mg  1-0-0",
        "Augmentin 625 Duo  1-0-1",
    ]
    y = 20
    for ln in lines:
        d.text((30, y), ln, fill="black")
        y += 45
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    caps = probe_capabilities()
    print("== Capabilities ==")
    for k, v in caps.items():
        print(f"  {k:10s}: {v}")

    db = get_medicine_db()
    idb = get_interaction_db()
    print(f"\nMedicineDB ok={db.ok} rows={len(db.df)} has_price={getattr(db,'has_price',False)}")
    print(f"InteractionDB ok={idb.ok} pairs={len(idb.pairs)}")

    # --- Logic path: feed known text directly (no OCR needed) ---
    text = "Crocin 500mg\nWarfarin 5mg\nAspirin 75mg\nAugmentin 625 Duo"
    entities, matched = ner.extract_entities(text, db, caps)
    print("\n== NER (dictionary path) ==")
    for e in entities:
        print(f"  {e.text!r} -> {e.matched_name} (match {e.match_score})")

    recs = recommend.recommend_for_entities(matched, db)
    print("\n== Recommendations ==")
    for r in recs:
        print(f"  {r.prescribed} ({r.composition})")
        for a in r.alternatives:
            print(f"     - {a['name']:24s} Rs{a['price']:.2f}  save {a['saving_pct']}%")

    inter = interactions.detect_interactions(matched, idb)
    print("\n== Interactions ==")
    for it in inter:
        print(f"  [{it['severity']}] {it['drug_a']} + {it['drug_b']}: {it['description'][:60]}...")

    # --- Full image pipeline (uses fallbacks where models are absent) ---
    print("\n== Full pipeline on synthetic image ==")
    report = run_pipeline(make_prescription_png())
    print(f"  message: {report.message}")
    print(f"  regions: {len(report.regions)}  ocr_engines: {[o.engine for o in report.ocr_results]}")
    print(f"  entities: {len(report.entities)}  recs: {len(report.recommendations)}  "
          f"interactions: {len(report.interactions)}  xai: {[x.method for x in report.xai]}")
    print(f"  timings_ms: {report.timings_ms}")

    assert matched, "Expected dictionary NER to match known drugs"
    assert recs, "Expected at least one recommendation"
    assert inter, "Expected at least one interaction (Warfarin+Aspirin)"
    assert report.success
    print("\nSMOKE TEST PASSED ✅")


if __name__ == "__main__":
    main()
