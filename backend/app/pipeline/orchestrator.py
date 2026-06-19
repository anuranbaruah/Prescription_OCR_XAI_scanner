"""End-to-end pipeline orchestration (synopsis Section 6 architecture).

upload bytes -> preprocess -> detect -> OCR -> NER -> recommend + interactions
            -> XAI -> AnalysisReport
"""
from __future__ import annotations

import concurrent.futures
import logging
import time

from ..config import settings
from ..data.loader import get_interaction_db, get_medicine_db
from ..schemas import (
    AnalysisReport,
    DetectedRegion,
    DrugEntity,
    DrugInteraction,
    DrugRecommendation,
    GenericAlternative,
    OCRResult,
    PrescriptionMedication,
    PrescriptionStructured,
    RecommendationFeature,
    XAIArtifact,
)
from ..utils import Timer, bgr_to_base64_png, bytes_to_bgr
from . import detection, interactions, ner, ocr, preprocessing, recommend
from .capabilities import probe_capabilities
from .xai import gradcam, lime_explain, shap_explain

logger = logging.getLogger("rxai.orchestrator")


def run_pipeline(image_bytes: bytes) -> AnalysisReport:
    caps = probe_capabilities()
    timer = Timer()
    report = AnalysisReport(device=settings.resolved_device, capabilities=caps)

    image_bgr = bytes_to_bgr(image_bytes)

    # 1) Preprocess
    with timer.stage("preprocess"):
        pre = preprocessing.preprocess(image_bgr)

    # 2) Detect text regions
    with timer.stage("detection"):
        regions, det_method = detection.detect_regions(image_bgr, pre.gray)
    report.regions = [
        DetectedRegion(bbox=list(r.bbox), confidence=round(r.confidence, 3))
        for r in regions
    ]
    annotated = detection.draw_regions(image_bgr, regions)
    report.preprocessed_image = bgr_to_base64_png(annotated)

    # 3) OCR. The local engines (TrOCR/EasyOCR/Tesseract) always run for the
    #    engine comparison. When a vision LLM is configured we fire ONE structured
    #    call *concurrently* (it is network-bound): it both fills the prescription
    #    card and serves as the "Vision LLM" engine row, so its latency overlaps
    #    the local OCR rather than adding to it. Nothing local is skipped.
    struct_future = pool = None
    vis_t0 = None
    if caps.get("vision"):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        vis_t0 = time.perf_counter()
        struct_future = pool.submit(ocr.extract_prescription, image_bgr)

    with timer.stage("ocr"):
        ocr_outputs, primary = ocr.run_ocr(image_bgr, pre.gray, regions, caps)

    if struct_future is not None:
        try:  # bounded wait — never let a slow vision call hang the analysis
            structured = struct_future.result(timeout=30)
        except Exception as exc:
            logger.warning("Vision structured call failed/timed out: %s", exc)
            structured = None
        vis_ms = round((time.perf_counter() - vis_t0) * 1000, 1)
        pool.shutdown(wait=False)
        if structured and structured.get("medications"):
            try:
                report.prescription = PrescriptionStructured(**structured)
            except Exception as exc:  # malformed JSON shouldn't break the run
                logger.warning("Structured prescription parse failed: %s", exc)
            vtext = "\n".join(
                " ".join(p for p in (m.get("name"), m.get("dosage")) if p)
                for m in structured["medications"]
            )
            vis_out = ocr.EngineOutput("Vision LLM", vtext, None, vis_ms)
            ocr_outputs = [vis_out] + [o for o in ocr_outputs if o.engine != "Vision LLM"]
            if settings.primary_ocr == "vision" and vtext.strip():
                primary = vis_out  # vision drives the downstream linking

    db = get_medicine_db()
    # spell-correct the primary text against the medical dictionary
    corrected = ocr.correct_text(primary.text, db.names) if db.ok else primary.text
    report.extracted_text = corrected
    report.ocr_results = [
        OCRResult(
            engine=o.engine,
            text=(corrected if o.engine == primary.engine else o.text),
            confidence=o.confidence,
            inference_ms=o.inference_ms,
        )
        for o in ocr_outputs
    ]

    # 4) NER + DB matching
    with timer.stage("ner"):
        entities, matched_rows = ner.extract_entities(corrected, db, caps)
    report.entities = [
        DrugEntity(
            text=e.text,
            score=round(e.score, 3),
            start=e.start,
            end=e.end,
            matched_name=e.matched_name,
            match_score=round(e.match_score, 1) if e.match_score is not None else None,
        )
        for e in entities
    ]

    # Fallback prescription card: if the structured vision call did not produce
    # one (rate-limited, timed out, returned no medications, or no vision key),
    # ALWAYS still attach a card so the "Extracted Prescription Result" view is
    # shown for EVERY upload — never silently missing. Medications are recovered
    # from the recognized drug names (canonical DB matches first, then OCR text
    # lines); when nothing is readable the card renders with an empty list and
    # the frontend shows "No medications could be read". Patient/doctor/date stay
    # blank (we only surface what we actually read) and a note explains why.
    if report.prescription is None:
        med_names: list[str] = []
        seen: set[str] = set()
        for row in matched_rows:  # canonical DB names first (most reliable)
            nm = (row.get("name") or "").strip()
            if nm and nm.lower() not in seen:
                med_names.append(nm)
                seen.add(nm.lower())
        if not med_names:  # nothing linked — fall back to OCR text lines
            for line in corrected.splitlines():
                nm = line.strip()
                # require a real word (>=3 letters) so numeric/symbol OCR junk
                # like "0 0" or "1-0-1" doesn't surface as a "medication".
                if len(nm) >= 3 and any(c.isalpha() for c in nm) and nm.lower() not in seen:
                    med_names.append(nm)
                    seen.add(nm.lower())
        report.prescription = PrescriptionStructured(
            medications=[PrescriptionMedication(name=nm) for nm in med_names],
            notes=(
                "Structured details (patient, doctor, date, dosage) were "
                "unavailable for this image; medications were recovered from "
                "the recognized text."
            ),
        )

    # 5) Generic recommendation
    with timer.stage("recommendation"):
        recs = recommend.recommend_for_entities(matched_rows, db)
    report.recommendations = [
        DrugRecommendation(
            prescribed=r.prescribed,
            prescribed_price=r.prescribed_price,
            composition=r.composition,
            alternatives=[GenericAlternative(**a) for a in r.alternatives],
            shap_features=[RecommendationFeature(**f) for f in r.features],
        )
        for r in recs
    ]

    # 6) Interaction detection
    with timer.stage("interactions"):
        idb = get_interaction_db()
        inter = interactions.detect_interactions(matched_rows, idb)
    report.interactions = [DrugInteraction(**i) for i in inter]

    # 7) XAI
    if settings.enable_xai:
        with timer.stage("xai"):
            artifacts = []
            gc = gradcam.gradcam_artifact(image_bgr, pre.gray)
            if gc:
                artifacts.append(gc)
            sh = shap_explain.shap_artifact(recs)
            if sh:
                artifacts.append(sh)
            lm = lime_explain.lime_artifact(corrected, entities, caps)
            if lm:
                artifacts.append(lm)
        report.xai = [XAIArtifact(**a) for a in artifacts]

    report.timings_ms = timer.timings
    report.message = (
        f"Detected {len(report.regions)} region(s) via {det_method}; "
        f"identified {len(matched_rows)} medicine(s)."
    )
    report.success = True
    return report
