"""Stage 3 — OCR text extraction (synopsis 7.3).

Engines (all optional, lazy-loaded):
  * TrOCR     — primary; ViT encoder + RoBERTa decoder (handwritten checkpoint)
  * EasyOCR   — comparison engine
  * Tesseract — open-source baseline (needs system binary)

Post-processing: SymSpell spell-correction with a medical dictionary built
from the medicine database, so OCR'd drug names get nudged toward real names.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ..config import settings
from .detection import Region

logger = logging.getLogger("rxai.ocr")

# lazy singletons
_trocr = None
_easyocr = None
_symspell = None
_vision = None

# Prompt for the vision-LLM engine: transcribe the prescription so the
# downstream entity-linker can match drug names. We ask for one medicine per
# line and faithful spelling/dosage, no commentary.
_VISION_PROMPT = (
    "You are an expert pharmacist transcribing a doctor's handwritten "
    "prescription. Read the image and output ONLY the medicines, one per line, "
    "each as the drug name followed by its strength and dosage exactly as "
    "written (e.g. 'Augmentin 625 1-0-1'). Do not add headings, numbering, "
    "explanations, or any text that is not on the prescription. If a token is "
    "illegible, give your best single reading."
)


@dataclass
class EngineOutput:
    engine: str
    text: str
    confidence: float | None
    inference_ms: float


# --------------------------------------------------------------------------- #
# TrOCR (primary)
# --------------------------------------------------------------------------- #
def load_trocr():
    """Load (and cache) the TrOCR processor+model. Shared with the XAI stage
    so we never hold two copies in (limited) GPU memory."""
    global _trocr
    if _trocr is not None:
        return _trocr
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    logger.info("Loading TrOCR: %s", settings.trocr_model)
    processor = TrOCRProcessor.from_pretrained(settings.trocr_model)
    model = VisionEncoderDecoderModel.from_pretrained(settings.trocr_model)
    model.to(settings.resolved_device).eval()
    _trocr = (processor, model)
    return _trocr


# backward-compatible alias
_load_trocr = load_trocr


# Max region crops fed to TrOCR in a single forward pass. Batching amortizes
# GPU kernel-launch/decoder overhead (the big latency win); the cap bounds VRAM
# so many regions don't OOM the 4 GB card.
TROCR_BATCH = 8


def _trocr_on_region(crop_bgr: np.ndarray) -> str:
    """Single-crop inference — kept for the XAI stage; the page path batches."""
    return _trocr_batch([crop_bgr])[0]


def _trocr_batch(crops_bgr: list[np.ndarray]) -> list[str]:
    """Transcribe several crops in chunked batches (one forward pass per chunk)."""
    import torch

    processor, model = _load_trocr()
    out: list[str] = []
    for i in range(0, len(crops_bgr), TROCR_BATCH):
        chunk = crops_bgr[i : i + TROCR_BATCH]
        images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in chunk]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(settings.resolved_device)
        with torch.no_grad():
            ids = model.generate(pixel_values, max_new_tokens=64)
        out.extend(t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True))
    return out


def run_trocr(regions: list[Region]) -> EngineOutput:
    start = time.perf_counter()
    lines: list[str] = []
    try:
        lines = _trocr_batch([r.crop for r in regions])
    except Exception as exc:  # fall back to per-region so one bad crop isn't fatal
        logger.warning("Batched TrOCR failed (%s); retrying per region", exc)
        for r in regions:
            try:
                lines.append(_trocr_on_region(r.crop))
            except Exception as exc2:
                logger.warning("TrOCR region failed: %s", exc2)
    text = "\n".join(l for l in lines if l)
    return EngineOutput("TrOCR", text, None, round((time.perf_counter() - start) * 1000, 1))


# --------------------------------------------------------------------------- #
# Vision LLM via any OpenAI-compatible API (Groq / OpenRouter / OpenAI / ...)
# --------------------------------------------------------------------------- #
def _load_vision():
    """Create (and cache) an OpenAI-compatible client. None if unavailable."""
    global _vision
    if _vision is not None:
        return _vision
    key = settings.resolved_vision_key
    if not key:
        return None
    try:
        from openai import OpenAI

        # Bounded so a slow/flaky provider can never stall an analysis: short
        # timeout + a single retry, then we fall back to the local engines.
        _vision = OpenAI(
            api_key=key, base_url=settings.vision_base_url,
            timeout=20.0, max_retries=1,
        )
        return _vision
    except Exception as exc:
        logger.warning("Vision-LLM client unavailable: %s", exc)
        return None


_STRUCT_PROMPT = (
    "You are an expert medical scribe reading a doctor's handwritten "
    "prescription. Extract its contents as JSON with EXACTLY this shape:\n"
    '{"patient_name": string|null, "doctor_name": string|null, '
    '"date": string|null, "medications": [{"name": string, "dosage": '
    'string|null, "duration": string|null, "frequency": string|null, '
    '"instructions": string|null}], "notes": string|null}\n'
    "Read the handwriting carefully. Use null when a field is genuinely not "
    "present (do not invent values). 'name' is the medicine/brand name. "
    "'frequency' is how often to take it (e.g. 'Once daily, before food'). "
    "Put any complaints/diagnosis/advice in 'notes'. Output ONLY the JSON."
)


def extract_prescription(image_bgr: np.ndarray) -> dict | None:
    """Structured prescription extraction via the vision LLM (JSON).

    Retries once when the call errors or comes back with no medications — a
    transient Groq hiccup or an occasional empty parse shouldn't drop the
    prescription card. Returns None only if both attempts fail.
    """
    import base64
    import json

    client = _load_vision()
    if client is None:
        return None
    png = cv2.imencode(".png", image_bgr)[1].tobytes()
    data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")

    def _attempt() -> dict | None:
        try:
            resp = client.chat.completions.create(
                model=settings.vision_model,
                max_tokens=1024,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _STRUCT_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            )
            data = json.loads(resp.choices[0].message.content or "{}")
        except Exception as exc:
            logger.warning("Structured prescription extraction failed: %s", exc)
            return None
        if not isinstance(data, dict):
            return None
        meds = data.get("medications")
        data["medications"] = [
            m for m in meds if isinstance(m, dict) and str(m.get("name") or "").strip()
        ] if isinstance(meds, list) else []
        return data

    for attempt in range(2):
        data = _attempt()
        if data and data.get("medications"):
            return data
    return data  # last attempt's result (may be empty-meds or None)


def run_vision_llm(image_bgr: np.ndarray) -> EngineOutput:
    """Transcribe the full prescription image via an OpenAI-compatible vision API."""
    import base64

    start = time.perf_counter()
    client = _load_vision()
    if client is None:
        return EngineOutput("Vision LLM", "", None, 0.0)
    png = cv2.imencode(".png", image_bgr)[1].tobytes()
    data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    try:
        resp = client.chat.completions.create(
            model=settings.vision_model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Vision-LLM OCR failed: %s", exc)
        text = ""
    return EngineOutput(
        "Vision LLM", text, None, round((time.perf_counter() - start) * 1000, 1)
    )


# --------------------------------------------------------------------------- #
# EasyOCR (comparison)
# --------------------------------------------------------------------------- #
def _load_easyocr():
    global _easyocr
    if _easyocr is not None:
        return _easyocr
    import easyocr

    gpu = settings.resolved_device == "cuda"
    _easyocr = easyocr.Reader(["en"], gpu=gpu)
    return _easyocr


def run_easyocr(image_bgr: np.ndarray) -> EngineOutput:
    start = time.perf_counter()
    reader = _load_easyocr()
    results = reader.readtext(image_bgr)
    lines = [text for _, text, conf in results]
    confs = [conf for _, _, conf in results]
    text = "\n".join(lines)
    conf = float(np.mean(confs)) if confs else None
    return EngineOutput("EasyOCR", text, conf, round((time.perf_counter() - start) * 1000, 1))


# --------------------------------------------------------------------------- #
# Tesseract (baseline)
# --------------------------------------------------------------------------- #
def run_tesseract(gray: np.ndarray) -> EngineOutput:
    import pytesseract

    start = time.perf_counter()
    text = pytesseract.image_to_string(gray)
    return EngineOutput(
        "Tesseract", text.strip(), None, round((time.perf_counter() - start) * 1000, 1)
    )


# --------------------------------------------------------------------------- #
# SymSpell post-correction
# --------------------------------------------------------------------------- #
def _load_symspell(dictionary_terms: list[str]):
    global _symspell
    if _symspell is not None:
        return _symspell
    from symspellpy import SymSpell

    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    for term in dictionary_terms:
        for word in str(term).lower().split():
            if word.isalpha() and len(word) > 2:
                sym.create_dictionary_entry(word, 1)
    _symspell = sym
    return _symspell


def correct_text(text: str, dictionary_terms: list[str]) -> str:
    """Lightweight per-word correction toward the medical dictionary."""
    if not text.strip() or not dictionary_terms:
        return text
    try:
        from symspellpy import Verbosity

        sym = _load_symspell(dictionary_terms)
        out_lines = []
        for line in text.splitlines():
            words = []
            for w in line.split():
                if w.isalpha() and len(w) > 2:
                    sug = sym.lookup(w.lower(), Verbosity.CLOSEST, max_edit_distance=2)
                    words.append(sug[0].term if sug else w)
                else:
                    words.append(w)
            out_lines.append(" ".join(words))
        return "\n".join(out_lines)
    except Exception as exc:
        logger.warning("SymSpell correction skipped: %s", exc)
        return text


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_ocr(
    image_bgr: np.ndarray,
    gray: np.ndarray,
    regions: list[Region],
    caps: dict[str, bool],
) -> tuple[list[EngineOutput], EngineOutput]:
    """Run every available engine; return (all_outputs, primary_output)."""
    outputs: list[EngineOutput] = []

    # NOTE: the Vision LLM runs once in the orchestrator (concurrently with these
    # engines) and is merged into the results there — we do NOT call it here, to
    # avoid a duplicate network request. The local engines below always run.
    if caps.get("trocr"):
        try:
            outputs.append(run_trocr(regions))
        except Exception as exc:
            logger.warning("TrOCR unavailable: %s", exc)
    if caps.get("easyocr"):
        try:
            outputs.append(run_easyocr(image_bgr))
        except Exception as exc:
            logger.warning("EasyOCR unavailable: %s", exc)
    if caps.get("tesseract"):
        try:
            outputs.append(run_tesseract(gray))
        except Exception as exc:
            logger.warning("Tesseract unavailable: %s", exc)

    if not outputs:
        outputs.append(EngineOutput("none", "", None, 0.0))

    # Choose primary by configured preference, but only if it actually produced
    # text; otherwise fall back to the first engine that did (so a missing vision
    # key, say, transparently degrades to TrOCR instead of an empty report).
    pref = {"vision": "Vision LLM", "trocr": "TrOCR", "easyocr": "EasyOCR", "tesseract": "Tesseract"}
    want = pref.get(settings.primary_ocr, "Vision LLM")
    preferred = next((o for o in outputs if o.engine == want and o.text.strip()), None)
    primary = preferred or next((o for o in outputs if o.text.strip()), outputs[0])
    return outputs, primary
