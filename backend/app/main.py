"""FastAPI entrypoint for the Explainable Prescription Analyzer."""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .schemas import AnalysisReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rxai")

# Models (TrOCR, EasyOCR, BioBERT) lazy-load on first use — that one-time GPU
# load is ~40 s. We warm them in a background thread at startup so the first
# real analysis is already hot. WARMUP tracks progress for the UI/health probe.
WARMUP = {"status": "cold"}  # cold -> warming -> ready | error


def _warmup() -> None:
    WARMUP["status"] = "warming"
    try:
        import cv2
        import numpy as np

        from .data.loader import get_interaction_db, get_medicine_db
        from .pipeline.orchestrator import run_pipeline

        get_medicine_db()
        get_interaction_db()
        # one tiny pass to load every model + build caches end-to-end
        blank = np.full((160, 480, 3), 255, np.uint8)
        run_pipeline(cv2.imencode(".png", blank)[1].tobytes())
        WARMUP["status"] = "ready"
        logger.info("Model warmup complete — analyses will be fast now.")
    except Exception as exc:  # never block serving on a warmup failure
        WARMUP["status"] = "error"
        logger.warning("Model warmup failed (models will load on first request): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warmup, name="warmup", daemon=True).start()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Handwritten prescription OCR + generic drug recommendation with XAI.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    """Liveness probe + which optional capabilities are available."""
    from .pipeline.capabilities import probe_capabilities

    return {
        "status": "ok",
        "device": settings.resolved_device,
        "primary_ocr": settings.primary_ocr,
        "warmup": WARMUP["status"],
        "capabilities": probe_capabilities(),
    }


@app.get("/api/model-comparison")
def model_comparison() -> dict:
    """Measured benchmark tables (from the evaluation harness) for the report UI.

    Returns an honest 'not measured' payload until `python -m app.eval.run_all`
    has been run — never fabricated figures.
    """
    from .pipeline.benchmarks import get_benchmarks

    return get_benchmarks()


@app.post("/api/analyze", response_model=AnalysisReport)
async def analyze(
    file: UploadFile = File(...),
) -> AnalysisReport:
    """Run the full pipeline on an uploaded prescription image or PDF."""
    from .utils import is_pdf

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    ctype = file.content_type or ""
    is_image = ctype.startswith("image/")
    is_pdf_file = ctype == "application/pdf" or is_pdf(data)
    if ctype and not is_image and not is_pdf_file:
        raise HTTPException(
            status_code=400, detail="Please upload an image or PDF file."
        )

    # Imported lazily so the server boots even while heavy deps install.
    from .pipeline.orchestrator import run_pipeline

    try:
        return run_pipeline(data)
    except Exception as exc:  # surface a clean error to the client
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc


@app.get("/")
def root() -> dict:
    return {"name": settings.app_name, "docs": "/docs", "health": "/api/health"}
