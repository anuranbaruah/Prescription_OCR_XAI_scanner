"""Pydantic response/request models shared across the API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
class OCRResult(BaseModel):
    engine: str
    text: str
    confidence: Optional[float] = None
    inference_ms: Optional[float] = None


class DetectedRegion(BaseModel):
    bbox: list[int] = Field(..., description="[x1, y1, x2, y2]")
    confidence: float
    text: str = ""


# --------------------------------------------------------------------------- #
# NER
# --------------------------------------------------------------------------- #
class DrugEntity(BaseModel):
    text: str
    score: float
    start: int
    end: int
    matched_name: Optional[str] = None
    match_score: Optional[float] = None


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
class GenericAlternative(BaseModel):
    name: str
    composition: str
    manufacturer: str
    price: float
    saving_pct: float


class RecommendationFeature(BaseModel):
    feature: str
    weight: float  # SHAP-derived contribution


class DrugRecommendation(BaseModel):
    prescribed: str
    prescribed_price: Optional[float] = None
    composition: str
    alternatives: list[GenericAlternative] = []
    shap_features: list[RecommendationFeature] = []


# --------------------------------------------------------------------------- #
# Interactions
# --------------------------------------------------------------------------- #
class DrugInteraction(BaseModel):
    drug_a: str
    drug_b: str
    severity: str  # major | moderate | minor
    description: str


# --------------------------------------------------------------------------- #
# XAI
# --------------------------------------------------------------------------- #
class XAIArtifact(BaseModel):
    method: str  # grad-cam | shap | lime
    target_stage: str
    title: str
    image_base64: Optional[str] = None
    note: Optional[str] = None


# --------------------------------------------------------------------------- #
# Structured prescription (vision-LLM extraction)
# --------------------------------------------------------------------------- #
class PrescriptionMedication(BaseModel):
    name: str
    dosage: Optional[str] = None
    duration: Optional[str] = None
    frequency: Optional[str] = None
    instructions: Optional[str] = None


class PrescriptionStructured(BaseModel):
    patient_name: Optional[str] = None
    doctor_name: Optional[str] = None
    date: Optional[str] = None
    medications: list[PrescriptionMedication] = []
    notes: Optional[str] = None


# --------------------------------------------------------------------------- #
# Top-level report
# --------------------------------------------------------------------------- #
class AnalysisReport(BaseModel):
    success: bool = True
    message: str = ""
    device: str = "cpu"
    timings_ms: dict[str, float] = {}

    preprocessed_image: Optional[str] = None  # base64 PNG
    regions: list[DetectedRegion] = []
    ocr_results: list[OCRResult] = []
    extracted_text: str = ""

    prescription: Optional[PrescriptionStructured] = None  # vision-LLM structured read
    entities: list[DrugEntity] = []
    recommendations: list[DrugRecommendation] = []
    interactions: list[DrugInteraction] = []
    xai: list[XAIArtifact] = []

    # capability flags so the frontend can show what actually ran
    capabilities: dict[str, bool] = {}
