"""Central configuration.

All heavy models are lazy-loaded, so toggling these flags lets you run the
pipeline on machines without a GPU or without every optional dependency
installed. Override any field via environment variables (prefix ``RXAI_``)
or a ``.env`` file.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"


def _resolve_data_path(p: str) -> str:
    """Resolve a configured CSV path. Relative paths (e.g. from a ``.env``) are
    resolved against the backend dir so the app and eval harness agree
    regardless of the current working directory."""
    path = Path(p)
    return str(path if path.is_absolute() else (BACKEND_DIR / path).resolve())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RXAI_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    # ---- Runtime ----
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    app_name: str = "Explainable Prescription Analyzer"

    # ---- Detection ----
    # Path to a fine-tuned YOLOv8 text-region model. If missing, the detector
    # falls back to treating the whole (preprocessed) image as one region.
    yolo_weights: str = str(MODELS_DIR / "yolov8n_rx.pt")
    yolo_conf: float = 0.25

    # ---- OCR ----
    # Primary OCR engine: "vision" | "trocr" | "easyocr" | "tesseract".
    # A multimodal LLM reads full handwritten prescriptions far better than the
    # pretrained TrOCR baseline; it is the deployed primary when a key is
    # configured, with automatic fallback to TrOCR otherwise.
    primary_ocr: str = "vision"
    trocr_model: str = "microsoft/trocr-base-handwritten"
    enable_easyocr: bool = True
    enable_tesseract: bool = True
    enable_trocr: bool = True

    # ---- Vision LLM OCR (any OpenAI-compatible API: Groq, OpenRouter, OpenAI, ...) ----
    # Default points at Groq, which is free and needs no credit card.
    enable_vision: bool = True
    vision_api_key: str = ""  # RXAI_VISION_API_KEY (or GROQ_API_KEY / OPENAI_API_KEY)
    vision_base_url: str = "https://api.groq.com/openai/v1"
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # ---- NER ----
    # A token-classification BioBERT checkpoint. Default is a public
    # biomedical NER model; swap for your fine-tuned i2b2 checkpoint.
    ner_model: str = "d4data/biomedical-ner-all"
    fuzzy_threshold: int = 85  # RapidFuzz score (0-100) for DB matching

    # ---- Data files (sample fallbacks bundled in repo) ----
    medicines_csv: str = str(DATA_DIR / "sample_medicines.csv")
    interactions_csv: str = str(DATA_DIR / "sample_interactions.csv")

    # ---- Recommendation ----
    top_k_generics: int = 3

    # ---- XAI ----
    enable_xai: bool = True

    # ---- CORS ----
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @property
    def resolved_vision_key(self) -> str:
        """Vision-LLM API key from config, falling back to common env vars."""
        import os

        return (
            self.vision_api_key
            or os.environ.get("GROQ_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )

    @property
    def medicines_csv_path(self) -> str:
        return _resolve_data_path(self.medicines_csv)

    @property
    def interactions_csv_path(self) -> str:
        return _resolve_data_path(self.interactions_csv)

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
