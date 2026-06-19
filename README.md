# Explainability-Driven Handwritten Medical Prescription Recognition

An end-to-end, **explainable** system that reads handwritten prescriptions,
extracts medicine names, recommends cheaper generic equivalents, flags drug
interactions, and **explains every decision** with Grad-CAM, SHAP, and LIME.

Doctors' handwriting is notoriously hard to read, and patients often overpay for
branded drugs when a cheaper, chemically identical generic exists. This project
turns a photo of a prescription into a structured, verifiable report: it detects
and transcribes the handwritten medicines, links them to a real medicine
database, suggests strength-matched generic substitutes ranked by price, checks
the prescription for known drug-drug interactions, and (unlike a black-box model)
shows *why* each result was produced through visual and feature-level
explanations. A React dashboard surfaces the full pipeline, and a FastAPI backend
serves a measured, reproducible evaluation harness.

## Features

- **Handwriting recognition** - YOLOv8 text-region detection followed by OCR.
  A fine-tuned TrOCR is the primary recognizer, with EasyOCR, Tesseract, and an
  optional vision-LLM engine for comparison.
- **Drug entity linking** - fuzzy matching of recognized text against a
  250k-row medicine catalog, guarded by a letter-driven rule that rejects
  digit-only false positives.
- **Generic recommendation** - strength-aware substitutes (same molecule *and*
  dosage) ranked by real MRP price, each with its cost saving.
- **Interaction detection** - medicine pairs checked against a DrugBank-derived
  drug-drug interaction database.
- **Explainability (XAI)** - Grad-CAM over the recognizer, SHAP over the
  recommender, and LIME over the entity linker.
- **Measured benchmarks** - an evaluation harness reports real detection, OCR,
  linking, and recommendation metrics. No numbers are hard-coded.

## Architecture

```
            +------------- React frontend (Vite, :5173) -------------+
            |  upload | results dashboard | XAI gallery | benchmarks |
            +----------------------------+---------------------------+
                                         |  /api
            +----------------------------v---------------------------+
            |              FastAPI backend (:8000)                   |
            |  preprocess -> YOLOv8 -> OCR (TrOCR/EasyOCR/Tesseract) |
            |  -> NER/linking -> DB match -> recommend + interactions|
            |  -> XAI (Grad-CAM | SHAP | LIME) -> JSON report        |
            +--------------------------------------------------------+
```

Every model is **lazy-loaded** and degrades gracefully, so the app runs even
before all heavy ML wheels finish installing (the UI shows which capabilities
are live).

## Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Copy `backend/.env.example` to `backend/.env` and fill in the values you need
(dataset paths, and an optional vision-LLM API key).

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 .

## Datasets and models

The repo ships small **sample** CSVs so everything runs out of the box. The full
datasets and the trained model weights are not committed (they are large); see
`dataset links.txt` for the data and `backend/training/` to reproduce the
models:

- `train_trocr_local.py` - fine-tune TrOCR on the handwriting datasets.
- `make_detection_dataset.py` + `train_yolo_local.py` - build a synthetic
  text-region dataset and train the YOLOv8 detector.

Point the backend at the full datasets via environment variables (see
`backend/README.md` and `backend/EVALUATION.md`).

## Evaluation

All reported metrics are produced by the harness, not hard-coded:

```bash
cd backend
python -m app.eval.run_all          # writes results/benchmarks.json
```

The frontend's Benchmarks panel shows the measured tables (and an honest
"not measured" state when a stage has not been evaluated). See
`backend/EVALUATION.md` for methodology and honest caveats (for example, the
detector is trained on synthetic pages, so its mAP is on a synthetic split).

## Notes

- First analysis is slow: TrOCR and the NER model download from Hugging Face on
  first use.
- GPU optional (4 GB+ recommended); CPU works but is slower.
- This is a decision-support tool, **not** a substitute for a pharmacist or
  physician.
