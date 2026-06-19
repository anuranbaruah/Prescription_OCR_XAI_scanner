# Backend — Explainable Prescription Analyzer

FastAPI service implementing the full pipeline:

```
upload → preprocess → YOLOv8 detect → OCR (TrOCR/EasyOCR/Tesseract)
       → BioBERT NER → DB match → generic recommendation + interaction check
       → XAI (Grad-CAM / SHAP / LIME) → JSON report
```

Every heavy model is **lazy-loaded** and **degrades gracefully**: if a
dependency or model file is missing, that stage is skipped and the report's
`capabilities` map says so. This means the server runs even before all wheels
finish installing.

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

> GPU (optional): install a CUDA torch build first:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

> Tesseract: install the system binary and ensure `tesseract` is on PATH
> (Windows: https://github.com/UB-Mannheim/tesseract/wiki).

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

- Health / capabilities: http://localhost:8000/api/health
- Swagger docs: http://localhost:8000/docs

## Datasets

The repo ships small **sample** CSVs in `data/` so the app runs out of the box.
For full results, download the real datasets and point the env vars at them:

| Env var | Dataset | Source |
|---|---|---|
| `RXAI_MEDICINES_CSV` | 11000 Medicine Details | Kaggle: `singhnavjot2062001/11000-medicine-details` |
| `RXAI_INTERACTIONS_CSV` | Drug-Drug Interactions | DrugBank / Kaggle DDI |
| `RXAI_YOLO_WEIGHTS` | Fine-tuned YOLOv8 text detector | Train on Kaggle Rx images |

Expected columns:
- **medicines**: `name, composition, manufacturer, price` (extra cols ignored)
- **interactions**: `drug_a, drug_b, severity, description`

## Evaluation (measured benchmarks)

All reported metrics come from a reproducible harness — see
[`EVALUATION.md`](EVALUATION.md). The `/api/model-comparison` endpoint and the
UI's Benchmarks panel serve **only measured results**
(`results/benchmarks.json`); until the harness is run they honestly report
"not measured" rather than placeholder numbers.

```bash
python -m app.eval.make_demo_ocr   # one-time: build the demo OCR images
python -m app.eval.run_all         # measure everything -> results/benchmarks.json
```
