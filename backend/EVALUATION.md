# Evaluation Harness

This document explains how the project's **measured** results are produced —
the prerequisite for any IEEE journal/conference submission. Every number the
API and UI report comes from this harness. Nothing is hard-coded.

> **Why this exists.** An earlier version of the codebase shipped the synopsis'
> *target* figures (TrOCR WER 6.2 %, BioBERT F1 0.923, etc.) as if they were
> measured. Reporting unmeasured numbers as experimental results is fabrication
> and will sink a submission. The harness replaces those placeholders with real
> measurements you can reproduce.

---

## TL;DR

```bash
cd backend
.venv\Scripts\activate
python -m app.eval.make_demo_ocr        # one-time: build the demo OCR images
python -m app.eval.run_all              # runs everything, writes results/benchmarks.json
```

The UI's **Benchmarks** panel then shows the measured tables with a
"Measured results" status badge. With no results file present it shows
"Not measured" and empty tables — never invented figures.

Run a single stage:

```bash
python -m app.eval.run_ocr_eval
python -m app.eval.run_ner_eval
python -m app.eval.run_recommend_eval
python -m app.eval.run_interaction_eval
```

---

## What is measured, and how

| Stage | Metric | Method | Module |
|-------|--------|--------|--------|
| OCR | WER, CER, latency | Levenshtein vs ground-truth transcription, micro-averaged | `run_ocr_eval` |
| NER | Precision/Recall/F1 | Micro-averaged over matched canonical drug names (curated reference lexicon) | `run_ner_eval` |
| Recommendation | Coverage, cost saving, equivalence precision | Real catalog: fraction of brands with a cheaper **same-strength** generic; saving from real MRP prices | `run_recommend_eval` |
| Interactions | Recall/Precision/F1 | Deployed detector vs real DrugBank DDI pairs mapped to real brands | `run_interaction_eval` |
| Recognition→generic | Word / linking / generic accuracy | End-to-end: BD handwriting image → OCR → linker → generic, vs the BD `GENERIC_NAME` ground truth | `bd_truth` |
| End-to-end | Mean latency | Full pipeline wall-clock per image | `run_all` |
| Detection (mAP) | **not measured** | Needs fine-tuned YOLOv8 + box-annotated test split (see below) | — |

Metric definitions live in `app/eval/metrics.py` (standard edit-distance WER/CER
and micro-averaged set P/R/F1).

---

## Dataset manifests

The evaluators read small, explicit manifests. The bundled **demo** sets live in
`backend/data/eval/`. Point an evaluator at a different directory with an
environment variable to run on the full public datasets.

| Set | Env var | Files | Format |
|-----|---------|-------|--------|
| OCR | `RXAI_EVAL_OCR_DIR` | `manifest.csv` + `images/` | columns `image,text` |
| NER | `RXAI_EVAL_NER_DIR` | `labeled.jsonl` | `{"text": "...", "drugs": ["Crocin 500mg", ...]}` |
| Recommend | `RXAI_EVAL_RECOMMEND_DIR` | `cases.jsonl` | `{"brand": "...", "expected_generic": "..."}` |
| Interactions | `RXAI_EVAL_INTERACTION_DIR` | `pairs.jsonl` | `{"drugs": ["A", "B"], "interacts": true}` |

NER `drugs` and recommend `brand`/`expected_generic` must be **canonical names
as they appear in the medicine DB** (`name` column), so matched outputs are
comparable.

---

## Scaling to the full public datasets (for real journal numbers)

The bundled demo sets are deliberately tiny and, for OCR, **printed rather than
handwritten** — they exist to validate the harness, not to be cited. To produce
publishable numbers:

1. **Handwritten Rx OCR.** Two real word-level handwritten medicine-name
   datasets are already present under `<project>/dataset`:
   *Handwritten Rx* (1,115 test images) and *Doctor's Handwritten Prescription
   BD* (780 test images, 78 classes, also carries `GENERIC_NAME`). Generate
   their OCR manifests once, then evaluate (optionally subsampled):
   ```bash
   python -m app.eval.prepare_ocr_manifests
   set RXAI_EVAL_OCR_DIR=../dataset/Handwritten Rx
   set RXAI_EVAL_OCR_LIMIT=300      # omit to run the full test split
   python -m app.eval.run_ocr_eval
   ```
   These are **word-level** sets (one handwritten medicine name per image), so
   the reported headline metrics are **exact-match word accuracy** and **CER**;
   WER is included but is noisy on single-word references.

2. **Medicine DB + recommendation — WIRED.** The real Kaggle *A-Z Medicine
   Dataset of India* (~254k rows, real MRP prices + split compositions) is wired
   via `backend/.env`:
   ```ini
   RXAI_MEDICINES_CSV=../dataset/A_Z_medicines_dataset_of_India.csv
   ```
   The catalog has **no brand→generic ground-truth label**, so a labelled
   accuracy is not honestly computable on it. `run_recommend_eval` instead
   reports, over a fixed-seed sample of `RXAI_EVAL_RECOMMEND_N` brands (default
   500), what *is* objectively checkable:
   - **substitution coverage** — fraction of prescribed brands for which a
     cheaper **same-strength** generic exists (dosage-aware: a 250 mg drug is not
     a substitute for a 500 mg prescription — see `strength_composition`);
   - **cost saving** — mean/median/max % vs the real MRP of the cheaper generic;
   - **equivalence precision** — returned alternatives share the dosage-aware
     composition key (a validation check; ~1.0 by construction).

   *Caveats:* coverage/saving are descriptive properties of the catalog, not a
   correctness label. Formulation is **not** yet matched (a same-mg *suspension*
   can be offered for a *tablet*); discontinued drugs are excluded from
   alternatives.

   **End-to-end recognition→generic accuracy (real ground truth) — `bd_truth`.**
   The catalog has no brand→generic label, but the *Doctor's BD* dataset does
   (`MEDICINE_NAME → GENERIC_NAME`, human-authored, e.g. `Aceta → Paracetamol`).
   Those brands are Bangladeshi and are absent from the Indian catalog, so they
   are scored against BD used as its **own** brand→generic knowledge base (what a
   deployed system would carry for that market). For every BD **test** image the
   evaluator runs each OCR engine, links the (garbled) text back to a brand with
   the deployed fuzzy linker (`MedicineDB.find`), and checks the linked generic
   against the label:
   ```bash
   python -m app.eval.bd_truth                 # full BD test split, all engines
   set RXAI_EVAL_BD_LIMIT=150                   # quick fixed-seed subsample
   python -m app.eval.bd_truth
   ```
   Three metrics per engine: **word accuracy** (raw OCR), **linking accuracy**
   (OCR→correct brand), and **generic accuracy** (the clinical endpoint:
   image→correct generic). Generic accuracy is typically **higher than word
   accuracy** because entity linking recovers OCR spelling errors — that recovery
   margin is the contribution (prior handwritten-Rx papers stop at CER and never
   report a downstream generic accuracy). This is wired into `run_all` (block
   `recognition_to_generic`; tune with `--bd-limit`, disable with `--bd-limit 0`).

3. **Interactions — WIRED.** The real DrugBank-derived DDI list (~191k
   ingredient-level pairs) is wired via `backend/.env`:
   ```ini
   RXAI_INTERACTIONS_CSV=../dataset/db_drug_interactions.csv
   ```
   `run_interaction_eval` builds a balanced set (`RXAI_EVAL_INTERACTION_N`,
   default 200) of positive/negative pairs **drawn from the real DDI list**
   intersected with catalog ingredients, then maps each ingredient to a real
   single-ingredient brand and runs the **deployed** detector
   (brand → composition → active ingredients → DDI lookup). P/R/F1 therefore
   measure our brand→ingredient extraction fidelity against a known knowledge
   base. *Caveat:* this is **near-ceiling by construction** (positives are built
   from cleanly mappable single-ingredient brands) — it validates the pipeline
   plumbing at scale, it is not evidence the DDI list itself is complete. Only
   ~631 of the catalog's distinct ingredient tokens exactly match DDI names
   (spelling variants like *Amoxycillin*/*Amoxicillin* and dosage/salt fragments
   reduce overlap); normalising those is the next recall lever.

4. **NER.** The NER entity-*linking* ablation deliberately links against the
   **curated reference lexicon** (`data/sample_medicines.csv`), not the full
   wired catalog: the 12-example gold set is authored against those canonical
   names, so the greedy-vs-letter-guard precision ablation is only meaningful
   when gold labels and the linker's lexicon share naming. (Linking the
   sample-authored gold against the 254k real catalog yields F1≈0 purely from
   canonical-name mismatch — a measurement artifact, not a regression.) To run
   the ablation against the real catalog, author a real-catalog gold set and set
   `RXAI_EVAL_NER_DB=../dataset/A_Z_medicines_dataset_of_India.csv`. Annotating a
   prescription-text test set (drug spans from the OCR ground-truth) and
   expanding `ner/labeled.jsonl` is the path to a larger NER benchmark.

Re-run `python -m app.eval.run_all` and the measured tables update.

---

## Known gaps that affect what you can claim

These are real findings from running the harness — address them before writing
the results section:

- **Drug identification is entity-linking, not general neural NER — and that's
  the defensible design.** Off-the-shelf `d4data/biomedical-ner-all` classifies
  tokens like *"Tab Crocin"* as `Diagnostic_procedure` and produces **zero**
  DRUG entities (F1 = 0.000) — general biomedical NER does not recognise Indian
  brand names, which are proper nouns from a closed formulary. The effective
  method is a **lexicon entity-linker** (fuzzy-match OCR tokens to the medicine
  DB). Its precision was fixed with a **letter-driven guard** that rejects
  digit-driven false positives (a `"500 1-0-1"` fragment matching *Glucophage
  500* on the shared dosage number):

  | NER method | P | R | F1 |
  |---|---|---|---|
  | Lexicon linker (greedy) | 0.708 | 1.000 | 0.829 |
  | Lexicon linker (+letter guard) | **1.000** | **1.000** | **1.000** |
  | BioBERT NER → DB | 0.000 | 0.000 | 0.000 |
  | Combined (deployed) | 1.000 | 1.000 | 1.000 |

  (Measured on the 12-example gold set — small; re-measure on a larger annotated
  prescription set before citing.) A neural NER only adds value for drugs **not**
  in the formulary; to claim that, fine-tune on a drug-NER corpus (i2b2
  medication / BC5CDR-chem) and evaluate on out-of-DB drug mentions.

- **Detection mAP cannot be reported** until a YOLOv8 text-region model is
  fine-tuned and evaluated against a box-annotated test split. The pipeline
  currently uses a classical morphology fallback. Either fine-tune YOLOv8
  (needs more GPU than the local GTX 1650 — use Colab/Kaggle) or reframe the
  paper to use the morphology segmenter and drop the detection-comparison table.

- **Pretrained OCR is far too weak on this task — fine-tuning is the
  contribution.** Measured on the full test splits: on *Handwritten Rx* (1,115
  imgs) TrOCR-base-handwritten gets **9.9 % word accuracy / 70 % CER**; on
  *Doctor's BD* (780 imgs) **13.2 % / 59 % CER** (EasyOCR is best on BD at
  10.6 % / 54.6 % CER). Off-the-shelf checkpoints (trained on English sentences,
  not medicine names) cannot read these. The OCR result for the paper comes from
  **fine-tuning** TrOCR — a ready-to-run Colab notebook is at
  `backend/training/finetune_trocr_colab.ipynb` (see `training/README.md`). It
  trains on both datasets' train splits and saves a checkpoint; re-measure by
  setting `RXAI_TROCR_MODEL` to the downloaded model and re-running the harness.
  Report exact-match word accuracy + CER, baseline vs fine-tuned.

- **Recommendation now measured on the real catalog (254k drugs).** Over 500
  sampled brands: **97.8 % substitution coverage**, **72.0 % mean / 79.6 %
  median cost saving** (max 99.7 %), equivalence precision 1.0. These are
  honest descriptive properties of the catalog, not a labelled accuracy (no
  brand→generic ground truth exists in *that* catalog). A real labelled
  brand→generic accuracy **is** now measured separately on the *Doctor's BD*
  `GENERIC_NAME` ground truth via the end-to-end `bd_truth` evaluator (see
  above). Remaining lever for a stronger catalog claim: match **formulation**
  (tablet vs suspension).

- **Interaction P/R/F1 = 1.0 on 200 real DrugBank pairs**, but **near-ceiling by
  construction** — positives are real DDI pairs mapped to cleanly resolvable
  single-ingredient brands, so the score validates the deployed
  brand→ingredient→DDI pipeline at scale rather than proving DDI-list
  completeness. Frame it that way in the paper. The honest open problem is
  ingredient-name normalisation: only ~631 catalog ingredient tokens exactly
  match DDI names (spelling variants + dosage/salt fragments), which caps recall
  on free-form multi-ingredient prescriptions.
