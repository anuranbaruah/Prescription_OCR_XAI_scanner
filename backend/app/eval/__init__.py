"""Evaluation harness for the prescription-analysis pipeline.

This package produces *measured* metrics on held-out data — WER/CER for OCR,
precision/recall/F1 for NER, composition-match accuracy for the recommender,
and recall for interaction detection. The runner scripts write their output to
``backend/results/benchmarks.json``, which the API and UI consume.

Nothing here invents numbers. If you have not run the evaluators, the API
reports the metrics as "not measured" rather than showing placeholder figures.
"""
