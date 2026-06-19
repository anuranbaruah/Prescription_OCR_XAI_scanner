"""Run the full evaluation harness and write ``results/benchmarks.json``.

Aggregates every measured table (OCR, NER, recommendation, interactions) plus
an optional end-to-end latency measurement into the JSON the API/UI serve.
Anything that cannot be measured on the current setup (e.g. detection mAP,
which needs fine-tuned YOLO weights + box-annotated test images) is emitted as
an explicit "not measured" row rather than a fabricated number.

Run:  python -m app.eval.run_all                 # full, with latency
      python -m app.eval.run_all --no-latency    # skip end-to-end timing
      python -m app.eval.run_all \\
          --ocr-dir "../dataset/Handwritten Rx" \\
          --ocr-dir "../dataset/Doctor's ... BD dataset/Testing"
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os

from ..config import settings
from . import (
    bd_truth,
    run_interaction_eval,
    run_ner_eval,
    run_ocr_eval,
    run_recommend_eval,
)
from .datasets import load_ocr_samples
from .results_store import save_benchmarks


def _detection_table() -> dict:
    """Detection mAP requires fine-tuned YOLO weights + box-annotated test
    images. Until those exist we report honestly rather than inventing mAP."""
    import os
    from pathlib import Path

    has_weights = Path(settings.yolo_weights).exists()
    return {
        "title": "Text Region Detection (not measured)",
        "columns": ["Model", "mAP@0.5", "mAP@0.5:0.95", "Inference (ms)"],
        "rows": [],
        "best_row": None,
        "measured": False,
        "note": (
            "Pipeline currently uses a morphology fallback for region detection. "
            "mAP requires fine-tuned YOLOv8 weights and a box-annotated test split "
            + ("(weights present — add annotated test set to evaluate)."
               if has_weights else "(no fine-tuned weights present).")
        ),
    }


def _eval_ocr_datasets(ocr_dirs: list[str], limit: int | None) -> list[dict]:
    """Evaluate OCR on each dataset dir in turn (models are cached between
    runs). Returns one measured table per dataset, in the given order."""
    tables: list[dict] = []
    for d in ocr_dirs:
        if d:
            os.environ["RXAI_EVAL_OCR_DIR"] = d
        else:
            os.environ.pop("RXAI_EVAL_OCR_DIR", None)
        if limit:
            os.environ["RXAI_EVAL_OCR_LIMIT"] = str(limit)
        else:
            os.environ.pop("RXAI_EVAL_OCR_LIMIT", None)
        table = run_ocr_eval.evaluate()
        print(f"  [{table.get('source')}] n={table['n_samples']}")
        for i, row in enumerate(table["rows"]):
            mark = "*" if i == table["best_row"] else " "
            print(f"   {mark} {row[0]:<22} Acc={row[1]}%  CER={row[3]}%  ({row[4]}ms)")
        tables.append(table)
    return tables


def _latency(n_samples: int) -> dict:
    """Measure real end-to-end pipeline latency on a few images."""
    import cv2

    from ..pipeline.orchestrator import run_pipeline
    from ..utils import imread_unicode

    samples = load_ocr_samples()[:n_samples]
    times: list[float] = []
    for s in samples:
        bgr = imread_unicode(s.image_path)
        if bgr is None:
            continue
        data = cv2.imencode(".png", bgr)[1].tobytes()
        rep = run_pipeline(data)
        times.append(sum(rep.timings_ms.values()))
    mean_s = (sum(times) / len(times) / 1000.0) if times else 0.0
    return {"mean_end_to_end_s": round(mean_s, 2), "n_samples": len(times)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-latency", action="store_true", help="skip end-to-end timing")
    parser.add_argument("--latency-samples", type=int, default=3)
    parser.add_argument(
        "--ocr-dir", action="append", default=[],
        help="OCR dataset dir (repeatable). First is the primary table; "
             "others are stored alongside. Defaults to RXAI_EVAL_OCR_DIR / demo.",
    )
    parser.add_argument(
        "--ocr-limit", type=int, default=None,
        help="subsample N images per OCR dataset (default: full test split)",
    )
    parser.add_argument(
        "--reuse-ocr", action="store_true",
        help="reuse OCR tables + latency from the existing benchmarks.json "
             "instead of re-measuring (fast refresh of the other stages)",
    )
    parser.add_argument(
        "--bd-limit", type=int, default=None,
        help="subsample N BD images for the end-to-end recognition->generic "
             "eval (default: full BD test split; 0 disables it)",
    )
    args = parser.parse_args()

    prev_latency = None
    prev_r2g = None
    if args.reuse_ocr:
        from .results_store import load_benchmarks
        prev = load_benchmarks() or {}
        prev_r2g = prev.get("recognition_to_generic")
        ocr_tables = prev.get("ocr_datasets") or ([prev["ocr"]] if prev.get("ocr") else [])
        assert ocr_tables, "--reuse-ocr needs an existing benchmarks.json with OCR results"
        print("== OCR (reused from benchmarks.json) ==")
        for t in ocr_tables:
            print(f"  [{t.get('source')}] n={t.get('n_samples')}")
        # Recover the previously measured latency row, if any.
        for row in prev.get("system", {}).get("rows", []):
            if str(row[0]).startswith("Mean end-to-end"):
                val = str(row[1]).split()[0]
                prev_latency = {"mean_end_to_end_s": float(val),
                                "n_samples": prev.get("_latency_n", 3)}
        args.no_latency = True
    else:
        print("== OCR ==")
        ocr_dirs = args.ocr_dir or [os.environ.get("RXAI_EVAL_OCR_DIR", "")]
        ocr_tables = _eval_ocr_datasets(ocr_dirs, args.ocr_limit)
    ocr = ocr_tables[0]  # primary

    print("\n== NER =="); ner = run_ner_eval.evaluate(); run_ner_eval.main()
    print("\n== Recommendation =="); rec = run_recommend_eval.evaluate(); run_recommend_eval.main()
    print("\n== Interactions =="); inter = run_interaction_eval.evaluate(); run_interaction_eval.main()

    # End-to-end recognition -> generic on the BD brand->generic ground truth.
    # Reuses OCR models, so it's skipped under --reuse-ocr (the prior block is
    # carried forward) and when --bd-limit 0 is passed.
    r2g = prev_r2g
    if not args.reuse_ocr and args.bd_limit != 0:
        print("\n== End-to-end recognition -> generic (BD ground truth) ==")
        if args.bd_limit:
            os.environ["RXAI_EVAL_BD_LIMIT"] = str(args.bd_limit)
        else:
            os.environ.pop("RXAI_EVAL_BD_LIMIT", None)
        r2g = bd_truth.evaluate()
        if r2g.get("measured"):
            bd_truth.main()
        else:
            print(f"  (not measured: {r2g.get('reason', 'no BD dataset')})")

    latency = prev_latency
    if not args.no_latency:
        print("\n== End-to-end latency ==")
        # Measure latency on the primary OCR dataset.
        if ocr_dirs[0]:
            os.environ["RXAI_EVAL_OCR_DIR"] = ocr_dirs[0]
        os.environ.pop("RXAI_EVAL_OCR_LIMIT", None)
        latency = _latency(args.latency_samples)
        print(f"  mean end-to-end: {latency['mean_end_to_end_s']} s (n={latency['n_samples']})")

    # ---- assemble the system summary from measured numbers only ----
    sys_rows = []
    if ocr["rows"] and ocr["best_row"] is not None:
        best = ocr["rows"][ocr["best_row"]]
        sys_rows.append([
            f"Best OCR ({best[0]})",
            f"CER {best[3]}%",
            f"WordAcc {best[1]}% / WER {best[2]}% on {ocr.get('source')} (n={ocr['n_samples']})",
        ])
    if ner["rows"] and ner["best_row"] is not None:
        best = ner["rows"][ner["best_row"]]
        sys_rows.append([f"Best NER F1 ({best[0]})", f"{best[3]}", f"P {best[1]} / R {best[2]}"])
    if rec["measured"]:
        sys_rows.append(["Generic substitution coverage", f"{rec.get('coverage', rec['accuracy']) * 100:.1f}%", f"n={rec['n_samples']} brands"])
        sys_rows.append(["Mean generic cost saving", f"{rec['mean_cost_saving_pct']}%", f"median {rec.get('median_cost_saving_pct', '?')}% / max {rec['max_cost_saving_pct']}%"])
    if inter["measured"]:
        sys_rows.append(["Interaction detection F1", f"{inter['f1']:.3f}", f"recall {inter['recall'] * 100:.1f}% / precision {inter['precision'] * 100:.1f}% (n={inter['n_samples']})"])
    if r2g and r2g.get("measured") and r2g.get("rows"):
        best = max(r2g["rows"], key=lambda r: r["generic_accuracy"])
        sys_rows.append([
            f"End-to-end generic accuracy ({best['engine']})",
            f"{best['generic_accuracy']}%",
            f"vs {best['word_accuracy']}% raw OCR word acc; BD ground truth, n={r2g['n_samples']}",
        ])
    if latency:
        sys_rows.append(["Mean end-to-end latency", f"{latency['mean_end_to_end_s']} s", f"{settings.resolved_device}, n={latency['n_samples']}"])

    system = {
        "title": "End-to-End System Performance (measured)",
        "columns": ["Metric", "Value", "Notes"],
        "rows": sys_rows,
        "best_row": None,
        "measured": bool(sys_rows),
    }

    tables_measured = [ocr["measured"], ner["measured"], system["measured"]]
    status = (
        "measured" if all(tables_measured)
        else "partial" if any(tables_measured)
        else "not_measured"
    )

    ocr_srcs = ", ".join(
        f"'{t.get('source')}' (n={t['n_samples']})" for t in ocr_tables
    )
    is_real = any("demo" not in str(t.get("source", "demo")) for t in ocr_tables)
    ocr_note = (
        f"OCR measured on real handwriting: {ocr_srcs}."
        if is_real
        else "OCR measured on the small printed demo set."
    )
    rec_note = (
        f"Recommendation measured on {rec.get('source', 'demo set')}: no brand->generic "
        "ground truth exists in the catalog, so coverage + real cost-saving + "
        "equivalence precision are reported (not a labelled accuracy)."
        if rec.get("mode") == "catalog"
        else "Recommendation measured on the curated demo gold set."
    )
    inter_note = (
        f"Interaction measured on {inter.get('source', 'demo set')}: positives/negatives "
        "drawn from the real DrugBank DDI list mapped to real brands, so P/R/F1 "
        "reflect brand->ingredient extraction fidelity against a known KB "
        "(near-ceiling by construction)."
        if inter.get("mode") == "real_ddi"
        else "Interaction measured on the curated demo gold pairs."
    )
    payload = {
        "status": status,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "device": settings.resolved_device,
        "disclaimer": (
            f"{ocr_note} {rec_note} {inter_note} NER is measured on a small "
            "curated gold set. See EVALUATION.md for methodology and caveats."
        ),
        "ocr": ocr,
        "ocr_datasets": ocr_tables,
        "detection": _detection_table(),
        "ner": ner,
        "recommendation": rec,
        "interactions": inter,
        "system": system,
    }
    if r2g and r2g.get("measured"):
        payload["recognition_to_generic"] = r2g
    path = save_benchmarks(payload)
    print(f"\nWrote measured benchmarks -> {path}  (status={status})")


if __name__ == "__main__":
    main()
