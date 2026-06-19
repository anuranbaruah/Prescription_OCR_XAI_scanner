r"""Fine-tune YOLOv8n for handwritten text-region detection.

Trains on the synthetic page dataset built by ``make_detection_dataset.py`` and
installs the best checkpoint at ``backend/models/yolov8n_rx.pt`` — the path
``detection.py`` looks for, so the deployed pipeline switches from the
morphology fallback to YOLOv8 automatically once this finishes.

    cd backend
    .venv\Scripts\python training\make_detection_dataset.py --train 400 --val 80
    .venv\Scripts\python training\train_yolo_local.py --epochs 60

Reports val mAP@0.5 and mAP@0.5:0.95 (ultralytics standard detection metrics).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
DATA_YAML = BACKEND / "data" / "detection" / "data.yaml"
RUNS_DIR = BACKEND / "results" / "yolo"
WEIGHTS_OUT = BACKEND / "models" / "yolov8n_rx.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", default="yolov8n.pt", help="base checkpoint (auto-downloaded)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} not found — run make_detection_dataset.py first.")

    import torch
    from ultralytics import YOLO

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"Training {args.base} on {DATA_YAML} (device={device}, imgsz={args.imgsz})")

    model = YOLO(args.base)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(RUNS_DIR),
        name="rx_text_region",
        exist_ok=True,
        patience=15,
        verbose=True,
    )
    # Install the trained weights FIRST so a flaky final-val step can't leave the
    # pipeline without a detector.
    best = RUNS_DIR / "rx_text_region" / "weights" / "best.pt"
    if best.exists():
        WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, WEIGHTS_OUT)
        print(f"Installed detector -> {WEIGHTS_OUT}  (pipeline will now use YOLOv8)")
    else:
        print(f"WARNING: best.pt not found at {best}")

    # workers=0 avoids spawning DataLoader subprocesses, which can hit a Windows
    # paging-file error (WinError 1455) when other GPU processes are running.
    metrics = model.val(
        data=str(DATA_YAML), imgsz=args.imgsz, device=device, workers=0, verbose=False
    )
    box = metrics.box
    print(f"\nVAL  mAP@0.5={box.map50:.4f}  mAP@0.5:0.95={box.map:.4f}  "
          f"precision={box.mp:.4f}  recall={box.mr:.4f}")


if __name__ == "__main__":
    main()
