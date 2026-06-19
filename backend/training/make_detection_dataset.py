r"""Synthesize a text-region DETECTION dataset for YOLOv8 from the word-crops.

The bundled handwriting datasets are single word-crops (one medicine name per
512x512 image) with NO page-level bounding boxes, so a real detector can't be
trained on them directly. This script composites those crops onto blank
"prescription page" canvases at known positions, which yields exact YOLO-format
bounding boxes for free (no manual annotation). YOLOv8 then learns to localize
handwritten medicine/text regions on a page.

Honesty note (for the paper): this is a SYNTHETIC layout dataset built from real
handwriting crops. mAP on its held-out split measures region localization on
synthetic pages; transfer to real photographed prescriptions is a stated
limitation, but it is a genuine, reproducible detector — far beyond the
morphology fallback.

    cd backend
    .venv\Scripts\python training\make_detection_dataset.py --train 400 --val 80

Output: backend/data/detection/{images,labels}/{train,val} + data.yaml
"""
from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
DATASET_ROOT = BACKEND.parent / "dataset"
OUT = BACKEND / "data" / "detection"


def _crop_sources() -> list[str]:
    """Word-crop image paths (Handwritten Rx train split; BD as backup)."""
    paths = glob.glob(str(DATASET_ROOT / "Handwritten Rx" / "Train_Set" / "*.jpg"))
    if not paths:
        for bd in DATASET_ROOT.glob("*Handwritten Prescription BD*"):
            paths += glob.glob(str(bd / "Training" / "training_words" / "*.png"))
    return paths


def _tight_word(path: str) -> np.ndarray | None:
    """Load a crop and return the tight ink region (whitespace removed)."""
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ys, xs = np.where(thr > 0)
    if len(xs) < 30:  # essentially blank
        return None
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = 4
    h, w = gray.shape
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    word = img[y1:y2, x1:x2]
    return word if word.size else None


def _page(words: list[np.ndarray], rng: random.Random) -> tuple[np.ndarray, list[tuple]]:
    """Compose words top-to-bottom onto a page; return (image, [(cls,xc,yc,w,h)])."""
    W, H = 768, 1024
    # slight off-white page tint for realism
    tint = rng.randint(244, 255)
    page = np.full((H, W, 3), tint, np.uint8)
    boxes: list[tuple] = []
    y = rng.randint(24, 70)
    while y < H - 80 and words:
        target_h = rng.randint(34, 58)
        x = rng.randint(28, 130)
        # 1-2 words on this line
        for _ in range(rng.choice([1, 1, 2])):
            if not words:
                break
            w0 = words.pop()
            scale = target_h / w0.shape[0]
            nw = max(8, int(w0.shape[1] * scale))
            nh = target_h
            if x + nw > W - 24:
                break
            resized = cv2.resize(w0, (nw, nh), interpolation=cv2.INTER_AREA)
            page[y:y + nh, x:x + nw] = resized
            xc = (x + nw / 2) / W
            yc = (y + nh / 2) / H
            boxes.append((0, xc, yc, nw / W, nh / H))
            x += nw + rng.randint(18, 60)
        y += target_h + rng.randint(14, 46)
    return page, boxes


def _write_split(name: str, n: int, sources: list[str], rng: random.Random) -> int:
    img_dir = OUT / "images" / name
    lbl_dir = OUT / "labels" / name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for i in range(n):
        k = rng.randint(5, 11)
        picks = [rng.choice(sources) for _ in range(k * 2)]
        words = [w for w in (_tight_word(p) for p in picks) if w is not None][:k]
        if len(words) < 3:
            continue
        page, boxes = _page(words, rng)
        if not boxes:
            continue
        stem = f"{name}_{i:04d}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), page)
        with (lbl_dir / f"{stem}.txt").open("w") as fh:
            for c, xc, yc, w, h in boxes:
                fh.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        made += 1
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--val", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sources = _crop_sources()
    if not sources:
        raise SystemExit("No word-crop sources found under dataset/.")
    print(f"Word-crop sources: {len(sources)}")
    rng = random.Random(args.seed)
    nt = _write_split("train", args.train, sources, rng)
    nv = _write_split("val", args.val, sources, rng)

    yaml = OUT / "data.yaml"
    yaml.write_text(
        f"path: {OUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: text_region\n",
        encoding="utf-8",
    )
    print(f"Wrote {nt} train + {nv} val synthetic pages -> {OUT}")
    print(f"data.yaml -> {yaml}")


if __name__ == "__main__":
    main()
