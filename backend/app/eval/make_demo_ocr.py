"""Generate the bundled *demo* OCR evaluation set.

This renders a handful of prescription lines to PNGs so the OCR evaluator runs
end-to-end out of the box and produces *real* WER/CER numbers. These are
**printed** images, not handwriting — they exist to exercise and validate the
harness, NOT to stand in for journal results. For publishable handwritten-OCR
numbers, point ``RXAI_EVAL_OCR_DIR`` at the Kaggle handwritten prescription
test split (see ``EVALUATION.md``).

Run:  python -m app.eval.make_demo_ocr
"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .datasets import EVAL_DIR

LINES = [
    "Tab Crocin 500 1-0-1 after food",
    "Augmentin 625 Duo twice daily",
    "Pan 40 before breakfast",
    "Azithral 500 once daily x 3 days",
    "Telma 40 in the morning for BP",
    "Atorva 10 at bedtime",
    "Glycomet 500 with meals",
    "Ecosprin 75 once daily",
]


def _font(size: int = 28) -> ImageFont.FreeTypeFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    out_dir = EVAL_DIR / "ocr"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    font = _font()

    rows = []
    for i, line in enumerate(LINES):
        img = Image.new("RGB", (640, 80), "white")
        draw = ImageDraw.Draw(img)
        draw.text((12, 22), line, fill="black", font=font)
        rel = f"images/line_{i:02d}.png"
        img.save(out_dir / rel)
        rows.append({"image": rel, "text": line})

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} demo OCR images + manifest to {out_dir}")


if __name__ == "__main__":
    main()
