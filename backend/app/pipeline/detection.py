"""Stage 2 — Text-region detection with YOLOv8 (synopsis 7.2).

If a fine-tuned YOLOv8 text-region model is present at ``settings.yolo_weights``
we use it to localize text regions and crop them. Otherwise we fall back to a
classical contour/MSER-based line segmentation so the pipeline still produces
sensible regions for OCR without any trained weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import settings

_yolo_model = None  # lazy singleton


@dataclass
class Region:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    crop: np.ndarray = field(repr=False, default=None)


def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    if not Path(settings.yolo_weights).exists():
        return None
    from ultralytics import YOLO

    _yolo_model = YOLO(settings.yolo_weights)
    return _yolo_model


def _yolo_regions(image_bgr: np.ndarray) -> list[Region]:
    model = _load_yolo()
    if model is None:
        return []
    res = model.predict(
        image_bgr, conf=settings.yolo_conf, device=settings.resolved_device, verbose=False
    )[0]
    regions: list[Region] = []
    for box in res.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        conf = float(box.conf[0])
        crop = image_bgr[max(0, y1):y2, max(0, x1):x2]
        if crop.size:
            regions.append(Region((x1, y1, x2, y2), conf, crop))
    # top-to-bottom reading order
    regions.sort(key=lambda r: r.bbox[1])
    return regions


def _fallback_regions(gray: np.ndarray, image_bgr: np.ndarray) -> list[Region]:
    """Morphology-based text line detection used when no YOLO weights exist."""
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # dilate horizontally to merge characters into line blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dil = cv2.dilate(thr, kernel, iterations=1)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    regions: list[Region] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < 0.03 * w or ch < 0.012 * h:  # drop specks
            continue
        if cw * ch < 400:
            continue
        pad = 3
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + cw + pad), min(h, y + ch + pad)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size:
            regions.append(Region((x1, y1, x2, y2), 0.5, crop))
    regions.sort(key=lambda r: r.bbox[1])

    # If segmentation found nothing useful, fall back to the whole image.
    if not regions:
        regions = [Region((0, 0, w, h), 0.3, image_bgr)]
    return regions


def detect_regions(image_bgr: np.ndarray, gray: np.ndarray) -> tuple[list[Region], str]:
    """Return (regions, method_used)."""
    if Path(settings.yolo_weights).exists():
        regions = _yolo_regions(image_bgr)
        if regions:
            return regions, "yolov8"
    return _fallback_regions(gray, image_bgr), "morphology-fallback"


def draw_regions(image_bgr: np.ndarray, regions: list[Region]) -> np.ndarray:
    out = image_bgr.copy()
    for i, r in enumerate(regions):
        x1, y1, x2, y2 = r.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(
            out, str(i + 1), (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1, cv2.LINE_AA,
        )
    return out
