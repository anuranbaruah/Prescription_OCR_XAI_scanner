"""Stage 1 — Image preprocessing (synopsis 7.1).

Pipeline: grayscale -> CLAHE contrast normalization -> Gaussian denoise ->
deskew -> Otsu binarization. We keep two outputs:
  * ``gray``   : enhanced grayscale (best input for transformer OCR like TrOCR,
                 which prefers natural-looking images over hard binarization)
  * ``binary`` : Otsu-thresholded image (useful for classical OCR / detection)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessOutput:
    original: np.ndarray   # BGR
    gray: np.ndarray       # enhanced grayscale
    binary: np.ndarray     # Otsu binary
    deskew_angle: float


def _deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate skew from foreground pixels and rotate to correct it."""
    inv = cv2.bitwise_not(gray)
    thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.shape[0] < 50:
        return gray, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    # Only correct meaningful skew to avoid degrading near-straight scans.
    if abs(angle) < 0.5 or abs(angle) > 30:
        return gray, 0.0
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        gray, m, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, float(angle)


def preprocess(image_bgr: np.ndarray) -> PreprocessOutput:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # (ii) CLAHE — adaptive histogram equalization for uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # (iii) Gaussian blur 3x3 — light denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # geometric correction
    gray, angle = _deskew(gray)

    # (iv) Otsu binarization
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return PreprocessOutput(
        original=image_bgr, gray=gray, binary=binary, deskew_angle=angle
    )
