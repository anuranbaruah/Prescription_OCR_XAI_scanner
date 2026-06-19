"""Shared helpers: image <-> base64, timing."""
from __future__ import annotations

import base64
import time
from contextlib import contextmanager

import cv2
import numpy as np


def bgr_to_base64_png(image: np.ndarray) -> str:
    """Encode a BGR (or grayscale) OpenCV image to a base64 PNG data URI."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def matplotlib_fig_to_base64(fig) -> str:
    """Encode a matplotlib figure to a base64 PNG data URI."""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return "data:image/png;base64," + encoded


def imread_unicode(path) -> np.ndarray | None:
    """Read an image as BGR, tolerating non-ASCII paths.

    ``cv2.imread`` uses the ANSI file API on Windows and silently returns None
    for paths containing Unicode characters (e.g. a U+2019 apostrophe in a
    dataset folder name). Reading the bytes ourselves and decoding avoids that.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def bytes_to_bgr(data: bytes) -> np.ndarray:
    """Decode raw upload bytes into a BGR OpenCV image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported or corrupt file.")
    return img


class Timer:
    """Accumulates named stage timings (milliseconds)."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round((time.perf_counter() - start) * 1000, 1)
