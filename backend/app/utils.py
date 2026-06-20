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
    """Decode raw upload bytes into a BGR OpenCV image.

    Accepts both raster images and PDFs — a PDF is rendered (and its pages
    vertically stacked) into a single image so patients can upload either.
    """
    if is_pdf(data):
        return pdf_bytes_to_bgr(data)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported or corrupt file.")
    return img


def is_pdf(data: bytes) -> bool:
    """True if the bytes look like a PDF (``%PDF`` magic, allowing a small BOM)."""
    return b"%PDF-" in data[:1024]


def pdf_bytes_to_bgr(data: bytes, max_pages: int = 5, dpi: int = 200) -> np.ndarray:
    """Render a PDF prescription into one BGR image.

    Each page is rasterised at ``dpi`` and pages are stacked vertically (capped
    at ``max_pages``) so multi-page prescriptions are processed as a single
    image without losing later pages. Requires PyMuPDF (``pymupdf``).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ValueError(
            "PDF support needs the 'pymupdf' package. Install it or upload an image."
        ) from exc

    zoom = dpi / 72.0  # PDF user-space is 72 dpi
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[np.ndarray] = []
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            if doc.page_count == 0:
                raise ValueError("The PDF has no pages.")
            for page in doc[: max(1, max_pages)]:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                # PyMuPDF gives RGB(A); convert to BGR for OpenCV.
                code = cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR
                pages.append(cv2.cvtColor(arr, code))
    except ValueError:
        raise
    except Exception as exc:  # corrupt/encrypted PDF, etc.
        raise ValueError(f"Could not read the PDF: {exc}") from exc

    if len(pages) == 1:
        return pages[0]

    # Stack pages vertically, padding narrower pages to the widest width and
    # separating them with a thin white gutter.
    width = max(p.shape[1] for p in pages)
    gutter = np.full((16, width, 3), 255, np.uint8)
    canvas: list[np.ndarray] = []
    for i, p in enumerate(pages):
        if p.shape[1] != width:
            pad = np.full((p.shape[0], width - p.shape[1], 3), 255, np.uint8)
            p = np.hstack([p, pad])
        if i:
            canvas.append(gutter)
        canvas.append(p)
    return np.vstack(canvas)


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
