"""Grad-CAM / attention visualization over the prescription image (synopsis 7.7).

Primary method: **attention rollout** over the TrOCR Vision-Transformer encoder.
We run the ViT encoder with ``output_attentions=True``, multiply (A + I) across
layers, and read the CLS->patch attention to obtain a per-patch saliency map.
This shows which image regions the OCR model attended to — the same goal as
Grad-CAM for CNNs, adapted to a transformer encoder.

Fallback (no TrOCR/torch): an edge-density saliency map highlighting the
text-bearing regions of the image, overlaid as a heatmap.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from ...config import settings
from ...utils import bgr_to_base64_png

logger = logging.getLogger("rxai.xai.gradcam")


def _overlay_heatmap(image_bgr: np.ndarray, heat: np.ndarray) -> np.ndarray:
    heat = heat.astype(np.float32)
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    heat_resized = cv2.resize(heat, (image_bgr.shape[1], image_bgr.shape[0]))
    heat_u8 = np.uint8(255 * heat_resized)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.55, colored, 0.45, 0)


def _vit_attention_map(image_bgr: np.ndarray) -> np.ndarray | None:
    """Attention-rollout saliency from the TrOCR ViT encoder."""
    try:
        import torch
        from PIL import Image

        from ..ocr import load_trocr

        processor, model = load_trocr()  # reuse the OCR-loaded model

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pixel_values = processor(images=Image.fromarray(rgb), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(settings.resolved_device)

        with torch.no_grad():
            enc = model.encoder(pixel_values, output_attentions=True)
        attns = enc.attentions  # tuple(layers) of (b, heads, seq, seq)
        if not attns:
            return None

        # attention rollout
        result = None
        for a in attns:
            a = a[0].mean(dim=0)  # avg heads -> (seq, seq)
            a = a + torch.eye(a.size(0), device=a.device)
            a = a / a.sum(dim=-1, keepdim=True)
            result = a if result is None else a @ result

        cls_to_patches = result[0, 1:]  # drop CLS self
        n = cls_to_patches.numel()
        grid = int(round(n ** 0.5))
        if grid * grid != n:
            return None
        heat = cls_to_patches.reshape(grid, grid).detach().cpu().numpy()
        return heat
    except Exception as exc:
        logger.warning("ViT attention map unavailable: %s", exc)
        return None


def _edge_saliency(gray: np.ndarray) -> np.ndarray:
    """Fallback: local edge/ink density highlights handwritten text regions."""
    edges = cv2.Canny(gray, 50, 150)
    density = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), sigmaX=9)
    return density


def gradcam_artifact(image_bgr: np.ndarray, gray: np.ndarray) -> dict | None:
    """Return an XAI artifact dict for the image-level explanation."""
    if not settings.enable_xai:
        return None

    heat = _vit_attention_map(image_bgr)
    method_note = (
        "Attention rollout over the TrOCR ViT encoder — highlights regions the "
        "OCR model attended to during recognition."
    )
    if heat is None:
        heat = _edge_saliency(gray)
        method_note = (
            "Fallback edge/ink-density saliency (TrOCR not available) — "
            "approximates text-bearing regions."
        )

    try:
        overlay = _overlay_heatmap(image_bgr, heat)
        return {
            "method": "grad-cam",
            "target_stage": "OCR / Vision Transformer",
            "title": "Where the model looked (image attention heatmap)",
            "image_base64": bgr_to_base64_png(overlay),
            "note": method_note,
        }
    except Exception as exc:
        logger.warning("Grad-CAM overlay failed: %s", exc)
        return None
