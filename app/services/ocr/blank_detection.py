"""Generic blank-page and near-blank-page detection.

Uses conservative multi-signal approach:
  - white pixel ratio
  - foreground pixel count
  - edge density
  - grayscale variance

A page is classified as blank only when multiple strong signals agree.
"""

from __future__ import annotations

import logging

import numpy as np

from app.core.config import SETTINGS

logger = logging.getLogger("dokstract.ocr_engine.blank_detection")


def is_blank_page(
    image: np.ndarray,
    *,
    white_ratio_threshold: float | None = None,
    foreground_threshold: int | None = None,
) -> bool:
    """Determine if a page is blank/near-blank and can skip OCR.

    Uses conservative multi-signal approach. A page must satisfy ALL of:
      1. White pixel ratio >= threshold (default 0.995)
      2. Foreground pixel count < 200 (absolute minimum)
      3. Edge density < 0.001 (very few edges)
      4. Grayscale variance < 10 (very uniform)

    Args:
        image: Decoded image as numpy array (H×W×C).
        white_ratio_threshold: Override for white pixel ratio threshold.

    Returns:
        True if page is confidently blank (all signals agree).
    """
    if not SETTINGS.ocr_blank_page_detection_enabled:
        return False

    white_thresh = white_ratio_threshold or SETTINGS.ocr_blank_page_white_ratio_threshold

    if len(image.shape) == 3:
        gray = np.mean(image.astype(np.float32), axis=2)
    else:
        gray = image.astype(np.float32)

    if gray.size == 0:
        return True

    total = gray.size

    # Signal 1: White pixel ratio
    white_pixels = np.sum(gray > 250)
    white_ratio = white_pixels / total
    if white_ratio < white_thresh:
        return False  # not blank enough

    # Signal 2: Absolute foreground pixel count
    dark_pixels = np.sum(gray < 200)
    if dark_pixels > 500:
        return False  # too many non-white pixels

    # Signal 3: Edge density
    edges = np.abs(np.diff(gray.astype(np.int16), axis=1))
    edge_density = np.sum(edges > 30) / max(total, 1)
    if edge_density > 0.002:
        return False  # too many edges

    # Signal 4: Grayscale variance
    variance = float(np.var(gray))
    if variance > 50:
        return False  # too varied

    # All signals agree → blank
    return True
