"""Generic low-content page detection.

Classifies pages with sparse content (stamps, seals, page numbers)
without skipping OCR. Used to select lower render DPI or lighter processing.
"""

from __future__ import annotations

import logging

import numpy as np

from app.core.config import SETTINGS

logger = logging.getLogger("dokstract.ocr_engine.low_content")


def is_low_content_page(
    image: np.ndarray,
    *,
    foreground_ratio_threshold: float = 0.05,
    edge_density_threshold: float = 0.005,
) -> bool:
    """Detect pages with minimal content that still need OCR.

    Uses conservative signals:
      - Foreground pixel ratio < 5%
      - Edge density < 0.5%
      - Both must be true for classification

    Low-content pages still get OCR but at potentially lower DPI.
    """
    if len(image.shape) == 3:
        gray = np.mean(image.astype(np.float32), axis=2)
    else:
        gray = image.astype(np.float32)

    if gray.size == 0:
        return True

    total = gray.size

    # Foreground ratio
    foreground = np.sum(gray < 200)
    fg_ratio = foreground / total

    # Edge density
    edges = np.abs(np.diff(gray.astype(np.int16), axis=1))
    edge_density = np.sum(edges > 30) / max(total, 1)

    return fg_ratio < foreground_ratio_threshold and edge_density < edge_density_threshold
