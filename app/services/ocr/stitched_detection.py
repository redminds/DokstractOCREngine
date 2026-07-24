"""Generic stitched-page detection using image geometry.

Detects single images that appear to contain multiple document pages
combined into one scan. Uses conservative, explainable heuristics:
  - extreme aspect ratio
  - repeated horizontal whitespace bands
  - multiple page-sized content regions

No document-type-specific rules. No LLM. No OCR word counting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.core.config import SETTINGS

logger = logging.getLogger("dokstract.ocr_engine.stitched_page")


@dataclass
class StitchedPageResult:
    """Result of stitched-page detection."""

    suspected_stitched_page: bool
    confidence: float  # 0.0 – 1.0
    reasons: list[str] = field(default_factory=list)
    estimated_visual_pages: int = 1


def detect_stitched_page(
    image: np.ndarray,
    *,
    aspect_signal_threshold: float | None = None,
    confidence_threshold: float | None = None,
) -> StitchedPageResult:
    """Detect whether a single image contains multiple stitched document pages.

    Args:
        image: Decoded image as numpy array (H×W×C).
        aspect_signal_threshold: Override soft aspect threshold.
        confidence_threshold: Override confidence threshold.

    Returns:
        StitchedPageResult with suspicion flag, confidence, and reasons.
    """
    if not SETTINGS.ocr_stitched_page_detection_enabled:
        return StitchedPageResult(suspected_stitched_page=False, confidence=0.0)

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return StitchedPageResult(suspected_stitched_page=False, confidence=0.0)

    soft_ar = aspect_signal_threshold or SETTINGS.ocr_stitched_page_aspect_signal_threshold
    conf_thresh = confidence_threshold or SETTINGS.ocr_stitched_page_confidence_threshold
    reasons: list[str] = []
    signals: list[float] = []

    # ── Signal 1: Soft aspect ratio signal ───────────────────────────────
    aspect_ratio = max(width / max(height, 1), height / max(width, 1))
    ar_signal = min(1.0, max(0.0, (aspect_ratio - 1.0) / (soft_ar * 2 - 1.0)))
    if aspect_ratio >= soft_ar:
        reasons.append("elevated_aspect_ratio")
        signals.append(min(1.0, ar_signal))
    else:
        signals.append(0.0)

    # ── Signal 2: Repeated horizontal whitespace bands ───────────────────
    if len(image.shape) == 3:
        gray = np.mean(image.astype(np.float32), axis=2)
    else:
        gray = image.astype(np.float32)
    row_means = np.mean(gray, axis=1)
    row_means = row_means / max(np.max(row_means), 1.0)

    white_threshold = 0.85
    white_rows = row_means > white_threshold
    band_heights: list[int] = []
    current_band = 0
    for is_white in white_rows:
        if is_white:
            current_band += 1
        else:
            if current_band > 0:
                band_heights.append(current_band)
            current_band = 0
    if current_band > 0:
        band_heights.append(current_band)

    min_band_height = max(10, int(height * 0.02))
    significant_bands = [h for h in band_heights if h >= min_band_height]
    ws_signal = 0.0
    if len(significant_bands) >= 2:
        reasons.append("repeated_horizontal_whitespace")
        ws_signal = min(1.0, len(significant_bands) / 10.0)
    signals.append(ws_signal)

    # ── Signal 3: Multiple similarly-sized content regions ───────────────
    # Also check vertical whitespace for horizontal stitching
    col_means = np.mean(gray, axis=0)
    col_means = col_means / max(np.max(col_means), 1.0)
    white_cols = col_means > white_threshold
    v_band_widths: list[int] = []
    current_v = 0
    for is_white_v in white_cols:
        if is_white_v:
            current_v += 1
        else:
            if current_v > 0:
                v_band_widths.append(current_v)
            current_v = 0
    if current_v > 0:
        v_band_widths.append(current_v)

    min_v_band = max(10, int(width * 0.02))
    sig_v_bands = [w for w in v_band_widths if w >= min_v_band]
    if len(sig_v_bands) >= 2:
        reasons.append("repeated_vertical_whitespace")
        ws_signal = max(ws_signal, min(1.0, len(sig_v_bands) / 8.0))
    signals[1] = ws_signal  # update whitespace signal with max of h/v

    # Region-based signal
    region_heights: list[int] = []
    in_content = False
    content_start = 0
    for i, is_white in enumerate(white_rows):
        if not is_white and not in_content:
            in_content = True
            content_start = i
        elif is_white and in_content:
            region_h = i - content_start
            if region_h >= min_band_height:
                region_heights.append(region_h)
            in_content = False
    if in_content:
        region_h = len(white_rows) - content_start
        if region_h >= min_band_height:
            region_heights.append(region_h)

    region_signal = 0.0
    if len(region_heights) >= 3:
        median_h = float(np.median(region_heights))
        if median_h > 0:
            similar_count = sum(
                1 for h in region_heights
                if 0.5 * median_h <= h <= 2.0 * median_h
            )
            if similar_count >= 3:
                reasons.append("multiple_page_sized_regions")
                region_signal = min(1.0, similar_count / 6.0)
    signals.append(region_signal)

    # ── Estimate visual pages ───────────────────────────────────────────
    estimated_pages = 1
    if region_signal > 0.3:
        median_h = float(np.median(region_heights)) if region_heights else height
        if median_h > 0:
            estimated_pages = max(1, min(20, int(height / median_h)))

    # ── Combine signals ─────────────────────────────────────────────────
    if not reasons:
        return StitchedPageResult(suspected_stitched_page=False, confidence=0.0)

    confidence = 0.4 * signals[0] + 0.35 * signals[1] + 0.25 * signals[2]
    confidence = min(1.0, max(0.0, confidence))
    suspected = confidence >= conf_thresh

    return StitchedPageResult(
        suspected_stitched_page=suspected,
        confidence=round(confidence, 4),
        reasons=reasons,
        estimated_visual_pages=estimated_pages,
    )
