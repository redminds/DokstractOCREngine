"""Convert raw PaddleOCR output into stable internal OCRItem models.

This module isolates PaddleOCR-version-specific parsing so that the rest
of the pipeline operates on engine-agnostic internal models.
"""

from __future__ import annotations

import logging
from typing import Any

from .geometry import polygon_to_bbox, normalize_bbox
from .models import OCRItem

logger = logging.getLogger("dokstract.ocr_engine.paddle_adapter")


def adapt_paddle_result(
    raw_result: list[list[Any]] | None,
    page_number: int,
    page_width: float,
    page_height: float,
) -> list[OCRItem]:
    """Convert raw PaddleOCR ocr() output into internal OCRItem list.

    PaddleOCR 2.7.x returns:
        [
            [  # page-level list of detections
                [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],  [text, confidence] ],
                ...
            ],
            ...
        ]
    or None if no text detected.

    Args:
        raw_result: Raw output from PaddleOCR.ocr(image, cls=False).
        page_number: 1-based page number for ID generation.
        page_width: Rendered image width in pixels.
        page_height: Rendered image height in pixels.

    Returns:
        List of validated OCRItem objects. May be empty if no text detected.
    """
    items: list[OCRItem] = []

    if not raw_result:
        logger.debug("PaddleOCR returned None/empty for page %d", page_number)
        return items

    item_index = 0

    for line_group in raw_result:
        if not line_group:
            continue
        for detection in line_group:
            try:
                item = _parse_detection(detection, page_number, item_index, page_width, page_height)
                if item is not None:
                    items.append(item)
                    item_index += 1
            except Exception:
                logger.debug(
                    "Skipping malformed PaddleOCR detection on page %d at index %d",
                    page_number,
                    item_index,
                    exc_info=True,
                )
                continue

    logger.debug("Adapted %d OCR items from PaddleOCR for page %d", len(items), page_number)
    return items


def _parse_detection(
    detection: list[Any],
    page_number: int,
    item_index: int,
    page_width: float,
    page_height: float,
) -> OCRItem | None:
    """Parse a single PaddleOCR detection into an OCRItem or None if invalid.

    Expected detection format:
        [
            [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # polygon
            [text, confidence]                        # recognition
        ]
    """
    if not detection or len(detection) < 2:
        return None

    polygon_data = detection[0]
    rec_data = detection[1]

    # Validate polygon: must have at least 3 points
    if not polygon_data or len(polygon_data) < 3:
        return None

    # Validate recognition data
    if not rec_data or len(rec_data) < 2:
        return None

    # Extract polygon coordinates
    polygon: list[list[float]] = []
    for pt in polygon_data:
        if len(pt) < 2:
            return None
        polygon.append([float(pt[0]), float(pt[1])])

    # Extract text and confidence
    text = str(rec_data[0]).strip() if rec_data[0] else ""
    confidence = float(rec_data[1])

    # Clamp confidence to [0, 1]
    if confidence < 0.0:
        confidence = 0.0
    elif confidence > 1.0:
        # If PaddleOCR returns 0-100 scale, normalize
        if confidence > 1.0:
            confidence = confidence / 100.0 if confidence <= 100.0 else 1.0

    # Calculate bounding boxes
    bbox = polygon_to_bbox(polygon)

    # Validate bbox is reasonable
    if not bbox.is_valid:
        return None

    # Clamp bbox to page boundaries (with small tolerance)
    bbox = _clamp_bbox_to_page(bbox, page_width, page_height)

    normalized_bbox = normalize_bbox(bbox, page_width, page_height)

    return OCRItem(
        item_id=f"p{page_number}_i{item_index}",
        page_number=page_number,
        text=text,
        confidence=confidence,
        polygon=polygon,
        bbox=bbox,
        normalized_bbox=normalized_bbox,
    )


def _clamp_bbox_to_page(bbox: "BBox", page_width: float, page_height: float) -> "BBox":
    """Clamp bounding box to page boundaries, allowing 1px tolerance."""
    from .models import BBox as _BBox

    tolerance = 1.0
    return _BBox(
        x1=max(0.0, bbox.x1),
        y1=max(0.0, bbox.y1),
        x2=min(page_width + tolerance, max(0.0, bbox.x2)),
        y2=min(page_height + tolerance, max(0.0, bbox.y2)),
    )
