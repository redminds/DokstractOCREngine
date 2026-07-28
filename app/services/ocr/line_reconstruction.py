"""Generic geometry-based line reconstruction.

Groups OCR items into lines using vertical overlap, without any
document-type-specific rules.  Works with any document layout.
"""

from __future__ import annotations

import logging

from .geometry import bbox_union, median_item_height, vertical_overlap
from .models import OCRItem, OCRLine

logger = logging.getLogger("dokstract.ocr_engine.line_reconstruction")

# Fraction of an item's height that must vertically overlap with the
# representative item of an existing line for the item to be merged.
_DEFAULT_OVERLAP_THRESHOLD = 0.4

# When median item height cannot be determined, use this fallback (pixels).
_DEFAULT_MEDIAN_HEIGHT_FALLBACK = 20.0


def reconstruct_lines(
    items: list[OCRItem],
    page_width: float,
    page_height: float,
    overlap_threshold: float = _DEFAULT_OVERLAP_THRESHOLD,
) -> list[OCRLine]:
    """Group OCR items into lines based on vertical overlap.

    Algorithm:
      1. Sort items by y1 (top-to-bottom), then x1 (left-to-right).
      2. For each item, check if it vertically overlaps with an existing
         line's representative item above the threshold.
      3. If yes, add to that line.  If no, start a new line.
      4. Sort items within each line left-to-right.
      5. Sort lines top-to-bottom.
      6. Build OCRLine objects with combined text, avg confidence, and bbox.

    Args:
        items: OCR items from PaddleOCR adapter (unsorted).
        page_width: Page width (unused; reserved for future relative thresholds).
        page_height: Page height (unused; reserved for future relative thresholds).
        overlap_threshold: Minimum vertical overlap ratio to merge into an existing line.

    Returns:
        List of reconstructed OCRLine objects with reading_order assigned.
    """
    if not items:
        return []

    if not (0.0 < overlap_threshold <= 1.0):
        overlap_threshold = _DEFAULT_OVERLAP_THRESHOLD

    # Sort items by y (top to bottom), then x (left to right)
    sorted_items = sorted(items, key=lambda it: (round(it.bbox.y1, 1), round(it.bbox.x1, 1)))

    # Determine overlap threshold relative to median text height
    med_height = median_item_height(sorted_items)
    if med_height <= 0:
        med_height = _DEFAULT_MEDIAN_HEIGHT_FALLBACK

    # Group items into raw line groups
    line_groups: list[list[OCRItem]] = []

    for item in sorted_items:
        placed = False
        for group in line_groups:
            # Use the first item of the group as the representative
            rep = group[0]
            overlap = vertical_overlap(item.bbox, rep.bbox)

            if overlap >= overlap_threshold:
                group.append(item)
                placed = True
                break

        if not placed:
            line_groups.append([item])

    # Sort items within each line left-to-right
    for group in line_groups:
        group.sort(key=lambda it: it.bbox.x1)

    # Sort line groups top-to-bottom (by the topmost item in each group)
    line_groups.sort(key=lambda g: min(it.bbox.y1 for it in g))

    # Build OCRLine objects
    lines: list[OCRLine] = []
    for line_idx, group in enumerate(line_groups):
        page_number = group[0].page_number

        # Combined text: space-separated words
        combined_text = " ".join(it.text for it in group if it.text)

        # Average confidence (only from items with positive confidence)
        confidences = [it.confidence for it in group if it.confidence > 0.0]
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        # Union bounding box
        line_bbox = group[0].bbox
        for it in group[1:]:
            line_bbox = bbox_union(line_bbox, it.bbox)

        # Collect item IDs
        item_ids = [it.item_id for it in group]

        line = OCRLine(
            line_id=f"p{page_number}_l{line_idx}",
            page_number=page_number,
            text=combined_text,
            confidence=avg_confidence,
            bbox=line_bbox,
            item_ids=item_ids,
            reading_order=line_idx,
        )
        lines.append(line)

        # Back-link: assign line_id to each item
        for it in group:
            it.line_id = line.line_id

    logger.debug(
        "Reconstructed %d lines from %d items (overlap_threshold=%.2f, median_height=%.1f)",
        len(lines),
        len(items),
        overlap_threshold,
        med_height,
    )

    return lines
