"""Generic block reconstruction from OCR lines.

Groups nearby lines into blocks using inter-line gap analysis.
Classifies blocks as 'text', 'table_candidate', or 'unknown'.
No document-type-specific rules.
"""

from __future__ import annotations

import logging
from statistics import median

from .geometry import bbox_union
from .models import OCRLine, OCRBlock

logger = logging.getLogger("dokstract.ocr_engine.block_reconstruction")

# Maximum gap ratio (relative to median inter-line spacing) for merging lines
# into the same block.  Larger values merge more aggressively.
_DEFAULT_GAP_RATIO = 2.5

# Minimum horizontal overlap ratio for lines to be considered part of the same block.
_DEFAULT_H_OVERLAP_RATIO = 0.15


def reconstruct_blocks(
    lines: list[OCRLine],
    page_width: float,
    page_height: float,
    gap_ratio: float = _DEFAULT_GAP_RATIO,
    h_overlap_ratio: float = _DEFAULT_H_OVERLAP_RATIO,
) -> list[OCRBlock]:
    """Group nearby OCR lines into generic blocks.

    Algorithm:
      1. Sort lines by reading_order (top-to-bottom).
      2. Calculate median inter-line gap.
      3. Walk through lines: if the gap to the previous line is within
         threshold AND there is sufficient horizontal overlap, add to
         current block; otherwise start a new block.
      4. Classify each block as 'text', 'table_candidate', or 'unknown'.
      5. Assign block_ids to lines.

    Args:
        lines: OCR lines (must have reading_order assigned).
        page_width: Page width (unused; reserved).
        page_height: Page height (unused; reserved).
        gap_ratio: Maximum inter-line gap as multiple of median gap.
        h_overlap_ratio: Minimum horizontal overlap ratio for block merging.

    Returns:
        List of OCRBlock objects with reading_order assigned.
    """
    if not lines:
        return []

    # Sort by reading order
    sorted_lines = sorted(lines, key=lambda l: l.reading_order)

    # Calculate median inter-line spacing
    gaps: list[float] = []
    for i in range(1, len(sorted_lines)):
        gap = sorted_lines[i].bbox.y1 - sorted_lines[i - 1].bbox.y2
        if gap >= 0:
            gaps.append(gap)

    median_gap = median(gaps) if gaps else 20.0
    if median_gap <= 0:
        median_gap = 20.0

    threshold = median_gap * max(0.5, gap_ratio)

    # When only 2 lines exist, the single gap becomes the median,
    # which would always merge.  Use a tighter cap in that case.
    if len(gaps) <= 1 and len(sorted_lines) >= 2:
        # Cap threshold at 3x the average line height for small line counts
        avg_height = sum(l.bbox.height for l in sorted_lines) / len(sorted_lines)
        threshold = min(threshold, avg_height * 3.0)

    # Group lines into raw blocks
    raw_blocks: list[list[OCRLine]] = []
    current: list[OCRLine] = []

    for line in sorted_lines:
        if not current:
            current.append(line)
            continue

        prev = current[-1]
        gap = line.bbox.y1 - prev.bbox.y2

        # Horizontal overlap ratio
        x_overlap = max(0.0, min(prev.bbox.x2, line.bbox.x2) - max(prev.bbox.x1, line.bbox.x1))
        prev_width = prev.bbox.width
        overlap_ratio = x_overlap / prev_width if prev_width > 0 else 0.0

        if gap <= threshold and overlap_ratio >= h_overlap_ratio:
            current.append(line)
        else:
            raw_blocks.append(current)
            current = [line]

    if current:
        raw_blocks.append(current)

    # Build OCRBlock objects
    blocks: list[OCRBlock] = []
    for block_idx, block_lines in enumerate(raw_blocks):
        page_number = block_lines[0].page_number

        # Union bbox
        block_bbox = block_lines[0].bbox
        for l in block_lines[1:]:
            block_bbox = bbox_union(block_bbox, l.bbox)

        block_type = _classify_block(block_lines)

        block = OCRBlock(
            block_id=f"p{page_number}_b{block_idx}",
            page_number=page_number,
            block_type=block_type,
            bbox=block_bbox,
            line_ids=[l.line_id for l in block_lines],
            reading_order=block_idx,
        )
        blocks.append(block)

        # Back-link block_id to lines
        for l in block_lines:
            l.block_id = block.block_id

    logger.debug(
        "Reconstructed %d blocks from %d lines (median_gap=%.1f, threshold=%.1f)",
        len(blocks),
        len(lines),
        median_gap,
        threshold,
    )

    return blocks


def _classify_block(lines: list[OCRLine]) -> str:
    """Classify a block as 'text', 'table_candidate', or 'unknown'.

    Table candidates are identified by:
      - Multiple lines (>= 3)
      - Lines have similar x-ranges (low variance in x1 and x2)
      - Relatively few words per line (suggests table cells)
      - Lines are close together vertically
    """
    if len(lines) < 2:
        return "text"

    n = len(lines)

    # Calculate variance in x1 and x2 positions
    x1s = [l.bbox.x1 for l in lines]
    x2s = [l.bbox.x2 for l in lines]

    if n > 1:
        x1_mean = sum(x1s) / n
        x2_mean = sum(x2s) / n
        x1_var = sum((x - x1_mean) ** 2 for x in x1s) / n
        x2_var = sum((x - x2_mean) ** 2 for x in x2s) / n
    else:
        x1_var = 0.0
        x2_var = 0.0

    # Average word count per line
    avg_words = sum(len(l.text.split()) for l in lines) / n if n > 0 else 0.0

    # Average line height
    avg_height = sum(l.bbox.height for l in lines) / n if n > 0 else 0.0

    # Table candidate criteria:
    # - At least 3 lines
    # - Low x1 and x2 variance (aligned columns)
    # - Short lines (few words, typical of table cells)
    # - Lines not too tall (table rows are usually single-line)
    is_table = (
        n >= 3
        and x1_var < 8000.0
        and x2_var < 8000.0
        and avg_words < 6.0
        and avg_height < 100.0
    )

    if is_table:
        return "table_candidate"

    return "text"
