"""Digital PDF fast path — extract text from embedded PDF text spans.

Converts PyMuPDF text spans into OCRItems following the same canonical
model as PaddleOCR output, enabling mixed digital/scanned page processing.
"""

from __future__ import annotations

import logging
from typing import Any

import fitz

from app.core.config import SETTINGS
from app.services.ocr.models import BBox, OCRItem, OCRLine, OCRBlock, OCRPage
from app.services.ocr.geometry import normalize_bbox, bbox_union
from app.services.ocr.line_reconstruction import reconstruct_lines
from app.services.ocr.block_reconstruction import reconstruct_blocks
from app.services.ocr.reading_order import (
    assign_reading_order_items,
    assign_reading_order_lines,
    assign_reading_order_blocks,
)

logger = logging.getLogger("dokstract.ocr_engine.digital_pdf")


def _has_usable_digital_text(page: fitz.Page) -> bool:
    """Quick check: does the page have enough embedded text to try extraction?"""
    if not SETTINGS.ocr_digital_pdf_fast_path_enabled:
        return False
    try:
        text = page.get_text("text")
        if not text or len(text.strip()) < SETTINGS.ocr_digital_pdf_min_characters:
            return False
        printable = sum(1 for c in text if c.isprintable() or c.isspace())
        if len(text) > 0 and printable / len(text) < 0.7:
            return False
        return True
    except Exception:
        return False


def extract_digital_page(
    page: fitz.Page,
    page_number: int,
    page_width: float,
    page_height: float,
) -> OCRPage | None:
    """Extract text from a digital PDF page as canonical OCRPage.

    Returns None if the embedded text is insufficient (caller should fall
    back to PaddleOCR).

    Args:
        page: PyMuPDF Page object.
        page_number: 1-based page number.
        page_width: Rendered page width in pixels.
        page_height: Rendered page height in pixels.

    Returns:
        OCRPage with items/lines/blocks, or None if text is insufficient.
    """
    if not SETTINGS.ocr_digital_pdf_fast_path_enabled:
        return None

    try:
        text_dict = page.get_text("dict")
    except Exception:
        return None

    blocks_data = text_dict.get("blocks", [])
    if not blocks_data:
        return None

    # Convert text spans to OCRItems
    items: list[OCRItem] = []
    item_index = 0
    total_chars = 0
    printable_chars = 0

    page_rect = page.rect
    scale_x = page_width / page_rect.width if page_rect.width > 0 else 1.0
    scale_y = page_height / page_rect.height if page_rect.height > 0 else 1.0

    for block in blocks_data:
        if block.get("type") != 0:  # text block only
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                total_chars += len(text)
                printable_chars += sum(1 for c in text if c.isprintable() or c.isspace())

                bbox_rect = span.get("bbox")
                if not bbox_rect or len(bbox_rect) < 4:
                    continue

                x1 = bbox_rect[0] * scale_x
                y1 = bbox_rect[1] * scale_y
                x2 = bbox_rect[2] * scale_x
                y2 = bbox_rect[3] * scale_y

                if x2 <= x1 or y2 <= y1:
                    continue

                bbox = BBox(x1, y1, x2, y2)
                norm_bbox = normalize_bbox(bbox, page_width, page_height)

                # Digital text confidence: 1.0 (extracted directly from PDF text layer)
                # This means "directly from embedded source," not a probabilistic OCR score.
                item = OCRItem(
                    item_id=f"p{page_number}_i{item_index}",
                    page_number=page_number,
                    text=text,
                    confidence=1.0,
                    polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    bbox=bbox,
                    normalized_bbox=norm_bbox,
                )
                items.append(item)
                item_index += 1

    # Quality checks
    if total_chars < SETTINGS.ocr_digital_pdf_min_characters:
        logger.debug("Page %d: digital text too short (%d chars)", page_number, total_chars)
        return None

    if total_chars > 0:
        printable_ratio = printable_chars / total_chars
    else:
        printable_ratio = 0.0
    if printable_ratio < 0.85:
        logger.debug("Page %d: low printable ratio %.2f", page_number, printable_ratio)
        return None

    if not items:
        return None

    # Reconstruct lines and blocks from digital items
    lines = reconstruct_lines(items, page_width, page_height)
    blocks = reconstruct_blocks(lines, page_width, page_height)

    for line in lines:
        line.normalized_bbox = normalize_bbox(line.bbox, page_width, page_height)
    for block in blocks:
        block.normalized_bbox = normalize_bbox(block.bbox, page_width, page_height)
        block_lines_list = [l for l in lines if l.line_id in block.line_ids]
        block_confidences = [l.confidence for l in block_lines_list if l.confidence > 0.0]
        block.confidence = sum(block_confidences) / len(block_confidences) if block_confidences else 0.0

    assign_reading_order_items(items)
    assign_reading_order_lines(lines)
    assign_reading_order_blocks(blocks)

    for line in lines:
        if line.block_id:
            for item in items:
                if item.line_id == line.line_id:
                    item.block_id = line.block_id

    page_text = "\n".join(line.text for line in lines if line.text.strip())
    page_confidence = sum(it.confidence for it in items) / len(items) if items else 0.0

    logger.info("Page %d: digital text extracted (%d items, %d lines)", page_number, len(items), len(lines))

    return OCRPage(
        page_number=page_number,
        width=page_width,
        height=page_height,
        items=items,
        lines=lines,
        blocks=blocks,
        text=page_text,
        confidence=page_confidence,
        page_metrics={"page_total_ms": 0.0, "source": "digital_text"},
    )
