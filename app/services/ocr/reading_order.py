"""Reading-order assignment for items, lines, and blocks.

All functions are deterministic and stateless — given the same input,
they always produce the same output.
"""

from __future__ import annotations

from .models import OCRItem, OCRLine, OCRBlock


def assign_reading_order_items(items: list[OCRItem]) -> None:
    """Assign sequential reading-order indices to a list of OCR items in-place."""
    for i, item in enumerate(items):
        item.reading_order = i


def assign_reading_order_lines(lines: list[OCRLine]) -> None:
    """Assign sequential reading-order indices to a list of OCR lines in-place."""
    for i, line in enumerate(lines):
        line.reading_order = i


def assign_reading_order_blocks(blocks: list[OCRBlock]) -> None:
    """Assign sequential reading-order indices to a list of OCR blocks in-place."""
    for i, block in enumerate(blocks):
        block.reading_order = i


def sort_items_top_to_bottom_left_to_right(items: list[OCRItem]) -> list[OCRItem]:
    """Sort OCR items top-to-bottom, then left-to-right.

    Primary sort: y1 (topmost first).
    Secondary sort: x1 (leftmost first) within similar y positions.
    """
    return sorted(items, key=lambda it: (round(it.bbox.y1), round(it.bbox.x1)))


def sort_lines_reading_order(lines: list[OCRLine]) -> list[OCRLine]:
    """Sort lines in reading order: top-to-bottom by bbox.y1."""
    return sorted(lines, key=lambda l: l.bbox.y1)
