"""Tests for block reconstruction module."""

from __future__ import annotations

import pytest

from app.services.ocr.models import BBox, OCRLine
from app.services.ocr.block_reconstruction import reconstruct_blocks


def _line(line_id: str, page: int, text: str, x1: float, y1: float, x2: float, y2: float, confidence: float = 0.9, reading_order: int = 0, item_ids: list[str] | None = None) -> OCRLine:
    return OCRLine(
        line_id=line_id,
        page_number=page,
        text=text,
        confidence=confidence,
        bbox=BBox(x1, y1, x2, y2),
        item_ids=item_ids or [],
        reading_order=reading_order,
    )


class TestReconstructBlocks:
    def test_single_line(self):
        lines = [_line("p1_l0", 1, "Hello", 10, 10, 100, 30, reading_order=0)]
        blocks = reconstruct_blocks(lines, page_width=200, page_height=100)

        assert len(blocks) == 1
        assert blocks[0].block_type == "text"

    def test_two_nearby_lines_merged(self):
        lines = [
            _line("p1_l0", 1, "Line 1", 10, 10, 200, 30, reading_order=0),
            _line("p1_l1", 1, "Line 2", 10, 35, 200, 55, reading_order=1),  # gap=5
        ]
        blocks = reconstruct_blocks(lines, page_width=300, page_height=200)

        assert len(blocks) == 1
        assert blocks[0].line_ids == ["p1_l0", "p1_l1"]

    def test_distant_lines_separate(self):
        lines = [
            _line("p1_l0", 1, "Line 1", 10, 10, 200, 30, reading_order=0),
            # Very large gap (500px) ensures separation
            _line("p1_l1", 1, "Line 2", 10, 500, 200, 520, reading_order=1),
        ]
        blocks = reconstruct_blocks(lines, page_width=300, page_height=600)

        assert len(blocks) == 2

    def test_block_bbox_union(self):
        lines = [
            _line("p1_l0", 1, "A", 10, 10, 100, 30, reading_order=0),
            _line("p1_l1", 1, "B", 20, 35, 90, 55, reading_order=1),
        ]
        blocks = reconstruct_blocks(lines, page_width=200, page_height=100)

        assert len(blocks) == 1
        assert blocks[0].bbox.x1 == 10
        assert blocks[0].bbox.y1 == 10
        assert blocks[0].bbox.x2 == 100
        assert blocks[0].bbox.y2 == 55

    def test_table_candidate_detection(self):
        """Multiple short, aligned lines → table_candidate."""
        lines = [
            _line("p1_l0", 1, "A  123", 100, 10, 200, 25, reading_order=0),
            _line("p1_l1", 1, "B  456", 100, 30, 200, 45, reading_order=1),
            _line("p1_l2", 1, "C  789", 100, 50, 200, 65, reading_order=2),
        ]
        blocks = reconstruct_blocks(lines, page_width=400, page_height=200)

        assert len(blocks) == 1
        assert blocks[0].block_type == "table_candidate"

    def test_text_block(self):
        """Longer lines → text block."""
        lines = [
            _line("p1_l0", 1, "This is a long paragraph line", 10, 10, 500, 30, reading_order=0),
            _line("p1_l1", 1, "Another paragraph line here", 10, 35, 500, 55, reading_order=1),
        ]
        blocks = reconstruct_blocks(lines, page_width=600, page_height=200)

        assert len(blocks) == 1
        assert blocks[0].block_type == "text"

    def test_empty_lines(self):
        blocks = reconstruct_blocks([], page_width=100, page_height=100)
        assert blocks == []

    def test_block_back_links(self):
        lines = [
            _line("p1_l0", 1, "A", 10, 10, 100, 30, reading_order=0),
            _line("p1_l1", 1, "B", 10, 35, 100, 55, reading_order=1),
        ]
        blocks = reconstruct_blocks(lines, page_width=200, page_height=100)

        assert len(blocks) == 1
        assert lines[0].block_id == "p1_b0"
        assert lines[1].block_id == "p1_b0"

    def test_reading_order_assignment(self):
        lines = [
            _line("p1_l0", 1, "Block 1", 10, 10, 100, 30, reading_order=0),
            # Large gap (300px) ensures separate blocks
            _line("p1_l1", 1, "Block 2", 10, 300, 100, 320, reading_order=1),
        ]
        blocks = reconstruct_blocks(lines, page_width=200, page_height=400)

        assert len(blocks) == 2
        assert blocks[0].reading_order == 0
        assert blocks[1].reading_order == 1

    def test_gap_ratio_parameter(self):
        """With very small gap_ratio, even small gaps create separate blocks."""
        lines = [
            _line("p1_l0", 1, "A", 10, 10, 100, 30, reading_order=0),
            _line("p1_l1", 1, "B", 10, 35, 100, 55, reading_order=1),
        ]
        # gap_ratio=0.1 with median_gap=5 → threshold=0.5, gap=5 > 0.5 → separate
        blocks = reconstruct_blocks(lines, page_width=200, page_height=100, gap_ratio=0.1)

        assert len(blocks) == 2

    def test_different_columns_not_merged(self):
        """Lines in different columns with low horizontal overlap → separate blocks."""
        lines = [
            _line("p1_l0", 1, "Left A", 10, 10, 100, 30, reading_order=0),
            _line("p1_l1", 1, "Right B", 300, 10, 400, 30, reading_order=1),  # no h-overlap
        ]
        blocks = reconstruct_blocks(lines, page_width=500, page_height=100, h_overlap_ratio=0.2)

        # No horizontal overlap → should be separate blocks
        assert len(blocks) == 2
