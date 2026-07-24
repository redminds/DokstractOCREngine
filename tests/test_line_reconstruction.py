"""Tests for line reconstruction module."""

from __future__ import annotations

import pytest

from app.services.ocr.models import BBox, OCRItem
from app.services.ocr.line_reconstruction import reconstruct_lines


def _item(item_id: str, page: int, x1: float, y1: float, x2: float, y2: float, text: str = "x", confidence: float = 0.9) -> OCRItem:
    return OCRItem(
        item_id=item_id,
        page_number=page,
        text=text,
        confidence=confidence,
        polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        bbox=BBox(x1, y1, x2, y2),
        normalized_bbox=BBox(0, 0, 1, 1),
    )


class TestReconstructLines:
    def test_single_item(self):
        items = [_item("p1_i0", 1, 10, 10, 50, 20, "Hello")]
        lines = reconstruct_lines(items, page_width=100, page_height=100)

        assert len(lines) == 1
        assert lines[0].text == "Hello"
        assert lines[0].item_ids == ["p1_i0"]
        assert items[0].line_id == "p1_l0"

    def test_words_on_same_line(self):
        """Two items at the same y position should be grouped."""
        items = [
            _item("p1_i0", 1, 10, 10, 50, 30, "Hello"),
            _item("p1_i1", 1, 60, 10, 110, 30, "World"),
        ]
        lines = reconstruct_lines(items, page_width=200, page_height=100)

        assert len(lines) == 1
        assert lines[0].text == "Hello World"
        assert lines[0].item_ids == ["p1_i0", "p1_i1"]

    def test_two_separate_lines(self):
        """Items on different y positions should be in separate lines."""
        items = [
            _item("p1_i0", 1, 10, 10, 50, 30, "Line1"),
            _item("p1_i1", 1, 10, 80, 50, 100, "Line2"),
        ]
        lines = reconstruct_lines(items, page_width=200, page_height=200)

        assert len(lines) == 2
        assert lines[0].text == "Line1"
        assert lines[1].text == "Line2"

    def test_left_to_right_sorting_within_line(self):
        """Items within a line should be sorted left-to-right regardless of input order."""
        items = [
            _item("p1_i1", 1, 100, 10, 150, 30, "World"),
            _item("p1_i0", 1, 10, 10, 50, 30, "Hello"),
        ]
        lines = reconstruct_lines(items, page_width=200, page_height=100)

        assert len(lines) == 1
        assert lines[0].text == "Hello World"

    def test_top_to_bottom_line_sorting(self):
        """Lines should be sorted top-to-bottom regardless of input order."""
        items = [
            _item("p1_i1", 1, 10, 80, 50, 100, "Second"),
            _item("p1_i0", 1, 10, 10, 50, 30, "First"),
        ]
        lines = reconstruct_lines(items, page_width=200, page_height=200)

        assert len(lines) == 2
        assert lines[0].text == "First"
        assert lines[1].text == "Second"

    def test_slightly_misaligned_text(self):
        """Text with minor vertical offset should still be grouped."""
        items = [
            _item("p1_i0", 1, 10, 10, 50, 28, "Hello"),
            _item("p1_i1", 1, 60, 12, 110, 30, "World"),
        ]
        lines = reconstruct_lines(items, page_width=200, page_height=100, overlap_threshold=0.3)

        assert len(lines) == 1
        assert lines[0].text == "Hello World"

    def test_overlap_threshold_prevents_merge(self):
        """With high threshold, misaligned text stays separate."""
        items = [
            _item("p1_i0", 1, 10, 10, 50, 28, "Hello"),
            _item("p1_i1", 1, 60, 22, 110, 40, "World"),  # only slight overlap
        ]
        lines = reconstruct_lines(items, page_width=200, page_height=100, overlap_threshold=0.8)

        assert len(lines) == 2

    def test_empty_items(self):
        lines = reconstruct_lines([], page_width=100, page_height=100)
        assert lines == []

    def test_invalid_overlap_threshold(self):
        """Invalid thresholds should fall back to default."""
        items = [
            _item("p1_i0", 1, 10, 10, 50, 30, "A"),
            _item("p1_i1", 1, 60, 10, 110, 30, "B"),
        ]
        lines = reconstruct_lines(items, page_width=200, page_height=100, overlap_threshold=1.5)
        assert len(lines) == 1  # Falls back to 0.4 default → should merge

    def test_confidence_averaging(self):
        items = [
            _item("p1_i0", 1, 10, 10, 50, 30, "A", confidence=0.8),
            _item("p1_i1", 1, 60, 10, 110, 30, "B", confidence=0.6),
        ]
        lines = reconstruct_lines(items, page_width=200, page_height=100)
        assert len(lines) == 1
        assert lines[0].confidence == pytest.approx(0.7)

    def test_line_bbox_union(self):
        items = [
            _item("p1_i0", 1, 10, 10, 50, 30, "A"),
            _item("p1_i1", 1, 60, 10, 110, 30, "B"),
        ]
        lines = reconstruct_lines(items, page_width=200, page_height=100)
        assert len(lines) == 1
        assert lines[0].bbox.x1 == 10
        assert lines[0].bbox.y1 == 10
        assert lines[0].bbox.x2 == 110
        assert lines[0].bbox.y2 == 30

    def test_line_reading_order_sequential(self):
        items = [
            _item("p1_i0", 1, 10, 10, 50, 30, "A"),
            _item("p1_i1", 1, 10, 90, 50, 110, "B"),
            _item("p1_i2", 1, 10, 170, 50, 190, "C"),
        ]
        lines = reconstruct_lines(items, page_width=200, page_height=300)
        assert len(lines) == 3
        for i, line in enumerate(lines):
            assert line.reading_order == i

    def test_item_line_id_back_link(self):
        items = [
            _item("p1_i0", 1, 10, 10, 50, 30, "Hello"),
            _item("p1_i1", 1, 60, 10, 110, 30, "World"),
        ]
        lines = reconstruct_lines(items, page_width=200, page_height=100)

        assert items[0].line_id == "p1_l0"
        assert items[1].line_id == "p1_l0"

    def test_multi_column(self):
        """Items in different columns but same y should still group by vertical overlap."""
        # Left column
        left = [
            _item("p1_i0", 1, 10, 10, 100, 30, "Left1"),
            _item("p1_i1", 1, 10, 50, 100, 70, "Left2"),
        ]
        # Right column
        right = [
            _item("p1_i2", 1, 300, 10, 400, 30, "Right1"),
            _item("p1_i3", 1, 300, 50, 400, 70, "Right2"),
        ]
        items = left + right
        lines = reconstruct_lines(items, page_width=500, page_height=200)

        # Should create 2 lines, each containing both columns
        assert len(lines) == 2
        # First line: Left1 + Right1
        assert "Left1" in lines[0].text
        assert "Right1" in lines[0].text
        # Second line: Left2 + Right2
        assert "Left2" in lines[1].text
        assert "Right2" in lines[1].text
