"""Tests for reading order module."""

from __future__ import annotations

from app.services.ocr.models import BBox, OCRItem, OCRLine, OCRBlock
from app.services.ocr.reading_order import (
    assign_reading_order_items,
    assign_reading_order_lines,
    assign_reading_order_blocks,
    sort_items_top_to_bottom_left_to_right,
    sort_lines_reading_order,
)


def _item(x1: float, y1: float, x2: float, y2: float) -> OCRItem:
    return OCRItem(
        item_id="test",
        page_number=1,
        text="x",
        confidence=0.9,
        polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        bbox=BBox(x1, y1, x2, y2),
        normalized_bbox=BBox(0, 0, 1, 1),
    )


def _line(y1: float, text: str = "x") -> OCRLine:
    return OCRLine(
        line_id="test",
        page_number=1,
        text=text,
        confidence=0.9,
        bbox=BBox(0, y1, 100, y1 + 20),
    )


def _block(y1: float) -> OCRBlock:
    return OCRBlock(
        block_id="test",
        page_number=1,
        block_type="text",
        bbox=BBox(0, y1, 100, y1 + 50),
    )


class TestAssignReadingOrder:
    def test_items_sequential(self):
        items = [_item(0, 0, 10, 10), _item(0, 10, 10, 20), _item(0, 20, 10, 30)]
        assign_reading_order_items(items)
        assert [i.reading_order for i in items] == [0, 1, 2]

    def test_lines_sequential(self):
        lines = [_line(0), _line(20), _line(40)]
        assign_reading_order_lines(lines)
        assert [l.reading_order for l in lines] == [0, 1, 2]

    def test_blocks_sequential(self):
        blocks = [_block(0), _block(50), _block(100)]
        assign_reading_order_blocks(blocks)
        assert [b.reading_order for b in blocks] == [0, 1, 2]

    def test_empty_lists(self):
        assign_reading_order_items([])
        assign_reading_order_lines([])
        assign_reading_order_blocks([])
        # No exception = pass


class TestSortItems:
    def test_top_to_bottom(self):
        items = [
            _item(0, 30, 10, 40),
            _item(0, 0, 10, 10),
            _item(0, 15, 10, 25),
        ]
        sorted_items = sort_items_top_to_bottom_left_to_right(items)
        assert sorted_items[0].bbox.y1 == 0
        assert sorted_items[1].bbox.y1 == 15
        assert sorted_items[2].bbox.y1 == 30

    def test_same_y_left_to_right(self):
        items = [
            _item(50, 0, 60, 10),
            _item(10, 0, 20, 10),
        ]
        sorted_items = sort_items_top_to_bottom_left_to_right(items)
        assert sorted_items[0].bbox.x1 == 10
        assert sorted_items[1].bbox.x1 == 50

    def test_stable_deterministic(self):
        """Same input should always produce same output."""
        import copy
        items = [_item(0, 30, 10, 40), _item(0, 0, 10, 10), _item(0, 15, 10, 25)]
        result1 = sort_items_top_to_bottom_left_to_right(copy.deepcopy(items))
        result2 = sort_items_top_to_bottom_left_to_right(copy.deepcopy(items))
        for a, b in zip(result1, result2):
            assert a.bbox.y1 == b.bbox.y1


class TestSortLines:
    def test_lines_top_to_bottom(self):
        lines = [_line(50, "C"), _line(0, "A"), _line(25, "B")]
        sorted_lines = sort_lines_reading_order(lines)
        assert [l.text for l in sorted_lines] == ["A", "B", "C"]
