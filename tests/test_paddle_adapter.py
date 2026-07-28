"""Tests for PaddleOCR adapter."""

from __future__ import annotations

import pytest

from app.services.ocr.paddle_adapter import adapt_paddle_result
from app.services.ocr.models import BBox


class TestAdaptPaddleResult:
    """Test conversion of PaddleOCR raw output to internal OCRItem models."""

    @staticmethod
    def _make_detection(polygon, text="hello", confidence=0.95):
        """Create a PaddleOCR-style detection tuple."""
        return [polygon, [text, confidence]]

    def test_single_item(self):
        raw = [[
            self._make_detection(
                [[10, 20], [100, 20], [100, 40], [10, 40]],
                text="Hello",
                confidence=0.97,
            )
        ]]
        items = adapt_paddle_result(raw, page_number=1, page_width=500, page_height=800)

        assert len(items) == 1
        item = items[0]
        assert item.item_id == "p1_i0"
        assert item.page_number == 1
        assert item.text == "Hello"
        assert item.confidence == 0.97
        assert item.polygon == [[10, 20], [100, 20], [100, 40], [10, 40]]
        assert item.bbox.x1 == 10
        assert item.bbox.y1 == 20
        assert item.bbox.x2 == 100
        assert item.bbox.y2 == 40
        # Normalized bbox
        assert 0.0 <= item.normalized_bbox.x1 <= 1.0
        assert item.normalized_bbox.x1 == pytest.approx(10 / 500)
        assert item.normalized_bbox.y2 == pytest.approx(40 / 800)

    def test_multiple_items(self):
        raw = [[
            self._make_detection([[10, 10], [50, 10], [50, 30], [10, 30]], "A", 0.9),
            self._make_detection([[60, 10], [100, 10], [100, 30], [60, 30]], "B", 0.8),
        ]]
        items = adapt_paddle_result(raw, page_number=2, page_width=200, page_height=100)

        assert len(items) == 2
        assert items[0].item_id == "p2_i0"
        assert items[1].item_id == "p2_i1"
        assert items[0].text == "A"
        assert items[1].text == "B"

    def test_empty_result(self):
        items = adapt_paddle_result(None, page_number=1, page_width=100, page_height=100)
        assert items == []

    def test_empty_list(self):
        items = adapt_paddle_result([], page_number=1, page_width=100, page_height=100)
        assert items == []

    def test_empty_page(self):
        items = adapt_paddle_result([[]], page_number=1, page_width=100, page_height=100)
        assert items == []

    def test_malformed_detection_missing_polygon(self):
        raw = [[[None, ["text", 0.9]]]]  # polygon is None
        items = adapt_paddle_result(raw, page_number=1, page_width=100, page_height=100)
        assert items == []  # Should skip, not crash

    def test_malformed_detection_missing_recognition(self):
        raw = [[[[[10, 20], [30, 20], [30, 40], [10, 40]], None]]]
        items = adapt_paddle_result(raw, page_number=1, page_width=100, page_height=100)
        assert items == []

    def test_confidence_normalization_0_100_scale(self):
        """If PaddleOCR returns 0-100 scale, adapter should normalize to 0-1."""
        raw = [[
            self._make_detection(
                [[10, 20], [100, 20], [100, 40], [10, 40]],
                text="X",
                confidence=95.0,  # 0-100 scale
            )
        ]]
        items = adapt_paddle_result(raw, page_number=1, page_width=500, page_height=800)
        assert items[0].confidence == 0.95

    def test_confidence_already_0_1(self):
        raw = [[
            self._make_detection(
                [[10, 20], [100, 20], [100, 40], [10, 40]],
                text="X",
                confidence=0.88,
            )
        ]]
        items = adapt_paddle_result(raw, page_number=1, page_width=500, page_height=800)
        assert items[0].confidence == 0.88

    def test_negative_confidence_clamped(self):
        raw = [[
            self._make_detection(
                [[10, 20], [100, 20], [100, 40], [10, 40]],
                text="X",
                confidence=-0.5,
            )
        ]]
        items = adapt_paddle_result(raw, page_number=1, page_width=500, page_height=800)
        assert items[0].confidence == 0.0

    def test_invalid_bbox_clamped_to_page(self):
        """Items with coordinates outside page should be clamped."""
        raw = [[
            self._make_detection(
                [[-10, -10], [600, -10], [600, 900], [-10, 900]],
                text="X",
                confidence=0.9,
            )
        ]]
        items = adapt_paddle_result(raw, page_number=1, page_width=500, page_height=800)
        assert items[0].bbox.x1 == 0.0  # clamped
        assert items[0].bbox.y1 == 0.0
        assert items[0].bbox.x2 <= 501  # page_width + tolerance
        assert items[0].bbox.y2 <= 801

    def test_multiple_page_groups(self):
        """Multiple PaddleOCR page groups map to same page_number."""
        raw = [
            [self._make_detection([[10, 10], [50, 10], [50, 30], [10, 30]], "A", 0.9)],
            [self._make_detection([[10, 50], [50, 50], [50, 70], [10, 70]], "B", 0.8)],
        ]
        items = adapt_paddle_result(raw, page_number=1, page_width=200, page_height=200)
        assert len(items) == 2

    def test_empty_text(self):
        raw = [[
            self._make_detection(
                [[10, 20], [100, 20], [100, 40], [10, 40]],
                text="",
                confidence=0.5,
            )
        ]]
        items = adapt_paddle_result(raw, page_number=1, page_width=500, page_height=800)
        assert len(items) == 1
        assert items[0].text == ""

    def test_rotated_polygon(self):
        """Rotated/skewed text box."""
        raw = [[
            self._make_detection(
                [[50, 10], [110, 40], [80, 100], [20, 70]],
                text="skewed",
                confidence=0.85,
            )
        ]]
        items = adapt_paddle_result(raw, page_number=1, page_width=500, page_height=800)
        assert len(items) == 1
        assert items[0].bbox.x1 == 20
        assert items[0].bbox.y1 == 10
        assert items[0].bbox.x2 == 110
        assert items[0].bbox.y2 == 100
        # Original polygon preserved
        assert len(items[0].polygon) == 4

    def test_idempotent_ids(self):
        """Same input produces same IDs."""
        raw = [[
            self._make_detection([[10, 10], [50, 10], [50, 30], [10, 30]], "A", 0.9)
        ]]
        items1 = adapt_paddle_result(raw, page_number=3, page_width=200, page_height=100)
        items2 = adapt_paddle_result(raw, page_number=3, page_width=200, page_height=100)
        assert items1[0].item_id == items2[0].item_id
