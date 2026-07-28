"""Tests for OCR geometry helpers."""

from __future__ import annotations

import pytest

from app.services.ocr.models import BBox, OCRItem
from app.services.ocr.geometry import (
    polygon_to_bbox,
    normalize_bbox,
    bbox_union,
    vertical_overlap,
    horizontal_overlap,
    horizontal_distance,
    center_point,
    median_item_height,
)


class TestPolygonToBBox:
    def test_rectangle(self):
        poly = [[10, 20], [100, 20], [100, 50], [10, 50]]
        bbox = polygon_to_bbox(poly)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 50

    def test_rotated_polygon(self):
        """A rotated rectangle should produce the correct enclosing bbox."""
        poly = [[50, 10], [110, 40], [80, 100], [20, 70]]
        bbox = polygon_to_bbox(poly)
        assert bbox.x1 == 20
        assert bbox.y1 == 10
        assert bbox.x2 == 110
        assert bbox.y2 == 100

    def test_irregular_polygon(self):
        poly = [[5, 15], [30, 5], [55, 20], [40, 45], [10, 35]]
        bbox = polygon_to_bbox(poly)
        assert bbox.x1 == 5
        assert bbox.y1 == 5
        assert bbox.x2 == 55
        assert bbox.y2 == 45

    def test_empty_polygon(self):
        bbox = polygon_to_bbox([])
        assert bbox.x1 == 0
        assert bbox.y1 == 0
        assert bbox.x2 == 0
        assert bbox.y2 == 0

    def test_single_point(self):
        bbox = polygon_to_bbox([[42, 73]])
        assert bbox.x1 == 42
        assert bbox.y1 == 73
        assert bbox.x2 == 42
        assert bbox.y2 == 73


class TestNormalizeBBox:
    def test_basic(self):
        bbox = BBox(100, 200, 300, 400)
        result = normalize_bbox(bbox, 1000, 800)
        assert result.x1 == 0.1
        assert result.y1 == 0.25
        assert result.x2 == 0.3
        assert result.y2 == 0.5

    def test_clamped_to_0_1(self):
        bbox = BBox(-10, -5, 1050, 850)
        result = normalize_bbox(bbox, 1000, 800)
        assert 0.0 <= result.x1 <= 1.0
        assert 0.0 <= result.y1 <= 1.0
        assert 0.0 <= result.x2 <= 1.0
        assert 0.0 <= result.y2 <= 1.0
        # Negative values clamped to 0
        assert result.x1 == 0.0
        assert result.y1 == 0.0

    def test_zero_page_dimensions(self):
        bbox = BBox(10, 20, 30, 40)
        result = normalize_bbox(bbox, 0, 0)
        assert result.x1 == 0.0
        assert result.y1 == 0.0
        assert result.x2 == 0.0
        assert result.y2 == 0.0

    def test_negative_page_dimensions(self):
        bbox = BBox(10, 20, 30, 40)
        result = normalize_bbox(bbox, -1, -1)
        assert result.x1 == 0.0

    def test_full_page(self):
        bbox = BBox(0, 0, 1000, 800)
        result = normalize_bbox(bbox, 1000, 800)
        assert result.x1 == 0.0
        assert result.y1 == 0.0
        assert result.x2 == 1.0
        assert result.y2 == 1.0


class TestBBoxUnion:
    def test_disjoint_boxes(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(20, 20, 30, 30)
        result = bbox_union(a, b)
        assert result.x1 == 0
        assert result.y1 == 0
        assert result.x2 == 30
        assert result.y2 == 30

    def test_overlapping_boxes(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 15, 15)
        result = bbox_union(a, b)
        assert result.x1 == 0
        assert result.y1 == 0
        assert result.x2 == 15
        assert result.y2 == 15

    def test_one_contains_other(self):
        a = BBox(0, 0, 100, 100)
        b = BBox(20, 20, 50, 50)
        result = bbox_union(a, b)
        assert result.x1 == 0
        assert result.y1 == 0
        assert result.x2 == 100
        assert result.y2 == 100

    def test_negative_coordinates(self):
        a = BBox(-10, -10, 10, 10)
        b = BBox(-5, -5, 20, 20)
        result = bbox_union(a, b)
        assert result.x1 == -10
        assert result.y1 == -10
        assert result.x2 == 20
        assert result.y2 == 20


class TestVerticalOverlap:
    def test_full_overlap(self):
        a = BBox(0, 100, 100, 200)
        b = BBox(50, 100, 150, 200)
        assert vertical_overlap(a, b) == 1.0

    def test_no_overlap(self):
        a = BBox(0, 0, 100, 50)
        b = BBox(0, 100, 100, 150)
        assert vertical_overlap(a, b) == 0.0

    def test_partial_overlap(self):
        a = BBox(0, 100, 100, 200)
        b = BBox(0, 150, 100, 250)
        overlap = vertical_overlap(a, b)
        assert 0.4 < overlap < 0.6  # ~0.5

    def test_one_inside_other(self):
        a = BBox(0, 100, 100, 300)
        b = BBox(0, 150, 100, 200)
        assert vertical_overlap(a, b) == 1.0

    def test_adjacent_same_height(self):
        """Adjacent boxes (touching) should have near-zero overlap."""
        a = BBox(0, 0, 10, 10)
        b = BBox(0, 10, 10, 20)
        assert vertical_overlap(a, b) == 0.0

    def test_zero_height(self):
        a = BBox(0, 0, 10, 0)
        b = BBox(0, 0, 10, 10)
        assert vertical_overlap(a, b) == 0.0


class TestHorizontalOverlap:
    def test_full_overlap(self):
        a = BBox(100, 0, 200, 100)
        b = BBox(100, 50, 200, 150)
        assert horizontal_overlap(a, b) == 1.0

    def test_no_overlap(self):
        a = BBox(0, 0, 50, 100)
        b = BBox(100, 0, 150, 100)
        assert horizontal_overlap(a, b) == 0.0


class TestHorizontalDistance:
    def test_separated(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(20, 0, 30, 10)
        assert horizontal_distance(a, b) == 10.0

    def test_overlapping(self):
        a = BBox(0, 0, 20, 10)
        b = BBox(10, 0, 30, 10)
        assert horizontal_distance(a, b) == 0.0

    def test_reverse_order(self):
        a = BBox(50, 0, 60, 10)
        b = BBox(10, 0, 20, 10)
        assert horizontal_distance(a, b) == 30.0


class TestCenterPoint:
    def test_simple(self):
        cx, cy = center_point(BBox(0, 0, 10, 10))
        assert cx == 5.0
        assert cy == 5.0

    def test_offset(self):
        cx, cy = center_point(BBox(10, 20, 30, 40))
        assert cx == 20.0
        assert cy == 30.0


class TestMedianItemHeight:
    def test_odd_count(self):
        items = [
            _make_item(0, 0, 10, 30),
            _make_item(0, 40, 10, 50),
            _make_item(0, 70, 10, 90),
        ]
        # heights: 20, 10, 20 → sorted: [10, 20, 20] → median = 20
        assert median_item_height(items) == 20.0

    def test_even_count(self):
        items = [
            _make_item(0, 0, 10, 10),
            _make_item(0, 10, 10, 30),
            _make_item(0, 30, 10, 60),
            _make_item(0, 60, 10, 100),
        ]
        # heights: 10, 20, 30, 40 → median = 25
        assert median_item_height(items) == 25.0

    def test_empty(self):
        assert median_item_height([]) == 0.0

    def test_zero_height_items(self):
        # Item with y1==y2 (zero height). bbox.height > 0 filters it out.
        items = [_make_item(0, 0, 10, 0)]  # height=0
        assert median_item_height(items) == 0.0


def _make_item(x1: float, y1: float, x2: float, y2: float) -> OCRItem:
    return OCRItem(
        item_id="test",
        page_number=1,
        text="test",
        confidence=0.9,
        polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        bbox=BBox(x1, y1, x2, y2),
        normalized_bbox=BBox(0, 0, 1, 1),
    )
