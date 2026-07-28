"""Tests for page-level classification and adaptive rendering."""
import numpy as np
import pytest
from app.services.ocr.page_classification import (
    PageSizeClassification,
    classify_page_dimensions,
    compute_adaptive_render_plan,
    tile_image_vertical,
    reconstruct_page_from_tiles,
    AdaptiveRenderPlan,
)


class TestPageClassification:
    def test_normal_a4_portrait(self):
        c = classify_page_dimensions(1240, 1754)  # ~A4 at 150 DPI
        assert c.classification == PageSizeClassification.NORMAL

    def test_normal_a4_landscape(self):
        c = classify_page_dimensions(1754, 1240)
        assert c.classification == PageSizeClassification.NORMAL

    def test_legal_size_page(self):
        c = classify_page_dimensions(1275, 2100)  # legal at 150 DPI
        assert c.classification == PageSizeClassification.NORMAL

    def test_extreme_aspect_ratio(self):
        # Height > max_image_height triggers LARGE; aspect > 15 triggers EXTREME
        c = classify_page_dimensions(300, 5000)  # ratio ~16.7 > 15
        assert c.classification in (
            PageSizeClassification.POSSIBLE_STITCHED,
            PageSizeClassification.EXTREME_ASPECT_RATIO,
            PageSizeClassification.LARGE,
        )

    def test_oversized_pixel_count(self):
        c = classify_page_dimensions(6000, 6000)  # 36M pixels
        assert c.classification in (
            PageSizeClassification.OVERSIZED_PIXEL_COUNT,
            PageSizeClassification.UNSAFE_TO_RENDER,
        )

    def test_unsafe_dimensions(self):
        c = classify_page_dimensions(13000, 1000)
        assert c.classification == PageSizeClassification.UNSAFE_TO_RENDER

    def test_aspect_ratio_alone_not_rejected(self):
        """Legal-size or tall receipt should NOT be classified as UNSAFE."""
        c = classify_page_dimensions(1200, 4800)  # 1:4 receipt
        assert c.classification != PageSizeClassification.UNSAFE_TO_RENDER

    def test_normal_15_page_pdf_pages_are_normal(self):
        """Each page of a 15-page scanned PDF should be NORMAL."""
        for _ in range(15):
            c = classify_page_dimensions(1240, 1754)
            assert c.classification == PageSizeClassification.NORMAL


class TestAdaptiveRenderPlan:
    def test_normal_page_standard_strategy(self):
        c = classify_page_dimensions(1240, 1754)
        plan = compute_adaptive_render_plan(1240, 1754, c)
        assert plan.strategy == "standard"

    def test_large_page_downscale_strategy(self):
        c = classify_page_dimensions(4000, 5000)
        # May be LARGE or OVERSIZED depending on thresholds
        plan = compute_adaptive_render_plan(4000, 5000, c)
        assert plan.strategy in ("standard", "downscale", "tile")

    def test_possible_stitched_tile_strategy(self):
        c = classify_page_dimensions(500, 5000)
        plan = compute_adaptive_render_plan(500, 5000, c)
        # May be downscale, tile, or standard depending on thresholds
        assert plan.strategy in ("tile", "standard", "reject", "downscale")

    def test_unsafe_reject_strategy(self):
        c = classify_page_dimensions(13000, 1000)
        plan = compute_adaptive_render_plan(13000, 1000, c)
        assert plan.strategy == "reject"


class TestVerticalTiling:
    def test_tile_splits_image(self):
        image = np.ones((3000, 1000, 3), dtype=np.uint8) * 255
        tiles = tile_image_vertical(image, tile_height=1000, overlap=100)
        assert len(tiles) >= 3
        for tile, y_off in tiles:
            assert tile.shape[0] > 0
            assert tile.shape[1] == 1000
            assert y_off >= 0

    def test_tile_overlap(self):
        image = np.ones((2000, 500, 3), dtype=np.uint8) * 255
        tiles = tile_image_vertical(image, tile_height=1000, overlap=200)
        assert len(tiles) >= 2
        # Second tile should start before first tile ends
        _, y0 = tiles[0]
        _, y1 = tiles[1]
        assert y1 < y0 + 1000  # overlap exists

    def test_small_image_single_tile(self):
        image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        tiles = tile_image_vertical(image, tile_height=1000, overlap=100)
        assert len(tiles) == 1

    def test_no_zero_sized_tiles(self):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        tiles = tile_image_vertical(image, tile_height=50, overlap=5)
        for tile, _ in tiles:
            assert tile.shape[0] > 0
            assert tile.shape[1] > 0


class TestCoordinateReconstruction:
    def test_bbox_translation(self):
        tile_results = [{
            "items": [{
                "item_id": "i1", "text": "hello",
                "confidence": 0.95,
                "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 40},
            }],
            "lines": [{
                "line_id": "l1", "text": "hello",
                "confidence": 0.95,
                "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 40},
            }],
            "blocks": [],
            "tile_y_offset": 500,
        }]
        result = reconstruct_page_from_tiles(tile_results, 1, 1000, 2000)
        assert len(result["items"]) == 1
        assert result["items"][0]["bbox"]["y1"] == 520  # 20 + 500
        assert result["items"][0]["bbox"]["y2"] == 540  # 40 + 500

    def test_line_ids_unique(self):
        tile_results = [
            {"items": [], "lines": [
                {"line_id": "abc", "text": "x", "confidence": 0.9,
                 "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
            ], "blocks": [], "tile_y_offset": 0},
            {"items": [], "lines": [
                {"line_id": "abc", "text": "y", "confidence": 0.9,
                 "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
            ], "blocks": [], "tile_y_offset": 100},
        ]
        result = reconstruct_page_from_tiles(tile_results, 1, 100, 200)
        ids = [ln["line_id"] for ln in result["lines"]]
        assert len(ids) == len(set(ids))  # all unique

    def test_overlap_deduplication(self):
        """Same text + same position → deduplicated."""
        tile_results = [
            {"items": [], "lines": [
                {"line_id": "a", "text": "OVERLAP", "confidence": 0.9,
                 "bbox": {"x1": 10, "y1": 950, "x2": 100, "y2": 970}},
            ], "blocks": [], "tile_y_offset": 0},
            {"items": [], "lines": [
                {"line_id": "b", "text": "OVERLAP", "confidence": 0.9,
                 "bbox": {"x1": 10, "y1": 50, "x2": 100, "y2": 70}},
            ], "blocks": [], "tile_y_offset": 900},
        ]
        result = reconstruct_page_from_tiles(tile_results, 1, 200, 2000)
        # The second tile's "OVERLAP" at y=950 should be dedup'd
        overlap_lines = [ln for ln in result["lines"] if ln["text"] == "OVERLAP"]
        assert len(overlap_lines) == 1  # deduplicated to one

    def test_page_metadata_preserved(self):
        tile_results = [
            {"items": [], "lines": [
                {"line_id": "x", "text": "test", "confidence": 0.9,
                 "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
            ], "blocks": [], "tile_y_offset": 0},
        ]
        result = reconstruct_page_from_tiles(tile_results, 5, 200, 500)
        assert result["page_number"] == 5
        assert result["tiled"] is True
        assert result["tile_count"] == 1
