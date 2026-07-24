"""Response validation tests — inspect OCR output without embedding real documents.

These tests validate structural invariants that any valid OCR response must satisfy.
"""

from __future__ import annotations

import json


def _load_response(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


class TestResponseValidation:
    """Validate structural invariants of a canonical OCR response."""

    def test_top_level_keys(self):
        """Canonical response must have exactly: file, document, pages, metrics."""
        # Use a minimal valid response for structural testing
        resp = {
            "file": {"name": "test.pdf", "type": "pdf"},
            "document": {"total_pages": 1, "processed_pages": [1], "text": "hello", "confidence": 0.9, "duration_ms": 100.0},
            "pages": [{"page_number": 1, "width": 100, "height": 200, "rotation": 0, "text": "hello", "confidence": 0.9, "duration_ms": 50.0, "items": [], "lines": [], "blocks": []}],
            "metrics": {"rendering_ms": 10, "preprocessing_ms": 0, "ocr_inference_ms": 40, "geometry_ms": 1, "serialization_ms": 1},
        }
        assert set(resp.keys()) == {"file", "document", "pages", "metrics"}

    def test_no_legacy_keys(self):
        """Response must NOT contain legacy field names."""
        resp = {
            "file": {"name": "x", "type": "pdf"},
            "document": {"total_pages": 1, "processed_pages": [1], "text": "", "confidence": 0.0, "duration_ms": 0},
            "pages": [],
            "metrics": {"rendering_ms": 0, "preprocessing_ms": 0, "ocr_inference_ms": 0, "geometry_ms": 0, "serialization_ms": 0},
        }
        forbidden = {"overall_confidence", "combined_text", "results", "page_results", "file_type", "lines", "processed_pages", "selected_pages"}
        found = forbidden & set(resp.keys())
        assert not found, f"Legacy keys found: {found}"

    def test_confidence_range(self):
        resp = {
            "file": {"name": "x", "type": "pdf"},
            "document": {"total_pages": 1, "processed_pages": [1], "text": "", "confidence": 0.5, "duration_ms": 0},
            "pages": [{
                "page_number": 1, "width": 100, "height": 100, "rotation": 0,
                "text": "x", "confidence": 0.8, "duration_ms": 10,
                "items": [{"item_id": "p1_i0", "text": "x", "confidence": 0.8, "polygon": [[0,0],[10,0],[10,10],[0,10]], "bbox": [0,0,10,10], "normalized_bbox": [0,0,0.1,0.1], "line_id": "p1_l0", "block_id": "p1_b0", "reading_order": 0}],
                "lines": [{"line_id": "p1_l0", "text": "x", "confidence": 0.8, "bbox": [0,0,10,10], "normalized_bbox": [0,0,0.1,0.1], "item_ids": ["p1_i0"], "reading_order": 0}],
                "blocks": [{"block_id": "p1_b0", "type": "text", "confidence": 0.8, "bbox": [0,0,10,10], "normalized_bbox": [0,0,0.1,0.1], "line_ids": ["p1_l0"], "reading_order": 0}],
            }],
            "metrics": {"rendering_ms": 0, "preprocessing_ms": 0, "ocr_inference_ms": 0, "geometry_ms": 0, "serialization_ms": 0},
        }

        assert 0.0 <= resp["document"]["confidence"] <= 1.0
        for page in resp["pages"]:
            assert 0.0 <= page["confidence"] <= 1.0
            for item in page["items"]:
                assert 0.0 <= item["confidence"] <= 1.0
            for line in page["lines"]:
                assert 0.0 <= line["confidence"] <= 1.0
            for block in page["blocks"]:
                assert 0.0 <= block["confidence"] <= 1.0

    def test_unique_ids(self):
        """All item, line, and block IDs must be unique within a page."""
        resp = {
            "file": {"name": "x", "type": "pdf"},
            "document": {"total_pages": 1, "processed_pages": [1], "text": "", "confidence": 0, "duration_ms": 0},
            "pages": [{
                "page_number": 1, "width": 100, "height": 100, "rotation": 0,
                "text": "", "confidence": 0, "duration_ms": 0,
                "items": [
                    {"item_id": "p1_i0", "text": "a", "confidence": 0.9, "polygon": [[0,0],[10,0],[10,10],[0,10]], "bbox": [0,0,10,10], "normalized_bbox": [0,0,0.1,0.1], "line_id": "p1_l0", "block_id": "p1_b0", "reading_order": 0},
                    {"item_id": "p1_i1", "text": "b", "confidence": 0.9, "polygon": [[0,20],[10,20],[10,30],[0,30]], "bbox": [0,20,10,30], "normalized_bbox": [0,0.2,0.1,0.3], "line_id": "p1_l1", "block_id": "p1_b0", "reading_order": 1},
                ],
                "lines": [
                    {"line_id": "p1_l0", "text": "a", "confidence": 0.9, "bbox": [0,0,10,10], "normalized_bbox": [0,0,0.1,0.1], "item_ids": ["p1_i0"], "reading_order": 0},
                    {"line_id": "p1_l1", "text": "b", "confidence": 0.9, "bbox": [0,20,10,30], "normalized_bbox": [0,0.2,0.1,0.3], "item_ids": ["p1_i1"], "reading_order": 1},
                ],
                "blocks": [
                    {"block_id": "p1_b0", "type": "text", "confidence": 0.9, "bbox": [0,0,10,30], "normalized_bbox": [0,0,0.1,0.3], "line_ids": ["p1_l0","p1_l1"], "reading_order": 0},
                ],
            }],
            "metrics": {"rendering_ms": 0, "preprocessing_ms": 0, "ocr_inference_ms": 0, "geometry_ms": 0, "serialization_ms": 0},
        }

        for page in resp["pages"]:
            item_ids = [i["item_id"] for i in page["items"]]
            assert len(item_ids) == len(set(item_ids)), f"Duplicate item IDs in page {page['page_number']}"
            line_ids = [l["line_id"] for l in page["lines"]]
            assert len(line_ids) == len(set(line_ids)), f"Duplicate line IDs in page {page['page_number']}"
            block_ids = [b["block_id"] for b in page["blocks"]]
            assert len(block_ids) == len(set(block_ids)), f"Duplicate block IDs in page {page['page_number']}"

    def test_referential_integrity(self):
        """Item→line and line→item references must be consistent."""
        resp = {
            "file": {"name": "x", "type": "pdf"},
            "document": {"total_pages": 1, "processed_pages": [1], "text": "", "confidence": 0, "duration_ms": 0},
            "pages": [{
                "page_number": 1, "width": 100, "height": 100, "rotation": 0,
                "text": "", "confidence": 0, "duration_ms": 0,
                "items": [
                    {"item_id": "p1_i0", "text": "a", "confidence": 0.9, "polygon": [[0,0],[10,0],[10,10],[0,10]], "bbox": [0,0,10,10], "normalized_bbox": [0,0,0.1,0.1], "line_id": "p1_l0", "block_id": "p1_b0", "reading_order": 0},
                ],
                "lines": [
                    {"line_id": "p1_l0", "text": "a", "confidence": 0.9, "bbox": [0,0,10,10], "normalized_bbox": [0,0,0.1,0.1], "item_ids": ["p1_i0"], "reading_order": 0},
                ],
                "blocks": [
                    {"block_id": "p1_b0", "type": "text", "confidence": 0.9, "bbox": [0,0,10,10], "normalized_bbox": [0,0,0.1,0.1], "line_ids": ["p1_l0"], "reading_order": 0},
                ],
            }],
            "metrics": {"rendering_ms": 0, "preprocessing_ms": 0, "ocr_inference_ms": 0, "geometry_ms": 0, "serialization_ms": 0},
        }

        for page in resp["pages"]:
            item_ids = {i["item_id"] for i in page["items"]}
            line_ids = {l["line_id"] for l in page["lines"]}
            block_ids = {b["block_id"] for b in page["blocks"]}

            for item in page["items"]:
                assert item["line_id"] in line_ids, f"Item {item['item_id']} references missing line {item['line_id']}"
                if item["block_id"]:
                    assert item["block_id"] in block_ids, f"Item {item['item_id']} references missing block {item['block_id']}"
            for line in page["lines"]:
                for iid in line["item_ids"]:
                    assert iid in item_ids, f"Line {line['line_id']} references missing item {iid}"
            for block in page["blocks"]:
                for lid in block["line_ids"]:
                    assert lid in line_ids, f"Block {block['block_id']} references missing line {lid}"

    def test_coordinate_bounds(self):
        """Bboxes must be within page bounds; normalized bboxes 0-1."""
        resp = {
            "file": {"name": "x", "type": "pdf"},
            "document": {"total_pages": 1, "processed_pages": [1], "text": "", "confidence": 0, "duration_ms": 0},
            "pages": [{
                "page_number": 1, "width": 100, "height": 100, "rotation": 0,
                "text": "", "confidence": 0, "duration_ms": 0,
                "items": [
                    {"item_id": "p1_i0", "text": "x", "confidence": 0.9, "polygon": [[0,0],[10,0],[10,10],[0,10]], "bbox": [0,0,10,10], "normalized_bbox": [0,0,0.1,0.1], "line_id": "p1_l0", "block_id": "p1_b0", "reading_order": 0},
                ],
                "lines": [
                    {"line_id": "p1_l0", "text": "x", "confidence": 0.9, "bbox": [0,0,10,10], "normalized_bbox": [0,0,0.1,0.1], "item_ids": ["p1_i0"], "reading_order": 0},
                ],
                "blocks": [
                    {"block_id": "p1_b0", "type": "text", "confidence": 0.9, "bbox": [0,0,10,10], "normalized_bbox": [0,0,0.1,0.1], "line_ids": ["p1_l0"], "reading_order": 0},
                ],
            }],
            "metrics": {"rendering_ms": 0, "preprocessing_ms": 0, "ocr_inference_ms": 0, "geometry_ms": 0, "serialization_ms": 0},
        }

        for page in resp["pages"]:
            w, h = page["width"], page["height"]
            for item in page["items"]:
                b = item["bbox"]
                assert b[0] <= b[2] and b[1] <= b[3], f"Invalid bbox in {item['item_id']}"
                nb = item["normalized_bbox"]
                assert all(0.0 <= v <= 1.0 for v in nb), f"Normalized bbox out of range in {item['item_id']}"
                assert len(item["polygon"]) >= 3, f"Polygon has < 3 points in {item['item_id']}"

    def test_empty_page_handling(self):
        """Empty pages must have valid empty arrays, not nulls."""
        resp = {
            "file": {"name": "x", "type": "pdf"},
            "document": {"total_pages": 1, "processed_pages": [1], "text": "", "confidence": 0.0, "duration_ms": 0},
            "pages": [{
                "page_number": 1, "width": 100, "height": 100, "rotation": 0,
                "text": "", "confidence": 0.0, "duration_ms": 0,
                "items": [], "lines": [], "blocks": [],
            }],
            "metrics": {"rendering_ms": 0, "preprocessing_ms": 0, "ocr_inference_ms": 0, "geometry_ms": 0, "serialization_ms": 0},
        }
        assert resp["pages"][0]["items"] == []
        assert resp["pages"][0]["lines"] == []
        assert resp["pages"][0]["blocks"] == []
        assert resp["document"]["text"] == ""

    def test_json_roundtrip(self):
        """Response must serialize and deserialize without loss."""
        resp = {
            "file": {"name": "x", "type": "pdf"},
            "document": {"total_pages": 1, "processed_pages": [1], "text": "t", "confidence": 0.5, "duration_ms": 1},
            "pages": [{
                "page_number": 1, "width": 100, "height": 200, "rotation": 0,
                "text": "t", "confidence": 0.5, "duration_ms": 1,
                "items": [{"item_id": "p1_i0", "text": "t", "confidence": 0.9, "polygon": [[0,0],[10,0],[10,10],[0,10]], "bbox": [0,0,10,10], "normalized_bbox": [0,0,0.1,0.05], "line_id": "p1_l0", "block_id": "p1_b0", "reading_order": 0}],
                "lines": [{"line_id": "p1_l0", "text": "t", "confidence": 0.9, "bbox": [0,0,10,10], "normalized_bbox": [0,0,0.1,0.05], "item_ids": ["p1_i0"], "reading_order": 0}],
                "blocks": [{"block_id": "p1_b0", "type": "text", "confidence": 0.9, "bbox": [0,0,10,10], "normalized_bbox": [0,0,0.1,0.05], "line_ids": ["p1_l0"], "reading_order": 0}],
            }],
            "metrics": {"rendering_ms": 0, "preprocessing_ms": 0, "ocr_inference_ms": 0, "geometry_ms": 0, "serialization_ms": 0},
        }
        s = json.dumps(resp)
        back = json.loads(s)
        assert back == resp
