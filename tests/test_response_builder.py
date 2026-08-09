"""Tests for canonical response builder."""

from __future__ import annotations

import json

from app.services.ocr.models import BBox, OCRItem, OCRLine, OCRBlock, OCRPage
from app.services.ocr.response_builder import build_response
from app.services.ocr.table_reconstruction import detect_table_region


def _make_page(
    page_number: int,
    width: float = 500,
    height: float = 800,
    texts: list[str] | None = None,
    confidences: list[float] | None = None,
) -> OCRPage:
    """Create a test OCRPage with items, lines, and blocks."""
    if texts is None:
        texts = ["Hello", "World"]
    if confidences is None:
        confidences = [0.95, 0.88]

    items = []
    lines = []
    y = 10.0
    for i, (text, conf) in enumerate(zip(texts, confidences)):
        item = OCRItem(
            item_id=f"p{page_number}_i{i}",
            page_number=page_number,
            text=text,
            confidence=conf,
            polygon=[[10, y], [100, y], [100, y + 20], [10, y + 20]],
            bbox=BBox(10, y, 100, y + 20),
            normalized_bbox=BBox(10 / width, y / height, 100 / width, (y + 20) / height),
            line_id=f"p{page_number}_l{i}",
            block_id=f"p{page_number}_b0",
            reading_order=i,
        )
        items.append(item)

        line = OCRLine(
            line_id=f"p{page_number}_l{i}",
            page_number=page_number,
            text=text,
            confidence=conf,
            bbox=BBox(10, y, 100, y + 20),
            normalized_bbox=BBox(10 / width, y / height, 100 / width, (y + 20) / height),
            item_ids=[item.item_id],
            block_id=f"p{page_number}_b0",
            reading_order=i,
        )
        lines.append(line)
        y += 30.0

    block = OCRBlock(
        block_id=f"p{page_number}_b0",
        page_number=page_number,
        block_type="text",
        confidence=sum(confidences) / len(confidences),
        bbox=BBox(10, 10, 100, y),
        normalized_bbox=BBox(10 / width, 10 / height, 100 / width, y / height),
        line_ids=[l.line_id for l in lines],
        reading_order=0,
    )

    page_text = "\n".join(texts)
    page_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return OCRPage(
        page_number=page_number,
        width=width,
        height=height,
        items=items,
        lines=lines,
        blocks=[block],
        text=page_text,
        confidence=page_confidence,
        page_metrics={
            "preprocessing_ms": 0.0,
            "ocr_inference_ms": 100.0,
            "paddle_adaptation_ms": 1.0,
            "line_reconstruction_ms": 2.0,
            "block_reconstruction_ms": 0.5,
            "reading_order_ms": 0.1,
            "geometry_total_ms": 3.6,
            "page_total_ms": 103.6,
        },
    )


class TestCanonicalResponse:
    def test_file_section(self):
        page = _make_page(1)
        response = build_response(
            pages_data=[page], filename="test.pdf", file_type="pdf",
            total_pages=5, selected_pages=[1],
        )
        assert response["file"] == {"name": "test.pdf", "type": "pdf"}

    def test_document_section(self):
        page = _make_page(1, confidences=[0.95, 0.85])
        response = build_response(
            pages_data=[page], filename="test.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1], total_duration_ms=1234.5,
        )
        doc = response["document"]
        assert doc["total_pages"] == 1
        assert doc["processed_pages"] == [1]
        assert doc["confidence"] == 0.9
        assert doc["duration_ms"] == 1234.5
        assert "text" in doc

    def test_pages_array(self):
        page = _make_page(1)
        response = build_response(
            pages_data=[page], filename="test.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
        )
        assert isinstance(response["pages"], list)
        assert len(response["pages"]) == 1

    def test_page_structure(self):
        page = _make_page(1)
        response = build_response(
            pages_data=[page], filename="test.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
        )
        p = response["pages"][0]
        assert p["page_number"] == 1
        assert p["width"] == 500
        assert p["height"] == 800
        assert p["rotation"] == 0
        assert "text" in p
        assert "confidence" in p
        assert "metrics" in p
        assert "page_total_ms" in p["metrics"]
        # duration_ms should NOT be at page level anymore
        assert "duration_ms" not in p

    def test_item_structure(self):
        page = _make_page(1, texts=["Hello"])
        response = build_response(
            pages_data=[page], filename="test.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
        )
        items = response["pages"][0]["items"]
        assert len(items) == 1
        item = items[0]
        assert item["item_id"] == "p1_i0"
        assert item["text"] == "Hello"
        assert item["confidence"] == 0.95
        assert len(item["polygon"]) == 4
        assert len(item["bbox"]) == 4
        assert item["bbox"][0] < item["bbox"][2]
        assert item["bbox"][1] < item["bbox"][3]
        for v in item["normalized_bbox"]:
            assert 0.0 <= v <= 1.0
        assert item["line_id"] is not None
        assert item["block_id"] is not None
        assert "reading_order" in item

    def test_line_structure(self):
        page = _make_page(1)
        response = build_response(
            pages_data=[page], filename="test.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
        )
        lines = response["pages"][0]["lines"]
        assert len(lines) == 2
        line = lines[0]
        assert "line_id" in line
        assert "text" in line
        assert "confidence" in line
        assert "bbox" in line
        assert "normalized_bbox" in line
        assert "item_ids" in line
        assert "reading_order" in line

    def test_line_normalized_bbox(self):
        page = _make_page(1)
        response = build_response(
            pages_data=[page], filename="test.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
        )
        line = response["pages"][0]["lines"][0]
        assert line["normalized_bbox"] is not None
        for v in line["normalized_bbox"]:
            assert 0.0 <= v <= 1.0

    def test_block_structure(self):
        page = _make_page(1)
        response = build_response(
            pages_data=[page], filename="test.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
        )
        blocks = response["pages"][0]["blocks"]
        assert len(blocks) == 1
        block = blocks[0]
        assert block["type"] == "text"
        assert "confidence" in block
        assert "bbox" in block
        assert "normalized_bbox" in block
        assert "line_ids" in block
        assert "reading_order" in block

    def test_block_normalized_bbox(self):
        page = _make_page(1)
        response = build_response(
            pages_data=[page], filename="test.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
        )
        block = response["pages"][0]["blocks"][0]
        assert block["normalized_bbox"] is not None
        for v in block["normalized_bbox"]:
            assert 0.0 <= v <= 1.0

    def test_empty_page(self):
        page = OCRPage(page_number=1, width=500, height=800)
        response = build_response(
            pages_data=[page], filename="empty.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
        )
        assert response["document"]["total_pages"] == 1
        assert response["document"]["confidence"] == 0.0
        assert response["document"]["text"] == ""
        assert response["pages"][0]["items"] == []
        assert response["pages"][0]["lines"] == []
        assert response["pages"][0]["blocks"] == []

    def test_multi_page(self):
        p1 = _make_page(1, texts=["Page1"])
        p2 = _make_page(2, texts=["Page2"])
        response = build_response(
            pages_data=[p1, p2], filename="multi.pdf", file_type="pdf",
            total_pages=2, selected_pages=[1, 2],
        )
        assert len(response["pages"]) == 2
        assert response["pages"][0]["page_number"] == 1
        assert response["pages"][1]["page_number"] == 2
        assert "Page1" in response["document"]["text"]
        assert "Page2" in response["document"]["text"]

    def test_json_serializable(self):
        page = _make_page(1)
        response = build_response(
            pages_data=[page], filename="test.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
        )
        json_str = json.dumps(response)
        assert len(json_str) > 0

    def test_document_text_from_lines(self):
        p1 = _make_page(1, texts=["First line", "Second line"])
        response = build_response(
            pages_data=[p1], filename="test.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
        )
        assert response["document"]["text"] == "First line\nSecond line"

    def test_id_consistency(self):
        page = _make_page(1, texts=["Hello"])
        response = build_response(
            pages_data=[page], filename="test.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
        )
        p = response["pages"][0]
        item_id = p["items"][0]["item_id"]
        line_item_ids = p["lines"][0]["item_ids"]
        assert item_id in line_item_ids

    def test_metrics_section(self):
        page = _make_page(1)
        metrics = {"rendering_ms": 100.0, "ocr_inference_ms": 500.0, "geometry_total_ms": 5.0}
        response = build_response(
            pages_data=[page], filename="test.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1], metrics=metrics,
        )
        assert "metrics" in response
        assert response["metrics"]["rendering_ms"] == 100.0
        assert response["metrics"]["ocr_inference_ms"] == 500.0
        assert response["metrics"]["geometry_total_ms"] == 5.0
        assert "total_duration_ms" in response["metrics"]
        # No extra serialization/response-size metrics
        assert "serialization_ms" not in response["metrics"]
        assert "response_size_bytes" not in response["metrics"]

    def test_confidence_scale_0_to_1(self):
        """All confidence values should be in 0-1 range."""
        page = _make_page(1, texts=["A", "B"], confidences=[0.8, 0.9])
        response = build_response(
            pages_data=[page], filename="test.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
        )
        assert 0.0 <= response["document"]["confidence"] <= 1.0
        for item in response["pages"][0]["items"]:
            assert 0.0 <= item["confidence"] <= 1.0
        for line in response["pages"][0]["lines"]:
            assert 0.0 <= line["confidence"] <= 1.0
        assert 0.0 <= response["pages"][0]["confidence"] <= 1.0

    def test_no_legacy_fields(self):
        """New contract must NOT contain old legacy field names."""
        page = _make_page(1)
        response = build_response(
            pages_data=[page], filename="test.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
        )
        assert "overall_confidence" not in response
        assert "combined_text" not in response
        assert "results" not in response
        assert "page_results" not in response
        assert "file_type" not in response
        assert "processed_pages" not in response
        assert "selected_pages" not in response

    def test_serialized_table_page_keeps_item_line_references_resolved(self):
        # Build a compact but detector-valid table-like page directly with stable table lines.
        table_items = [
            OCRItem(
                item_id="p1_i0",
                page_number=1,
                text="H1",
                confidence=0.9,
                polygon=[[100, 800], [180, 800], [180, 860], [100, 860]],
                bbox=BBox(100, 800, 180, 860),
                normalized_bbox=BBox(0.05, 0.266, 0.09, 0.286),
            ),
            OCRItem(
                item_id="p1_i1",
                page_number=1,
                text="H2",
                confidence=0.9,
                polygon=[[400, 800], [500, 800], [500, 860], [400, 860]],
                bbox=BBox(400, 800, 500, 860),
                normalized_bbox=BBox(0.2, 0.266, 0.25, 0.286),
            ),
            OCRItem(
                item_id="p1_i2",
                page_number=1,
                text="H3",
                confidence=0.9,
                polygon=[[700, 800], [800, 800], [800, 860], [700, 860]],
                bbox=BBox(700, 800, 800, 860),
                normalized_bbox=BBox(0.35, 0.266, 0.4, 0.286),
            ),
            OCRItem(
                item_id="p1_i3",
                page_number=1,
                text="H4",
                confidence=0.9,
                polygon=[[1000, 800], [1100, 800], [1100, 860], [1000, 860]],
                bbox=BBox(1000, 800, 1100, 860),
                normalized_bbox=BBox(0.5, 0.266, 0.55, 0.286),
            ),
            OCRItem(
                item_id="p1_i4",
                page_number=1,
                text="H5",
                confidence=0.9,
                polygon=[[1300, 800], [1400, 800], [1400, 860], [1300, 860]],
                bbox=BBox(1300, 800, 1400, 860),
                normalized_bbox=BBox(0.65, 0.266, 0.7, 0.286),
            ),
            OCRItem(
                item_id="p1_i5",
                page_number=1,
                text="D1C1",
                confidence=0.9,
                polygon=[[110, 950], [180, 950], [180, 1000], [110, 1000]],
                bbox=BBox(110, 950, 180, 1000),
                normalized_bbox=BBox(0.055, 0.316, 0.09, 0.333),
            ),
            OCRItem(
                item_id="p1_i6",
                page_number=1,
                text="D1C2",
                confidence=0.9,
                polygon=[[410, 950], [500, 950], [500, 1000], [410, 1000]],
                bbox=BBox(410, 950, 500, 1000),
                normalized_bbox=BBox(0.205, 0.316, 0.25, 0.333),
            ),
            OCRItem(
                item_id="p1_i7",
                page_number=1,
                text="D1C3",
                confidence=0.9,
                polygon=[[710, 950], [810, 950], [810, 1000], [710, 1000]],
                bbox=BBox(710, 950, 810, 1000),
                normalized_bbox=BBox(0.355, 0.316, 0.405, 0.333),
            ),
            OCRItem(
                item_id="p1_i8",
                page_number=1,
                text="D1C4",
                confidence=0.9,
                polygon=[[1010, 950], [1110, 950], [1110, 1000], [1010, 1000]],
                bbox=BBox(1010, 950, 1110, 1000),
                normalized_bbox=BBox(0.505, 0.316, 0.555, 0.333),
            ),
            OCRItem(
                item_id="p1_i9",
                page_number=1,
                text="D1C5",
                confidence=0.9,
                polygon=[[1310, 950], [1410, 950], [1410, 1000], [1310, 1000]],
                bbox=BBox(1310, 950, 1410, 1000),
                normalized_bbox=BBox(0.655, 0.316, 0.705, 0.333),
            ),
        ]
        lines, _, meta = detect_table_region(table_items, 600, 400)
        assert meta is not None
        page = OCRPage(page_number=1, width=600, height=400, items=table_items, lines=lines, blocks=[])
        response = build_response(
            pages_data=[page], filename="table.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
        )
        p = response["pages"][0]
        line_ids = {line["line_id"] for line in p["lines"]}
        assert any(item["line_id"] is not None for item in p["items"])
        for item in p["items"]:
            if item["line_id"] is not None:
                assert item["line_id"] in line_ids

