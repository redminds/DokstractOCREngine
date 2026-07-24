"""Integration tests using a representative OCR fixture that simulates
a multi-row document with table-like content (no PaddleOCR dependency)."""

from __future__ import annotations

from app.services.ocr.models import OCRPage, BBox
from app.services.ocr.paddle_adapter import adapt_paddle_result
from app.services.ocr.line_reconstruction import reconstruct_lines
from app.services.ocr.block_reconstruction import reconstruct_blocks
from app.services.ocr.reading_order import (
    assign_reading_order_items,
    assign_reading_order_lines,
    assign_reading_order_blocks,
)
from app.services.ocr.response_builder import build_response


def _make_paddle_detection(text: str, confidence: float, polygon: list[list[float]]):
    """Create a PaddleOCR-format detection tuple."""
    return [polygon, [text, confidence]]


def _simulate_ocr_result(label_value_pairs: list[tuple[str, str, list[list[float]]]]) -> list[list]:
    """Convert label-value pairs at specific positions into PaddleOCR result format."""
    detections = []
    for text, _, polygon in label_value_pairs:
        detections.append(_make_paddle_detection(text, 0.95, polygon))
    return [detections]


class TestIntegrationFixture:
    """Full pipeline test with a simulated table-like document layout."""

    def test_full_pipeline_with_table_like_content(self):
        """Simulate a document with labels and values in a table-like arrangement.

        Layout (page 500x800):
          Row 1: "Permit No" (left)  "ABC123" (right)
          Row 2: "Site Area" (left)  "92.81"  (right)
          Row 3: "Date"      (left)  "2024-01-15" (right)
          Row 4: Paragraph text spanning full width
        """
        page_width = 500.0
        page_height = 800.0

        # Simulated PaddleOCR detections — note: intentionally out of order
        raw_result = [[
            _make_paddle_detection("Date", 0.97, [[20, 150], [80, 150], [80, 170], [20, 170]]),
            _make_paddle_detection("2024-01-15", 0.93, [[300, 150], [420, 150], [420, 170], [300, 170]]),
            _make_paddle_detection("Permit No", 0.98, [[20, 50], [120, 50], [120, 70], [20, 70]]),
            _make_paddle_detection("Site Area", 0.96, [[20, 100], [110, 100], [110, 120], [20, 120]]),
            _make_paddle_detection("92.81", 0.94, [[300, 100], [380, 100], [380, 120], [300, 120]]),
            _make_paddle_detection("ABC123", 0.99, [[300, 50], [400, 50], [400, 70], [300, 70]]),
            _make_paddle_detection(
                "This is a paragraph that spans", 0.91,
                [[20, 300], [480, 300], [480, 320], [20, 320]],
            ),
            _make_paddle_detection(
                "multiple lines of descriptive text.", 0.89,
                [[20, 330], [460, 330], [460, 350], [20, 350]],
            ),
        ]]

        # Step 1: Adapt
        items = adapt_paddle_result(raw_result, page_number=1, page_width=page_width, page_height=page_height)
        assert len(items) == 8

        # Step 2: Reconstruct lines
        lines = reconstruct_lines(items, page_width=page_width, page_height=page_height)
        # Should have 5 lines: 3 label-value pairs + 2 paragraph lines
        assert len(lines) >= 4  # some lines may merge if vertically close

        # Step 3: Reconstruct blocks
        blocks = reconstruct_blocks(lines, page_width=page_width, page_height=page_height)
        assert len(blocks) >= 1

        # Step 4: Reading order
        assign_reading_order_items(items)
        assign_reading_order_lines(lines)
        assign_reading_order_blocks(blocks)

        # Step 5: Build response
        page = OCRPage(
            page_number=1,
            width=page_width,
            height=page_height,
            items=items,
            lines=lines,
            blocks=blocks,
        )

        response = build_response(
            pages_data=[page],
            filename="test_document.pdf",
            file_type="pdf",
            total_pages=1,
            selected_pages=[1],
        )

        # Verify canonical response structure
        assert "file" in response
        assert "document" in response
        assert "pages" in response
        assert "metrics" in response
        # Document fields
        assert response["document"]["total_pages"] == 1
        assert "text" in response["document"]
        assert "confidence" in response["document"]

        # Verify structured output
        assert "pages" in response
        p = response["pages"][0]
        assert p["page_number"] == 1
        assert p["width"] == 500
        assert p["height"] == 800

        # Label/value pairs should be spatially distinguishable
        permit_items = [i for i in p["items"] if i["text"] == "Permit No"]
        abc_items = [i for i in p["items"] if i["text"] == "ABC123"]
        assert len(permit_items) == 1
        assert len(abc_items) == 1
        assert permit_items[0]["item_id"] != abc_items[0]["item_id"]

        # Find lines containing these items
        permit_line = next(
            (l for l in p["lines"] if permit_items[0]["item_id"] in l["item_ids"]),
            None,
        )
        abc_line = next(
            (l for l in p["lines"] if abc_items[0]["item_id"] in l["item_ids"]),
            None,
        )
        assert permit_line is not None
        assert abc_line is not None
        # Spatial positions should be distinct
        permit_center_x = (permit_items[0]["bbox"][0] + permit_items[0]["bbox"][2]) / 2
        abc_center_x = (abc_items[0]["bbox"][0] + abc_items[0]["bbox"][2]) / 2
        assert permit_center_x < abc_center_x  # Permit is left of ABC123

    def test_empty_document_pipeline(self):
        """Full pipeline with no text detected."""
        raw_result = None
        items = adapt_paddle_result(raw_result, page_number=1, page_width=500, page_height=800)
        assert items == []

        lines = reconstruct_lines(items, page_width=500, page_height=800)
        assert lines == []

        blocks = reconstruct_blocks(lines, page_width=500, page_height=800)
        assert blocks == []

        page = OCRPage(page_number=1, width=500, height=800)
        response = build_response(
            pages_data=[page],
            filename="empty.pdf",
            file_type="pdf",
            total_pages=1,
            selected_pages=[1],
        )

        assert response["document"]["total_pages"] == 1
        assert response["document"]["confidence"] == 0.0
        assert response["document"]["text"] == ""
        assert response["pages"][0]["items"] == []

    def test_multi_page_pipeline(self):
        """Two pages with separate content."""
        page_width = 500.0
        page_height = 800.0

        # Page 1
        raw1 = [[
            _make_paddle_detection("Page One", 0.99, [[20, 10], [120, 10], [120, 30], [20, 30]]),
        ]]
        # Page 2
        raw2 = [[
            _make_paddle_detection("Page Two", 0.98, [[20, 10], [120, 10], [120, 30], [20, 30]]),
        ]]

        pages_data = []
        for pg_num, raw in [(1, raw1), (2, raw2)]:
            items = adapt_paddle_result(raw, page_number=pg_num, page_width=page_width, page_height=page_height)
            lines = reconstruct_lines(items, page_width, page_height)
            blocks = reconstruct_blocks(lines, page_width, page_height)
            assign_reading_order_items(items)
            assign_reading_order_lines(lines)
            assign_reading_order_blocks(blocks)
            page = OCRPage(
                page_number=pg_num,
                width=page_width,
                height=page_height,
                items=items,
                lines=lines,
                blocks=blocks,
            )
            pages_data.append(page)

        response = build_response(
            pages_data=pages_data,
            filename="multi.pdf",
            file_type="pdf",
            total_pages=2,
            selected_pages=[1, 2],
        )

        assert response["document"]["processed_pages"] == [1, 2]
        assert len(response["pages"]) == 2
        assert response["pages"][0]["page_number"] == 1
        assert response["pages"][1]["page_number"] == 2

    def test_bounding_box_validity(self):
        """All bboxes in the response should have x1 < x2 and y1 < y2."""
        raw_result = [[
            _make_paddle_detection("Test", 0.95, [[10, 20], [100, 20], [100, 40], [10, 40]]),
        ]]

        items = adapt_paddle_result(raw_result, page_number=1, page_width=500, page_height=800)
        lines = reconstruct_lines(items, 500, 800)
        blocks = reconstruct_blocks(lines, 500, 800)
        assign_reading_order_items(items)
        assign_reading_order_lines(lines)
        assign_reading_order_blocks(blocks)

        page = OCRPage(page_number=1, width=500, height=800, items=items, lines=lines, blocks=blocks)
        response = build_response(
            pages_data=[page],
            filename="test.pdf",
            file_type="pdf",
            total_pages=1,
            selected_pages=[1],
        )

        p = response["pages"][0]
        for item in p["items"]:
            bbox = item["bbox"]
            assert bbox[0] <= bbox[2], f"x1 > x2 in item {item['item_id']}"
            assert bbox[1] <= bbox[3], f"y1 > y2 in item {item['item_id']}"
        for line in p["lines"]:
            bbox = line["bbox"]
            assert bbox[0] <= bbox[2], f"x1 > x2 in line {line['line_id']}"
            assert bbox[1] <= bbox[3], f"y1 > y2 in line {line['line_id']}"
        for block in p["blocks"]:
            bbox = block["bbox"]
            assert bbox[0] <= bbox[2], f"x1 > x2 in block {block['block_id']}"
            assert bbox[1] <= bbox[3], f"y1 > y2 in block {block['block_id']}"

    def test_confidence_scale_consistency(self):
        """All confidence values in the response should be in 0-1 range (items/lines/blocks)."""
        raw_result = [[
            _make_paddle_detection("A", 0.8, [[10, 10], [50, 10], [50, 30], [10, 30]]),
            _make_paddle_detection("B", 0.9, [[10, 50], [50, 50], [50, 70], [10, 70]]),
        ]]

        items = adapt_paddle_result(raw_result, page_number=1, page_width=200, page_height=200)
        lines = reconstruct_lines(items, 200, 200)
        blocks = reconstruct_blocks(lines, 200, 200)

        page = OCRPage(page_number=1, width=200, height=200, items=items, lines=lines, blocks=blocks)
        response = build_response(
            pages_data=[page],
            filename="test.pdf",
            file_type="pdf",
            total_pages=1,
            selected_pages=[1],
        )

        p = response["pages"][0]
        for item in p["items"]:
            assert 0.0 <= item["confidence"] <= 1.0, f"Item {item['item_id']} confidence out of range"
        for line in p["lines"]:
            assert 0.0 <= line["confidence"] <= 1.0, f"Line {line['line_id']} confidence out of range"
        # Document confidence is 0-1 scale
        assert 0.0 <= response["document"]["confidence"] <= 1.0
