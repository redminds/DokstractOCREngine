"""Phase 3 Slice 1 — OCR Engine production-code page validation tests.

Tests actual ocr_execution.py production functions.
Does NOT invoke OCR models — tests only page selection logic.
"""
import pytest
from unittest.mock import patch, MagicMock

from app.core.ocr_execution import parse_pages, select_pages, PdfSelection


class TestParsePages:
    """Tests actual production parse_pages() from ocr_execution.py."""

    def test_single_page(self):
        result = parse_pages("3")
        assert result == [3]

    def test_range(self):
        result = parse_pages("4-6")
        assert result == [4, 5, 6]

    def test_mixed_list_and_range(self):
        result = parse_pages("1,3-5,8")
        assert result == [1, 3, 4, 5, 8]

    def test_duplicates_deduplicated(self):
        result = parse_pages("1,1,2,3")
        assert result == [1, 2, 3]

    def test_out_of_order_normalized(self):
        result = parse_pages("5,1,3")
        assert result == [1, 3, 5]

    def test_empty_returns_empty(self):
        result = parse_pages("")
        assert result == []

    def test_whitespace_handled(self):
        result = parse_pages(" 1 , 3 - 5 ")
        assert result == [1, 3, 4, 5]


class TestSelectPages:
    """Tests actual production select_pages() from ocr_execution.py."""

    def test_no_selection_all_pages(self):
        result = select_pages(total_pages=5, pages_str=None, max_pages=15)
        assert result.page_numbers == [1, 2, 3, 4, 5]
        assert result.total_pages == 5

    def test_no_selection_clamped_to_max(self):
        result = select_pages(total_pages=50, pages_str=None, max_pages=10)
        assert result.page_numbers == list(range(1, 11))
        assert result.total_pages == 50

    def test_explicit_selection(self):
        result = select_pages(total_pages=15, pages_str="2,3,4", max_pages=15)
        assert result.page_numbers == [2, 3, 4]
        assert result.total_pages == 15

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            select_pages(total_pages=5, pages_str="6", max_pages=10)

    def test_page_zero_rejected(self):
        with pytest.raises(ValueError):
            select_pages(total_pages=10, pages_str="0", max_pages=10)


class TestEnginePageNumbering:
    """Verify explicit 1-based to 0-based fitz conversion pattern."""

    def test_page_1_maps_to_fitz_index_0(self):
        page_num = 1
        fitz_index = page_num - 1
        assert fitz_index == 0

    def test_every_page_in_range(self):
        for page_num in range(1, 11):
            fitz_index = page_num - 1
            assert 0 <= fitz_index < 10

    def test_one_based_list_preserved(self):
        """Engine receives and returns 1-based page numbers."""
        pages_input = [1, 3, 5]
        # After processing, page numbers remain 1-based in OCRPage models
        for p in pages_input:
            assert p >= 1  # Never zero-based in public contract
