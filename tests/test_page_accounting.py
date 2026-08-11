"""Phase 3 Slice 2 — Pure OCR page accounting tests. Does NOT import cv2."""
import pytest
from app.services.page_accounting import build_page_accounting_summary


class TestPageAccountingSummary:
    def test_all_success(self):
        results = [
            {"page_number": 1, "status": "success"},
            {"page_number": 2, "status": "success"},
            {"page_number": 3, "status": "success"},
            {"page_number": 4, "status": "success"},
        ]
        summary = build_page_accounting_summary(results, selected_page_count=4)
        assert summary["attempted_page_count"] == 4
        assert summary["successful_page_count"] == 4
        assert summary["failed_page_count"] == 0
        assert summary["skipped_page_count"] == 0

    def test_partial_failure(self):
        results = [
            {"page_number": 1, "status": "success"},
            {"page_number": 2, "status": "failed"},
            {"page_number": 3, "status": "success"},
            {"page_number": 4, "status": "success"},
        ]
        summary = build_page_accounting_summary(results, selected_page_count=4)
        assert summary["attempted_page_count"] == 4
        assert summary["successful_page_count"] == 3
        assert summary["failed_page_count"] == 1
        assert summary["attempted_page_count"] == summary["successful_page_count"] + summary["failed_page_count"]

    def test_all_failed(self):
        results = [{"page_number": 1, "status": "failed"}, {"page_number": 2, "status": "error"}]
        summary = build_page_accounting_summary(results, selected_page_count=2)
        assert summary["attempted_page_count"] == 2
        assert summary["successful_page_count"] == 0
        assert summary["failed_page_count"] == 2

    def test_unattempted_pages_excluded(self):
        """Only 4 of 15 pages were attempted."""
        results = [
            {"page_number": 1, "status": "success"},
            {"page_number": 3, "status": "success"},
            {"page_number": 5, "status": "success"},
            {"page_number": 7, "status": "success"},
        ]
        summary = build_page_accounting_summary(results, selected_page_count=4)
        assert summary["attempted_page_count"] == 4  # NOT 15!
        assert summary["successful_page_count"] == 4

    def test_skipped_pages(self):
        """3 attempted, 5 selected, 2 skipped."""
        results = [
            {"page_number": 1, "status": "success"},
            {"page_number": 2, "status": "success"},
            {"page_number": 3, "status": "failed"},
        ]
        summary = build_page_accounting_summary(results, selected_page_count=5)
        assert summary["attempted_page_count"] == 3
        assert summary["skipped_page_count"] == 2
        assert summary["attempted_page_count"] + summary["skipped_page_count"] == 5

    def test_no_selected_skipped_omitted(self):
        results = [{"page_number": 1, "status": "success"}]
        summary = build_page_accounting_summary(results)
        assert "skipped_page_count" not in summary

    def test_duplicates_deduplicated(self):
        """Duplicate page numbers count once (first result wins)."""
        results = [
            {"page_number": 1, "status": "success"},
            {"page_number": 1, "status": "failed"},  # duplicate — ignored
            {"page_number": 2, "status": "success"},
        ]
        summary = build_page_accounting_summary(results, selected_page_count=2)
        assert summary["attempted_page_count"] == 2  # Not 3
        assert summary["successful_page_count"] == 2  # First result was success

    def test_empty_results(self):
        summary = build_page_accounting_summary([], selected_page_count=4)
        assert summary["attempted_page_count"] == 0
        assert summary["successful_page_count"] == 0
        assert summary["failed_page_count"] == 0
        assert summary["skipped_page_count"] == 4

    def test_invariants(self):
        results = [
            {"page_number": 1, "status": "success"},
            {"page_number": 2, "status": "success"},
            {"page_number": 3, "status": "failed"},
            {"page_number": 4, "status": "success"},
            {"page_number": 5, "status": "error"},
        ]
        summary = build_page_accounting_summary(results, selected_page_count=5)
        assert summary["attempted_page_count"] == summary["successful_page_count"] + summary["failed_page_count"]
        assert summary["attempted_page_count"] + summary["skipped_page_count"] == 5
