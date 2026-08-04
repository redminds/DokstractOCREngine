"""Phase 3 Slice 2 — Pure OCR page accounting summary.

Does NOT import cv2, PaddleOCR, or any GPU library.
Used by OCR Engine result builder and worker reporting.
"""
from __future__ import annotations

from typing import Optional


def build_page_accounting_summary(
    page_results: list[dict],
    selected_page_count: Optional[int] = None,
) -> dict[str, int]:
    """Calculate canonical page counts from per-page OCR results.

    Args:
        page_results: List of per-page result dicts, each with 'page_number' (int) and optionally 'status'.
        selected_page_count: Total selected pages, used to calculate skipped.

    Returns dict with: attempted_page_count, successful_page_count, failed_page_count, (skipped_page_count).

    Rules:
        - Attempted = count of unique page results (successful + failed)
        - A page counts as attempted only when a result exists for it
        - Duplicate page numbers are deduplicated (first result wins)
        - skipped = selected - attempted (only when selected_page_count is provided)
    """
    seen_pages: set[int] = set()
    attempted = 0
    successful = 0
    failed = 0

    for result in page_results:
        page_num = result.get("page_number")
        if page_num is None:
            continue
        try:
            page_num = int(page_num)
        except (ValueError, TypeError):
            continue
        if page_num in seen_pages:
            continue
        seen_pages.add(page_num)
        attempted += 1

        status = str(result.get("status", "")).lower()
        if status in ("failed", "error"):
            failed += 1
        else:
            successful += 1

    summary: dict[str, int] = {
        "attempted_page_count": attempted,
        "successful_page_count": successful,
        "failed_page_count": failed,
    }

    if selected_page_count is not None:
        skipped = max(0, selected_page_count - attempted)
        summary["skipped_page_count"] = skipped

    return summary
