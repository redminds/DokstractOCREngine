"""Deterministic, privacy-safe OCR quality assessment.

This module deliberately evaluates only facts already produced by the OCR
pipeline.  It does not inspect OCR text, images, filenames, or geometry.
"""

from __future__ import annotations

from typing import Any


QUALITY_ASSESSMENT_VERSION = 1
QUALITY_REASON_CODES = {
    "page_processing_failed",
    "selected_page_processing_incomplete",
    "stitched_page_warning",
    "recovery_added_items",
    "recovery_attempted_no_items",
    "targeted_fallback_applied",
    "blank_page_classified",
    "digital_pdf_fast_path",
    "geometry_warning",
    "low_confidence",
    "low_text_coverage",
    "suspicious_empty_region",
}
QUALITY_REASON_SEVERITIES = {"info", "warning", "error"}

_REASON_DEFINITIONS = {
    "page_processing_failed": ("error", "A requested page did not complete OCR processing."),
    "selected_page_processing_incomplete": ("error", "Not all requested pages produced OCR output."),
    "stitched_page_warning": ("warning", "Page geometry triggered a generic review signal."),
    "recovery_added_items": ("info", "Generic OCR recovery added additional OCR evidence."),
    "recovery_attempted_no_items": ("warning", "Generic OCR recovery ran but added no additional OCR evidence."),
    "targeted_fallback_applied": ("info", "A bounded quality-gated OCR fallback was accepted."),
    "blank_page_classified": ("info", "The page was classified as blank and OCR was skipped."),
    "digital_pdf_fast_path": ("info", "Embedded PDF text extraction was used."),
    "geometry_warning": ("warning", "Canonical OCR geometry failed a generic integrity check."),
    "low_confidence": ("warning", "The page produced a low aggregate OCR confidence signal."),
    "low_text_coverage": ("warning", "OCR evidence covers an unusually small region of the page."),
    "suspicious_empty_region": ("warning", "The page contains a large empty vertical region between OCR evidence."),
}


def _pages(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _page_number(page: dict[str, Any]) -> int | None:
    value = page.get("page_number")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _reason(code: str, page_numbers: list[int] | None = None) -> dict[str, Any]:
    severity, safe_summary = _REASON_DEFINITIONS[code]
    return {
        "code": code,
        "severity": severity,
        "page_numbers": sorted(set(page_numbers or [])),
        "safe_summary": safe_summary,
    }


def build_quality_assessment(
    response: dict[str, Any],
    *,
    selected_pages: list[int] | None = None,
) -> dict[str, Any]:
    """Build the stable quality contract from existing page metrics only."""
    pages = _pages(response.get("pages"))
    selected = {int(page) for page in (selected_pages or [])}
    processed = {_page_number(page) for page in pages if _page_number(page) is not None}
    reasons: list[dict[str, Any]] = []
    page_summaries: list[dict[str, Any]] = []

    for page in pages:
        page_number = _page_number(page)
        metrics = page.get("metrics") or page.get("page_metrics") or {}
        metrics = metrics if isinstance(metrics, dict) else {}
        source = metrics.get("source")
        page_reasons: list[str] = []
        if source == "error":
            page_reasons.append("page_processing_failed")
            reasons.append(_reason("page_processing_failed", [page_number] if page_number else []))
        if metrics.get("stitched_warning"):
            page_reasons.append("stitched_page_warning")
            reasons.append(_reason("stitched_page_warning", [page_number] if page_number else []))
        recovery = metrics.get("recovery") if isinstance(metrics.get("recovery"), dict) else {}
        if recovery.get("attempted"):
            added = int(recovery.get("added_items") or 0)
            if added > 0:
                page_reasons.append("recovery_added_items")
                reasons.append(_reason("recovery_added_items", [page_number] if page_number else []))
            else:
                page_reasons.append("recovery_attempted_no_items")
                reasons.append(_reason("recovery_attempted_no_items", [page_number] if page_number else []))
        geometry = metrics.get("geometry_diagnostics") if isinstance(metrics.get("geometry_diagnostics"), dict) else {}
        quality_signals = set(geometry.get("quality_signals", []))
        if any(signal.startswith("GEOMETRY_") for signal in quality_signals):
            page_reasons.append("geometry_warning")
            reasons.append(_reason("geometry_warning", [page_number] if page_number else []))
        for signal, reason_code in (
            ("LOW_CONFIDENCE", "low_confidence"),
            ("LOW_TEXT_COVERAGE", "low_text_coverage"),
            ("SUSPICIOUS_EMPTY_REGION", "suspicious_empty_region"),
        ):
            if signal in quality_signals:
                page_reasons.append(reason_code)
                reasons.append(_reason(reason_code, [page_number] if page_number else []))
        if source == "blank":
            page_reasons.append("blank_page_classified")
            reasons.append(_reason("blank_page_classified", [page_number] if page_number else []))
        elif metrics.get("rendering_profile") == "digital":
            page_reasons.append("digital_pdf_fast_path")
            reasons.append(_reason("digital_pdf_fast_path", [page_number] if page_number else []))

        page_summaries.append({
            "page_number": page_number,
            "status": "failed" if source == "error" else "passed",
            "reason_codes": sorted(set(page_reasons)),
            "confidence": page.get("confidence"),
            "item_count": len(page.get("items") or []),
            "line_count": len(page.get("lines") or []),
            "rendering_profile": metrics.get("rendering_profile"),
        })

    if selected and selected != processed:
        reasons.append(_reason("selected_page_processing_incomplete", sorted(selected - processed)))

    processing = response.get("processing") if isinstance(response.get("processing"), dict) else {}
    image_processing = processing.get("image_processing") if isinstance(processing.get("image_processing"), dict) else {}
    targeted_fallback = image_processing.get("targeted_fallback")
    if isinstance(targeted_fallback, dict) and targeted_fallback.get("applied") is True:
        reasons.append(_reason("targeted_fallback_applied"))

    # Preserve deterministic insertion order while combining page-level facts.
    combined: dict[str, dict[str, Any]] = {}
    for item in reasons:
        existing = combined.get(item["code"])
        if existing is None:
            combined[item["code"]] = item
        else:
            existing["page_numbers"] = sorted(set(existing["page_numbers"]) | set(item["page_numbers"]))
    reasons = list(combined.values())
    severities = {item["severity"] for item in reasons}
    if "error" in severities:
        outcome = "failed"
    elif "warning" in severities:
        outcome = "needs_review"
    elif "recovery_added_items" in combined:
        outcome = "recovered"
    else:
        outcome = "passed"

    return {
        "version": QUALITY_ASSESSMENT_VERSION,
        "outcome": outcome,
        "reasons": reasons,
        "page_summaries": page_summaries,
    }
