"""Privacy-safe OCR health and failure diagnostics.

The structures in this module contain execution metadata and aggregate
signals only.  They deliberately do not include OCR text, images, filenames,
or domain-specific field names.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any


class OCRFailureCode(StrEnum):
    RENDER_FAILURE = "RENDER_FAILURE"
    DETECTION_FAILURE = "DETECTION_FAILURE"
    RECOGNITION_FAILURE = "RECOGNITION_FAILURE"
    PREDICTOR_RUNTIME_FAILURE = "PREDICTOR_RUNTIME_FAILURE"
    PREDICTOR_STATE_CORRUPTION = "PREDICTOR_STATE_CORRUPTION"
    GEOMETRY_INVALID = "GEOMETRY_INVALID"
    CACHE_READ_FAILURE = "CACHE_READ_FAILURE"
    CACHE_WRITE_FAILURE = "CACHE_WRITE_FAILURE"
    RESOURCE_EXHAUSTION = "RESOURCE_EXHAUSTION"
    PROCESSING_FAILURE = "PROCESSING_FAILURE"


def classify_failure(exc: BaseException, stage: str = "processing") -> OCRFailureCode:
    message = str(exc).lower()
    if "tensor holds no memory" in message or "preconditionnotmeterror" in message:
        return OCRFailureCode.PREDICTOR_STATE_CORRUPTION
    if "could not execute a primitive" in message or "mkldnn" in message or "onednn" in message:
        return OCRFailureCode.PREDICTOR_RUNTIME_FAILURE
    if "cache" in stage.lower():
        return OCRFailureCode.CACHE_WRITE_FAILURE if "write" in stage.lower() else OCRFailureCode.CACHE_READ_FAILURE
    if "render" in stage.lower():
        return OCRFailureCode.RENDER_FAILURE
    if "detect" in stage.lower():
        return OCRFailureCode.DETECTION_FAILURE
    if "recogn" in stage.lower():
        return OCRFailureCode.RECOGNITION_FAILURE
    if "limit" in message or "memory" in message or "resource" in message:
        return OCRFailureCode.RESOURCE_EXHAUSTION
    return OCRFailureCode.PROCESSING_FAILURE


def _safe_exception_type(exc: BaseException) -> str:
    return type(exc).__name__[:80]


def document_fingerprint(data: bytes) -> str:
    """Return a non-reversible document fingerprint for correlation only."""
    return hashlib.sha256(data).hexdigest()[:32]


def build_failure_bundle(
    *,
    execution_id: str,
    document_hash: str | None,
    engine_release: str,
    pipeline_version: str,
    recovery_version: str,
    failure_stage: str,
    exc: BaseException,
    page_number: int | None = None,
    warning_codes: list[str] | None = None,
    timings: dict[str, float] | None = None,
    configuration_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Build a sanitized bundle suitable for logs or internal telemetry."""
    code = classify_failure(exc, failure_stage)
    return {
        "bundle_version": 1,
        "execution_id": execution_id,
        "document_hash": document_hash,
        "engine_release": engine_release,
        "pipeline_version": pipeline_version,
        "recovery_version": recovery_version,
        "page_number": page_number,
        "failure_stage": failure_stage,
        "failure_code": code.value,
        "sanitized_exception": {"type": _safe_exception_type(exc), "code": code.value},
        "warning_codes": sorted(set(warning_codes or [])),
        "timings": {
            key: round(float(value), 3)
            for key, value in (timings or {}).items()
            if isinstance(value, (int, float))
        },
        "configuration_fingerprint": configuration_fingerprint,
    }


def build_ocr_health_summary(
    response: dict[str, Any],
    *,
    execution_id: str | None,
    engine_release: str,
    pipeline_version: str,
    recovery_version: str,
) -> dict[str, Any]:
    """Aggregate generic page health without copying customer text."""
    pages = response.get("pages") if isinstance(response.get("pages"), list) else []
    warning_codes: set[str] = set()
    recovery_pages: list[int] = []
    predictor_reset_count = 0
    page_summaries: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        metrics = page.get("metrics") if isinstance(page.get("metrics"), dict) else {}
        geometry = metrics.get("geometry_diagnostics") if isinstance(metrics.get("geometry_diagnostics"), dict) else {}
        signals = set(geometry.get("quality_signals", []))
        signals.update(metrics.get("warning_codes", []) if isinstance(metrics.get("warning_codes"), list) else [])
        warning_codes.update(str(signal) for signal in signals)
        recovery = metrics.get("recovery") if isinstance(metrics.get("recovery"), dict) else {}
        if recovery.get("attempted"):
            recovery_pages.append(int(page["page_number"])) if page.get("page_number") is not None else None
            warning_codes.add("RECOVERY_USED")
        predictor_reset_count += int(metrics.get("predictor_reset_count") or 0)
        page_summaries.append({
            "page_number": page.get("page_number"),
            "item_count": len(page.get("items") or []),
            "line_count": len(page.get("lines") or []),
            "confidence": page.get("confidence"),
            "warning_codes": sorted(signals),
            "recovery_triggered": bool(recovery.get("attempted")),
        })
    if any((metrics.get("source") == "error") for page in pages if isinstance(page, dict) for metrics in [page.get("metrics") or {}]):
        warning_codes.add("OCR_PARTIAL")
    status = "needs_review" if warning_codes else "normal"
    return {
        "version": 1,
        "execution_id": execution_id,
        "engine_release": engine_release,
        "pipeline_version": pipeline_version,
        "recovery_version": recovery_version,
        "status": status,
        "page_count": len(page_summaries),
        "recovery_pages": sorted(set(recovery_pages)),
        "predictor_reset_count": predictor_reset_count,
        "warning_codes": sorted(warning_codes),
        "pages": page_summaries,
    }
