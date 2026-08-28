from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from fastapi import FastAPI, Request, Response

SERVICE_NAME = "dokstract-ocr-engine"

ocr_engine_requests_total = Counter(
    "ocr_engine_requests_total",
    "Completed OCR execution attempts by caller service and status.",
    ["caller_service", "status"],
)
ocr_engine_pages_total = Counter(
    "ocr_engine_pages_total",
    "Pages processed by caller service.",
    ["caller_service"],
)
ocr_engine_request_duration_seconds = Histogram(
    "ocr_engine_request_duration_seconds",
    "OCR execution duration in seconds.",
    ["caller_service"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0),
)
ocr_execution_outbox_events = Gauge(
    "ocr_execution_outbox_events",
    "OCR execution outbox event counts by status.",
    ["status"],
)
ocr_execution_outbox_deliveries_total = Counter(
    "ocr_execution_outbox_deliveries_total",
    "OCR execution outbox delivery attempts by outcome.",
    ["status"],
)
ocr_execution_outbox_retry_attempts_total = Counter(
    "ocr_execution_outbox_retry_attempts_total",
    "OCR execution outbox retry attempts.",
)
ocr_execution_outbox_oldest_pending_age_seconds = Gauge(
    "ocr_execution_outbox_oldest_pending_age_seconds",
    "Age of the oldest OCR execution outbox event that is still pending or retryable.",
)
ocr_execution_outbox_last_successful_delivery_seconds = Gauge(
    "ocr_execution_outbox_last_successful_delivery_seconds",
    "Unix timestamp of the last successful OCR execution event delivery.",
)
ocr_execution_outbox_reporting_enabled = Gauge(
    "ocr_execution_outbox_reporting_enabled",
    "Whether OCR execution reporting is enabled and configured.",
)


def setup_metrics_endpoint(app: FastAPI) -> None:
    @app.get("/metrics", include_in_schema=False)
    async def metrics(_: Request) -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def record_ocr_execution_request(*, caller_service: str, status: str, page_count: int, duration_seconds: float) -> None:
    ocr_engine_requests_total.labels(caller_service=caller_service, status=status).inc()
    ocr_engine_pages_total.labels(caller_service=caller_service).inc(max(0, int(page_count)))
    ocr_engine_request_duration_seconds.labels(caller_service=caller_service).observe(max(0.0, float(duration_seconds)))


def set_outbox_reporting_enabled(enabled: bool) -> None:
    ocr_execution_outbox_reporting_enabled.set(1.0 if enabled else 0.0)


def update_outbox_metrics(*, pending: int, retry: int, dead_letter: int, oldest_pending_age_seconds: float | None, last_successful_delivery_at: str | None) -> None:
    ocr_execution_outbox_events.labels(status="pending").set(max(0, int(pending)))
    ocr_execution_outbox_events.labels(status="retry").set(max(0, int(retry)))
    ocr_execution_outbox_events.labels(status="dead_letter").set(max(0, int(dead_letter)))
    if oldest_pending_age_seconds is None:
        ocr_execution_outbox_oldest_pending_age_seconds.set(0.0)
    else:
        ocr_execution_outbox_oldest_pending_age_seconds.set(max(0.0, float(oldest_pending_age_seconds)))
    if last_successful_delivery_at:
        from datetime import datetime, timezone

        try:
            value = datetime.fromisoformat(last_successful_delivery_at.replace("Z", "+00:00"))
            ocr_execution_outbox_last_successful_delivery_seconds.set(value.replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            ocr_execution_outbox_last_successful_delivery_seconds.set(0.0)
    else:
        ocr_execution_outbox_last_successful_delivery_seconds.set(0.0)


def increment_outbox_retry_attempts() -> None:
    ocr_execution_outbox_retry_attempts_total.inc()
