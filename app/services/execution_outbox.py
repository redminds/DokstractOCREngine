from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi.concurrency import run_in_threadpool

from app.core.config import SETTINGS
from app.core.observability import (
    increment_outbox_retry_attempts,
    set_outbox_reporting_enabled,
    update_outbox_metrics,
)
from app.core.registry import EngineRegistry, OCREngineExecutionOutboxRecord
from app.services.platform_reporting_client import (
    PlatformExecutionReportingClient,
    PlatformReportingError,
    PlatformReportingHTTPError,
    PlatformReportingUnavailable,
)

logger = logging.getLogger("dokstract.ocr_engine.execution_outbox")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retry_delay_seconds(attempt_count: int) -> float:
    base = float(SETTINGS.platform_reporting_retry_base_delay_seconds)
    max_delay = float(SETTINGS.platform_reporting_retry_max_delay_seconds)
    exponent = max(0, int(attempt_count) - 1)
    return min(max_delay, base * (2 ** exponent))


def _is_retryable_exception(exc: Exception) -> tuple[bool, str]:
    if isinstance(exc, PlatformReportingUnavailable):
        return True, "transport"
    if isinstance(exc, PlatformReportingHTTPError):
        if exc.status_code in {429, 500, 502, 503, 504}:
            return True, "server"
        if exc.status_code in {400, 401, 403, 404}:
            return False, "permanent"
        return False, "permanent"
    if isinstance(exc, PlatformReportingError):
        return False, "permanent"
    return False, "permanent"


class OCREngineExecutionOutboxService:
    def __init__(self, registry: EngineRegistry):
        self._registry = registry
        self._reporting_client: PlatformExecutionReportingClient | None = None
        self._wake_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def is_enabled(self) -> bool:
        return bool(SETTINGS.platform_reporting_enabled and SETTINGS.platform_api_url.strip() and SETTINGS.platform_api_ocr_engine_token.strip())

    def snapshot(self) -> dict[str, Any]:
        stats = self._registry.get_ocr_execution_outbox_stats()
        update_outbox_metrics(
            pending=int(stats.get("pending") or 0),
            retry=int(stats.get("retry") or 0),
            dead_letter=int(stats.get("dead_letter") or 0),
            oldest_pending_age_seconds=stats.get("oldest_pending_age_seconds"),
            last_successful_delivery_at=stats.get("last_successful_delivery_at"),
        )
        set_outbox_reporting_enabled(self.is_enabled())
        return stats

    async def start(self) -> None:
        self.snapshot()
        if self._task is not None:
            return
        self._stop_event = asyncio.Event()
        self._wake_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="ocr-execution-outbox-worker")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        self._wake_event.set()
        try:
            await self._task
        finally:
            self._task = None
            if self._reporting_client is not None:
                await self._reporting_client.aclose()
                self._reporting_client = None

    def poke(self) -> None:
        self._wake_event.set()

    async def enqueue(self, payload: dict[str, Any]) -> None:
        await run_in_threadpool(self._registry.enqueue_ocr_execution_outbox_event, payload)
        self.snapshot()
        self.poke()

    async def deliver_due_events(self) -> int:
        if not self.is_enabled():
            self.snapshot()
            return 0

        due_records = await run_in_threadpool(
            self._registry.claim_due_ocr_execution_outbox_events,
            limit=max(1, int(SETTINGS.platform_reporting_batch_size)),
            claim_timeout_seconds=float(SETTINGS.platform_reporting_claim_timeout_seconds),
        )
        if not due_records:
            self.snapshot()
            return 0

        client = await self._get_client()
        processed = 0
        for record in due_records:
            await self._deliver_one(client, record)
            processed += 1

        await run_in_threadpool(
            self._registry.cleanup_delivered_ocr_execution_outbox_events,
            delivered_retention_seconds=int(SETTINGS.platform_reporting_delivered_retention_seconds),
        )
        self.snapshot()
        return processed

    async def _get_client(self) -> PlatformExecutionReportingClient:
        if self._reporting_client is None:
            self._reporting_client = PlatformExecutionReportingClient()
        return self._reporting_client

    async def _deliver_one(self, client: PlatformExecutionReportingClient, record: OCREngineExecutionOutboxRecord) -> None:
        payload = record.payload
        if not payload:
            await run_in_threadpool(
                self._registry.dead_letter_ocr_execution_outbox_event,
                event_id=record.event_id,
                last_error="Missing payload.",
                last_error_category="permanent",
            )
            return

        try:
            await client.report_ocr_execution(payload)
        except Exception as exc:
            retryable, category = _is_retryable_exception(exc)
            if retryable and int(record.attempt_count) < int(SETTINGS.platform_reporting_max_attempts):
                increment_outbox_retry_attempts()
                await run_in_threadpool(
                    self._registry.retry_ocr_execution_outbox_event,
                    event_id=record.event_id,
                    last_error=self._sanitize_error(str(exc), category),
                    last_error_category=category,
                    next_attempt_at=_now_iso_offset_seconds(_retry_delay_seconds(int(record.attempt_count))),
                )
                logger.warning(
                    "Failed to deliver OCR execution event %s, will retry (attempt=%s): %s",
                    record.event_id,
                    record.attempt_count,
                    exc,
                )
                return

            await run_in_threadpool(
                self._registry.dead_letter_ocr_execution_outbox_event,
                event_id=record.event_id,
                last_error=self._sanitize_error(str(exc), category),
                last_error_category=category,
            )
            logger.warning(
                "Dead-lettered OCR execution event %s after %s attempts: %s",
                record.event_id,
                record.attempt_count,
                exc,
            )
            return

        await run_in_threadpool(
            self._registry.mark_ocr_execution_outbox_delivered,
            event_id=record.event_id,
            delivered_at=_now_iso(),
        )

    async def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    processed = await self.deliver_due_events()
                except Exception as exc:
                    logger.warning("OCR execution outbox delivery loop error: %s", exc)
                    processed = 0

                if self._stop_event.is_set():
                    break
                if processed > 0:
                    continue
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=float(SETTINGS.platform_reporting_poll_interval_seconds))
                except asyncio.TimeoutError:
                    pass
                self._wake_event.clear()
        finally:
            if self._reporting_client is not None:
                await self._reporting_client.aclose()
                self._reporting_client = None

    @staticmethod
    def _sanitize_error(message: str, category: str) -> str:
        text = " ".join(str(message or "").split())
        if category == "permanent" and not text:
            text = "Permanent Platform reporting error."
        if len(text) > 240:
            text = text[:237] + "..."
        return text


def _now_iso_offset_seconds(seconds: float) -> str:
    return datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() + seconds, tz=timezone.utc).isoformat()
