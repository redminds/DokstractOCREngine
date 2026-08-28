from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from pathlib import Path

from app.core.registry import EngineRegistry, ReleaseSpec
from app.services.execution_outbox import OCREngineExecutionOutboxService


def _make_registry(db_path: Path) -> EngineRegistry:
    registry = EngineRegistry(db_path)
    registry.initialize(
        ReleaseSpec(
            release_tag="ocr-engine-2026.07.15",
            image_digest="sha256:dev-placeholder",
            supported_api_versions=("v1",),
            capabilities=("ocr", "pdf"),
            build_state="promoted",
            health_state="healthy",
            readiness_state="ready",
        )
    )
    return registry


def test_outbox_persists_event_across_registry_instances(monkeypatch):
    db_path = Path.cwd() / f"outbox_test_{os.getpid()}_persist.db"
    try:
        registry = _make_registry(db_path)
        service = OCREngineExecutionOutboxService(registry)

        monkeypatch.setattr(
            "app.services.execution_outbox.SETTINGS",
            SimpleNamespace(
                platform_reporting_enabled=False,
                platform_api_url="",
                platform_api_ocr_engine_token="",
                platform_reporting_request_timeout_seconds=1.0,
                platform_reporting_poll_interval_seconds=0.01,
                platform_reporting_claim_timeout_seconds=0.01,
                platform_reporting_retry_base_delay_seconds=0.01,
                platform_reporting_retry_max_delay_seconds=0.1,
                platform_reporting_max_attempts=3,
                platform_reporting_batch_size=10,
                platform_reporting_delivered_retention_seconds=60,
            ),
        )

        payload = {
            "event_id": "evt-001",
            "caller_service": "ocr-api",
            "operation": "extract",
            "page_count": 2,
            "status": "success",
            "duration_ms": 120,
        }

        asyncio.run(service.enqueue(payload))
        stats = registry.get_ocr_execution_outbox_stats()
        assert stats["pending"] == 1

        reloaded_registry = _make_registry(db_path)
        reloaded_stats = reloaded_registry.get_ocr_execution_outbox_stats()
        assert reloaded_stats["pending"] == 1
    finally:
        db_path.unlink(missing_ok=True)


def test_outbox_retry_marks_transport_failure_without_failing_ocr(monkeypatch):
    db_path = Path.cwd() / f"outbox_test_{os.getpid()}_retry.db"
    try:
        registry = _make_registry(db_path)
        service = OCREngineExecutionOutboxService(registry)

        monkeypatch.setattr(
            "app.services.execution_outbox.SETTINGS",
            SimpleNamespace(
                platform_reporting_enabled=True,
                platform_api_url="https://platform.example.test",
                platform_api_ocr_engine_token="platform-token",
                platform_reporting_request_timeout_seconds=1.0,
                platform_reporting_poll_interval_seconds=0.01,
                platform_reporting_claim_timeout_seconds=0.01,
                platform_reporting_retry_base_delay_seconds=0.01,
                platform_reporting_retry_max_delay_seconds=0.1,
                platform_reporting_max_attempts=3,
                platform_reporting_batch_size=10,
                platform_reporting_delivered_retention_seconds=60,
            ),
        )

        class DummyClient:
            async def report_ocr_execution(self, payload):
                from app.services.platform_reporting_client import PlatformReportingUnavailable

                raise PlatformReportingUnavailable("platform unreachable")

            async def aclose(self) -> None:
                return None

        monkeypatch.setattr("app.services.execution_outbox.PlatformExecutionReportingClient", DummyClient)

        asyncio.run(
            service.enqueue(
                {
                    "event_id": "evt-transport",
                    "caller_service": "schema-api",
                    "operation": "extract",
                    "page_count": 4,
                    "status": "success",
                    "duration_ms": 210,
                }
            )
        )

        processed = asyncio.run(service.deliver_due_events())
        assert processed == 1

        with registry._connection() as conn:
            row = conn.execute(
                "SELECT status, attempt_count, last_error_category FROM ocr_execution_outbox WHERE event_id = ?",
                ("evt-transport",),
            ).fetchone()
        assert row["status"] == "retry"
        assert row["attempt_count"] == 1
        assert row["last_error_category"] == "transport"
    finally:
        db_path.unlink(missing_ok=True)
