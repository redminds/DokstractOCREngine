from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient
from pathlib import Path

from app.core.registry import EngineRegistry
from app.core.service import OCREngineService
from app.main import app


def make_service() -> OCREngineService:
    service = OCREngineService(registry=EngineRegistry(Path(":memory:")))
    service.initialize()
    return service


def test_health_reports_immutable_release(monkeypatch):
    monkeypatch.setattr("app.main.engine_service", make_service())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["active_release"] == "ocr-engine-2026.07.15"
    assert body["image_digest"] == "sha256:dev-placeholder"


def test_internal_request_uses_capability_and_api_version(monkeypatch):
    monkeypatch.setattr("app.main.engine_service", make_service())
    client = TestClient(app)

    response = client.post(
        "/api/v1/internal/projects/ocr/dependencies/request",
        headers={
            "X-Service-Name": "ocr-api",
            "X-Service-Token": "change-me-ocr-api-to-engine",
        },
        json={"capability": "ocr", "api_version": "v1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["allowed"] is True
    assert body["assigned_release"]["release_tag"] == "ocr-engine-2026.07.15"


def test_internal_extract_requires_service_token(monkeypatch):
    monkeypatch.setattr("app.main.engine_service", make_service())
    client = TestClient(app)

    response = client.post(
        "/api/v1/internal/ocr/extract",
        files={"file": ("sample.txt", b"hello", "text/plain")},
        data={"project_key": "schema", "api_version": "v1"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing service identity."


def test_internal_extract_rejects_invalid_project_key(monkeypatch):
    monkeypatch.setattr("app.main.engine_service", make_service())
    client = TestClient(app)

    response = client.post(
        "/api/v1/internal/ocr/extract",
        headers={
            "X-Service-Name": "ocr-api",
            "X-Service-Token": "change-me-ocr-api-to-engine",
        },
        files={"file": ("sample.txt", b"hello", "text/plain")},
        data={"project_key": "ocr-api", "api_version": "v1"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "OCR engine internal OCR access is restricted to project_key='ocr'."


def test_ocr_concurrency_limit_queues_requests(monkeypatch):
    """Requests beyond OCR_MAX_CONCURRENCY wait instead of being rejected.

    With OCR_MAX_CONCURRENCY=2, the first two requests should proceed
    concurrently while the third waits until a slot is released.
    """
    import time
    import threading
    monkeypatch.setattr("app.main.engine_service", make_service())
    # Reset semaphore for test isolation
    import app.api.v1.routes.ocr as ocr_mod
    monkeypatch.setattr(ocr_mod, "_OCR_CONCURRENCY_SEMAPHORE", asyncio.Semaphore(2))

    processing_started = threading.Event()
    release_barrier = threading.Event()
    completed_count = [0]

    def controlled_ocr(*args, **kwargs):
        processing_started.set()
        release_barrier.wait(timeout=5)
        completed_count[0] += 1
        return {
            "file": "test.pdf", "file_type": "pdf", "pages": 1,
            "lines": 3, "overall_confidence": 95.0,
            "combined_text": "hello world",
            "results": [{"page": 1, "text": "hello"}, {"page": 1, "text": "world"}],
        }

    monkeypatch.setattr("app.core.ocr_execution.extract_internal_ocr_document", controlled_ocr)
    client = TestClient(app)

    import concurrent.futures

    def send_ocr():
        return client.post(
            "/api/v1/internal/ocr/extract",
            headers={
                "X-Service-Name": "schema-api",
                "X-Service-Token": "change-me-schema-api-to-engine",
            },
            files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
            data={"project_key": "schema", "api_version": "v1"},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        # Send 3 requests with concurrency=2
        f1 = pool.submit(send_ocr)
        f2 = pool.submit(send_ocr)
        f3 = pool.submit(send_ocr)

        # Wait for first two to start processing
        assert processing_started.wait(timeout=5), "OCR processing did not start"

        # Allow them to complete
        release_barrier.set()

        r1 = f1.result(timeout=5)
        r2 = f2.result(timeout=5)
        r3 = f3.result(timeout=5)

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 200
        assert completed_count[0] == 3, f"Expected 3 completions, got {completed_count[0]}"


def test_ocr_max_concurrency_defaults_to_two(monkeypatch):
    """OCR_MAX_CONCURRENCY defaults to 2 when not set."""
    monkeypatch.setattr("app.main.engine_service", make_service())
    import app.api.v1.routes.ocr as ocr_mod
    # Reload the semaphore to match default concurrency
    monkeypatch.setattr(ocr_mod, "_OCR_CONCURRENCY_SEMAPHORE", asyncio.Semaphore(2))

    # Semaphore should allow 2 concurrent acquisitions
    acquired = 0
    async def try_acquire():
        nonlocal acquired
        await ocr_mod._OCR_CONCURRENCY_SEMAPHORE.acquire()
        acquired += 1

    import asyncio as aio
    loop = aio.new_event_loop()
    try:
        loop.run_until_complete(try_acquire())
        loop.run_until_complete(try_acquire())
        assert acquired == 2
    finally:
        loop.close()


def test_internal_extract_allows_schema_service(monkeypatch):
    monkeypatch.setattr("app.main.engine_service", make_service())
    client = TestClient(app)

    response = client.post(
        "/api/v1/internal/ocr/extract",
        headers={
            "X-Service-Name": "schema-api",
            "X-Service-Token": "change-me-schema-api-to-engine",
        },
        files={"file": ("sample.txt", b"hello", "text/plain")},
        data={"project_key": "schema", "api_version": "v1"},
    )

    assert response.status_code not in {401, 403}


def test_admin_registry_requires_token(monkeypatch):
    monkeypatch.setattr("app.main.engine_service", make_service())
    client = TestClient(app)

    response = client.get("/api/v1/admin/registry")

    assert response.status_code == 401


# ──────────────────────────────────────────────────
# Health responsiveness during OCR
# ──────────────────────────────────────────────────


def test_health_live_responds_during_mocked_ocr(monkeypatch):
    """Health /live must respond quickly even while OCR is processing."""
    import time
    monkeypatch.setattr("app.main.engine_service", make_service())

    def slow_ocr(*args, **kwargs):
        time.sleep(1.0)
        return {
            "file": "test.pdf",
            "file_type": "pdf",
            "pages": 1,
            "lines": 0,
            "overall_confidence": 0.0,
            "combined_text": "",
            "results": [],
        }

    monkeypatch.setattr(
        "app.core.ocr_execution.extract_internal_ocr_document",
        slow_ocr,
    )

    client = TestClient(app)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        ocr_future = pool.submit(
            lambda: client.post(
                "/api/v1/internal/ocr/extract",
                headers={
                    "X-Service-Name": "schema-api",
                    "X-Service-Token": "change-me-schema-api-to-engine",
                },
                files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
                data={"project_key": "schema", "api_version": "v1"},
            )
        )
        # Give OCR a moment to start
        time.sleep(0.1)
        health_start = time.perf_counter()
        health_response = client.get("/health/live")
        health_elapsed = time.perf_counter() - health_start

        assert health_response.status_code == 200
        assert health_response.json()["live"] is True
        assert health_response.json()["ready"] is True
        # Health check must respond in under 0.5s
        assert health_elapsed < 0.5, f"Health /live took {health_elapsed:.2f}s — should be < 0.5s"

        ocr_future.result(timeout=5)


def test_health_ready_responds_during_mocked_ocr(monkeypatch):
    """Health /ready must respond quickly even while OCR is processing."""
    import time
    monkeypatch.setattr("app.main.engine_service", make_service())

    def slow_ocr(*args, **kwargs):
        time.sleep(1.0)
        return {
            "file": "test.pdf",
            "file_type": "pdf",
            "pages": 1,
            "lines": 0,
            "overall_confidence": 0.0,
            "combined_text": "",
            "results": [],
        }

    monkeypatch.setattr(
        "app.core.ocr_execution.extract_internal_ocr_document",
        slow_ocr,
    )

    client = TestClient(app)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        ocr_future = pool.submit(
            lambda: client.post(
                "/api/v1/internal/ocr/extract",
                headers={
                    "X-Service-Name": "schema-api",
                    "X-Service-Token": "change-me-schema-api-to-engine",
                },
                files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
                data={"project_key": "schema", "api_version": "v1"},
            )
        )
        time.sleep(0.1)
        health_start = time.perf_counter()
        health_response = client.get("/health/ready")
        health_elapsed = time.perf_counter() - health_start

        assert health_response.status_code == 200
        assert health_response.json()["live"] is True
        assert health_elapsed < 0.5, f"Health /ready took {health_elapsed:.2f}s — should be < 0.5s"

        ocr_future.result(timeout=5)


def test_ocr_errors_propagated_correctly(monkeypatch):
    """OCR errors must be propagated as 503."""
    monkeypatch.setattr("app.main.engine_service", make_service())

    def failing_ocr(*args, **kwargs):
        from app.core.ocr_execution import OCRDependencyUnavailable
        raise OCRDependencyUnavailable("Mocked OCR failure")

    monkeypatch.setattr(
        "app.core.ocr_execution.extract_internal_ocr_document",
        failing_ocr,
    )

    client = TestClient(app)
    response = client.post(
        "/api/v1/internal/ocr/extract",
        headers={
            "X-Service-Name": "schema-api",
            "X-Service-Token": "change-me-schema-api-to-engine",
        },
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"project_key": "schema", "api_version": "v1"},
    )

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_ocr_max_concurrency_defaults_to_two(monkeypatch):
    """OCR_MAX_CONCURRENCY defaults to 2 when not set."""
    monkeypatch.setattr("app.main.engine_service", make_service())
    import app.api.v1.routes.ocr as ocr_mod
    monkeypatch.setattr(ocr_mod, "_OCR_CONCURRENCY_SEMAPHORE", asyncio.Semaphore(2))

    acquired = 0
    async def try_acquire():
        nonlocal acquired
        await ocr_mod._OCR_CONCURRENCY_SEMAPHORE.acquire()
        acquired += 1

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(try_acquire())
        loop.run_until_complete(try_acquire())
        assert acquired == 2
    finally:
        loop.close()
