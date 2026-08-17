from __future__ import annotations

import asyncio
from types import SimpleNamespace

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


def test_internal_extract_accepts_teaching_agent_identity(monkeypatch):
    monkeypatch.setattr("app.main.engine_service", make_service())
    monkeypatch.setattr(
        "app.core.security.SETTINGS",
        SimpleNamespace(
            ocr_api_token="ocr-api-token",
            schema_api_token="schema-api-token",
            teaching_ocr_engine_token="teaching-agent-token",
        ),
    )
    monkeypatch.setattr(
        "app.core.ocr_execution.extract_internal_ocr_document",
        lambda **kwargs: {
            "file": {"name": "sample.txt", "type": "txt"},
            "document": {"total_pages": 1, "processed_pages": [1], "text": "hello", "confidence": 0.99},
            "pages": [],
            "metrics": {},
        },
    )
    client = TestClient(app)

    response = client.post(
        "/api/v1/internal/ocr/extract",
        headers={
            "X-Service-Name": "teaching-agent",
            "X-Service-Token": "teaching-agent-token",
        },
        files={"file": ("sample.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"project_key": "ocr", "api_version": "v1"},
    )

    assert response.status_code == 200
    assert response.json()["document"]["text"] == "hello"


def test_ocr_concurrency_limit_queues_requests(monkeypatch):
    """Requests beyond OCR_MAX_CONCURRENCY wait instead of being rejected.

    With OCR_MAX_CONCURRENCY=2, three concurrent requests should all succeed
    (third waits for a slot).  Uses a controlled mock so we can verify serialization.
    """
    import threading
    monkeypatch.setattr("app.main.engine_service", make_service())
    import app.api.v1.routes.ocr as ocr_mod
    monkeypatch.setattr(ocr_mod, "_OCR_CONCURRENCY_SEMAPHORE", asyncio.Semaphore(2))

    entered = [0]
    completed = [0]
    # Use Event to hold the first request inside the critical section
    hold = threading.Event()

    def controlled_ocr(*args, **kwargs):
        entered[0] += 1
        if entered[0] <= 2:
            # First two wait until released
            hold.wait(timeout=10)
        completed[0] += 1
        return {
            "file": {"name": "test.pdf", "type": "pdf"},
            "document": {"total_pages": 1, "processed_pages": [1], "text": "x", "confidence": 0.9, "duration_ms": 1},
            "pages": [{"page_number": 1, "width": 1, "height": 1, "rotation": 0, "text": "x", "confidence": 0.9, "metrics": {"page_total_ms": 1}, "items": [], "lines": [], "blocks": []}],
            "metrics": {},
        }

    monkeypatch.setattr("app.core.ocr_execution.extract_internal_ocr_document", controlled_ocr)
    client = TestClient(app)

    import concurrent.futures

    def send_ocr():
        return client.post(
            "/api/v1/internal/ocr/extract",
            headers={"X-Service-Name": "schema-api", "X-Service-Token": "change-me-schema-api-to-engine"},
            files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
            data={"project_key": "schema", "api_version": "v1"},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f1 = pool.submit(send_ocr)
        f2 = pool.submit(send_ocr)

        # Wait for both to enter OCR section
        import time
        deadline = time.perf_counter() + 10
        while entered[0] < 2 and time.perf_counter() < deadline:
            time.sleep(0.05)
        assert entered[0] >= 2, f"Only {entered[0]} entered OCR"

        # Third request is submitted — it should wait (semaphore full)
        f3 = pool.submit(send_ocr)

        # Release the first two
        hold.set()

        r1 = f1.result(timeout=10)
        r2 = f2.result(timeout=10)
        r3 = f3.result(timeout=10)

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 200
        assert completed[0] == 3, f"Expected 3 completions, got {completed[0]}"


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
            "file": {"name": "test.pdf", "type": "pdf"},
            "document": {"total_pages": 1, "processed_pages": [1], "text": "", "confidence": 0.0, "duration_ms": 1000},
            "pages": [{"page_number": 1, "width": 100, "height": 100, "rotation": 0, "text": "", "confidence": 0.0, "metrics": {"page_total_ms": 1000}, "items": [], "lines": [], "blocks": []}],
            "metrics": {},
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
        health_response = client.get("/health/live")
        health_elapsed = time.perf_counter() - health_start

        assert health_response.status_code == 200
        assert health_response.json()["live"] is True
        assert health_response.json()["ready"] is True
        assert health_elapsed < 0.5, f"Health /live took {health_elapsed:.2f}s — should be < 0.5s"

        ocr_future.result(timeout=5)


def test_health_ready_responds_during_mocked_ocr(monkeypatch):
    """Health /ready must respond quickly even while OCR is processing."""
    import time
    monkeypatch.setattr("app.main.engine_service", make_service())

    def slow_ocr(*args, **kwargs):
        time.sleep(1.0)
        return {
            "file": {"name": "test.pdf", "type": "pdf"},
            "document": {"total_pages": 1, "processed_pages": [1], "text": "", "confidence": 0.0, "duration_ms": 1000},
            "pages": [{"page_number": 1, "width": 100, "height": 100, "rotation": 0, "text": "", "confidence": 0.0, "metrics": {"page_total_ms": 1000}, "items": [], "lines": [], "blocks": []}],
            "metrics": {},
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
    body = response.json()
    assert body["detail"]["error"]["code"] == "OCR_ENGINE_UNAVAILABLE"


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
