from __future__ import annotations

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
        "/api/v1/internal/projects/ocr-api/dependencies/request",
        headers={"X-Engine-Internal-Token": "change-me-internal"},
        json={"capability": "ocr", "api_version": "v1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["allowed"] is True
    assert body["assigned_release"]["release_tag"] == "ocr-engine-2026.07.15"


def test_internal_extract_requires_internal_token(monkeypatch):
    monkeypatch.setattr("app.main.engine_service", make_service())
    client = TestClient(app)

    response = client.post(
        "/api/v1/internal/ocr/extract",
        files={"file": ("sample.txt", b"hello", "text/plain")},
        data={"project_key": "schema", "api_version": "v1"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid internal token."


def test_internal_extract_rejects_invalid_project_key(monkeypatch):
    monkeypatch.setattr("app.main.engine_service", make_service())
    client = TestClient(app)

    response = client.post(
        "/api/v1/internal/ocr/extract",
        headers={"X-Engine-Internal-Token": "change-me-internal"},
        files={"file": ("sample.txt", b"hello", "text/plain")},
        data={"project_key": "ocr-api", "api_version": "v1"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "OCR engine internal OCR access is restricted to project_key='schema'."


def test_internal_extract_rejects_when_engine_is_busy(monkeypatch):
    monkeypatch.setattr("app.main.engine_service", make_service())

    class BusyGate:
        def acquire(self, blocking=False):
            return False

        def release(self):
            raise AssertionError("release should not be called when acquisition fails")

    monkeypatch.setattr("app.api.v1.routes.ocr._ENGINE_REQUEST_GATE", BusyGate())
    client = TestClient(app)

    response = client.post(
        "/api/v1/internal/ocr/extract",
        headers={"X-Engine-Internal-Token": "change-me-internal"},
        files={"file": ("sample.txt", b"hello", "text/plain")},
        data={"project_key": "schema", "api_version": "v1"},
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "OCR engine is busy. Please retry."


def test_admin_registry_requires_token(monkeypatch):
    monkeypatch.setattr("app.main.engine_service", make_service())
    client = TestClient(app)

    response = client.get("/api/v1/admin/registry")

    assert response.status_code == 401
