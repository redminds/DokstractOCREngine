from __future__ import annotations

from types import SimpleNamespace

from app.core import security as security_module


def test_authenticate_service_accepts_teaching_agent_identity(monkeypatch):
    monkeypatch.setattr(
        security_module,
        "SETTINGS",
        SimpleNamespace(
            ocr_api_token="ocr-api-token",
            schema_api_token="schema-api-token",
            teaching_ocr_engine_token="teaching-agent-token",
            admin_token="admin-token",
        ),
    )

    assert (
        security_module.authenticate_service(
            x_service_name="teaching-agent",
            x_service_token="teaching-agent-token",
        )
        == "teaching-agent"
    )


def test_authenticate_service_rejects_wrong_teaching_token(monkeypatch):
    monkeypatch.setattr(
        security_module,
        "SETTINGS",
        SimpleNamespace(
            ocr_api_token="ocr-api-token",
            schema_api_token="schema-api-token",
            teaching_ocr_engine_token="teaching-agent-token",
            admin_token="admin-token",
        ),
    )

    try:
        security_module.authenticate_service(
            x_service_name="teaching-agent",
            x_service_token="wrong-token",
        )
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 401
    else:
        raise AssertionError("Expected authentication failure")
