from __future__ import annotations

from pathlib import Path

from app.core.registry import EngineRegistry
from app.core.service import OCREngineService


def make_service() -> OCREngineService:
    service = OCREngineService(registry=EngineRegistry(Path(":memory:")))
    service.initialize()
    return service


def test_request_assigned_to_active_release():
    service = make_service()

    outcome = service.request_dependency("ocr-api", "ocr", "v1")

    assert outcome.allowed is True
    assert outcome.manual_intervention_required is False
    assert outcome.assigned_release["release_tag"] == "ocr-engine-2026.07.15"
    assert service.registry.list_project_assignments()[0]["release_tag"] == "ocr-engine-2026.07.15"


def test_request_requires_promotion_for_compatible_future_release():
    service = make_service()
    service.register_release(
        release_tag="ocr-engine-2026.08.01",
        image_digest="sha256:new-digest",
        supported_api_versions=("v2",),
        capabilities=("ocr",),
        notes="compatibility update",
    )
    service.validate_release("ocr-engine-2026.08.01", healthy=True, ready=True, validation_report={"passed": True})

    outcome = service.request_dependency("schema-api", "ocr", "v2")

    assert outcome.allowed is False
    assert outcome.manual_intervention_required is True
    assert outcome.compatible_release["release_tag"] == "ocr-engine-2026.08.01"
    assert outcome.pending_request["status"] == "manual_required"
    assert service.registry.list_project_assignments() == []
    assert outcome.pending_request["requested_release_tag"] == "ocr-engine-2026.08.01"


def test_promote_release_reassigns_projects_and_rollback_restores_previous():
    service = make_service()
    service.request_dependency("ocr-api", "ocr", "v1")
    service.register_release(
        release_tag="ocr-engine-2026.08.01",
        image_digest="sha256:new-digest",
        supported_api_versions=("v1",),
        capabilities=("ocr",),
        notes="promoted image",
    )
    service.validate_release("ocr-engine-2026.08.01", healthy=True, ready=True, validation_report={"passed": True})

    promoted = service.promote_release("ocr-engine-2026.08.01", reason="dev promotion")
    assert promoted["active_release"]["release_tag"] == "ocr-engine-2026.08.01"
    assert service.registry.list_project_assignments()[0]["release_tag"] == "ocr-engine-2026.08.01"

    rolled_back = service.rollback_release(reason="failed smoke test")
    assert rolled_back["active_release"]["release_tag"] == "ocr-engine-2026.07.15"
    assert service.registry.list_project_assignments()[0]["release_tag"] == "ocr-engine-2026.07.15"
    assert service.registry.list_release_history()[0]["action"] == "rollback"


def test_failed_validation_of_active_release_triggers_automatic_rollback():
    service = make_service()
    service.request_dependency("ocr-api", "ocr", "v1")
    service.register_release(
        release_tag="ocr-engine-2026.08.01",
        image_digest="sha256:new-digest",
        supported_api_versions=("v1",),
        capabilities=("ocr",),
        notes="candidate image",
    )
    service.validate_release("ocr-engine-2026.08.01", healthy=True, ready=True, validation_report={"passed": True})
    service.promote_release("ocr-engine-2026.08.01", reason="dev promotion")

    failure = service.validate_release("ocr-engine-2026.08.01", healthy=False, ready=False, validation_report={"passed": False})

    assert failure["rollback_triggered"] is True
    assert failure["active_release"]["release_tag"] == "ocr-engine-2026.07.15"
    assert service.registry.list_project_assignments()[0]["release_tag"] == "ocr-engine-2026.07.15"
    assert service.registry.list_release_history()[0]["action"] == "rollback"
