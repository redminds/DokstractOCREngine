from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import require_admin_token
from app.core.service import OCREngineService


router = APIRouter(prefix="/api/v1/admin", tags=["admin"], dependencies=[Depends(require_admin_token)])


class ReleaseCreateRequest(BaseModel):
    release_tag: str = Field(..., min_length=1)
    image_digest: str = Field(..., min_length=1)
    supported_api_versions: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    notes: str = ""


class ReleaseValidationRequest(BaseModel):
    healthy: bool = True
    ready: bool = True
    validation_report: dict[str, object] = Field(default_factory=dict)


class PromotionRequest(BaseModel):
    reason: str = "manual promotion"


class RollbackRequest(BaseModel):
    reason: str = "manual rollback"


def get_engine_service() -> OCREngineService:
    from app.main import engine_service

    return engine_service


@router.get("/registry")
def registry_snapshot(service: OCREngineService = Depends(get_engine_service)):
    return service.snapshot()


@router.post("/releases")
def create_release(payload: ReleaseCreateRequest, service: OCREngineService = Depends(get_engine_service)):
    return service.register_release(
        release_tag=payload.release_tag,
        image_digest=payload.image_digest,
        supported_api_versions=tuple(payload.supported_api_versions),
        capabilities=tuple(payload.capabilities),
        notes=payload.notes,
    )


@router.post("/releases/{release_tag}/validate")
def validate_release(
    release_tag: str,
    payload: ReleaseValidationRequest,
    service: OCREngineService = Depends(get_engine_service),
):
    return service.validate_release(
        release_tag=release_tag,
        healthy=payload.healthy,
        ready=payload.ready,
        validation_report=payload.validation_report,
    )


@router.post("/releases/{release_tag}/promote")
def promote_release(
    release_tag: str,
    payload: PromotionRequest,
    service: OCREngineService = Depends(get_engine_service),
):
    return service.promote_release(release_tag=release_tag, reason=payload.reason)


@router.post("/releases/rollback")
def rollback_release(payload: RollbackRequest, service: OCREngineService = Depends(get_engine_service)):
    return service.rollback_release(reason=payload.reason)


@router.get("/audit-log")
def audit_log(limit: int = 100, service: OCREngineService = Depends(get_engine_service)):
    return service.audit_log(limit=limit)


@router.get("/notifications")
def notifications(limit: int = 50, service: OCREngineService = Depends(get_engine_service)):
    return service.audit_log(limit=limit)
