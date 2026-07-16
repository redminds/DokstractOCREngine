from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import require_internal_token
from app.core.service import OCREngineService


router = APIRouter(prefix="/api/v1/internal", tags=["internal"], dependencies=[Depends(require_internal_token)])


class DependencyRequest(BaseModel):
    capability: str = Field(..., min_length=1)
    capabilities: list[str] = Field(default_factory=list)
    api_version: str = Field(..., min_length=1)


def get_engine_service() -> OCREngineService:
    from app.main import engine_service

    return engine_service


@router.get("/releases/current")
def current_release(service: OCREngineService = Depends(get_engine_service)):
    return service.snapshot().get("active_release")


@router.post("/projects/{project_id}/dependencies/request")
def request_dependency(
    project_id: str,
    payload: DependencyRequest,
    service: OCREngineService = Depends(get_engine_service),
):
    outcome = service.request_dependency(
        project_id=project_id,
        capability=payload.capability,
        api_version=payload.api_version,
        capabilities=tuple(payload.capabilities) if payload.capabilities else None,
    )
    return outcome.__dict__
