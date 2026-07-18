from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.security import authenticate_service
from app.core.service import OCREngineService


router = APIRouter(prefix="/api/v1/internal", tags=["internal"])


class DependencyRequest(BaseModel):
    capability: str = Field(..., min_length=1)
    capabilities: list[str] = Field(default_factory=list)
    api_version: str = Field(..., min_length=1)


def get_engine_service() -> OCREngineService:
    from app.main import engine_service

    return engine_service


@router.get("/releases/current")
def current_release(_service_name: str = Depends(authenticate_service), service: OCREngineService = Depends(get_engine_service)):
    return service.snapshot().get("active_release")


@router.post("/projects/{project_id}/dependencies/request")
def request_dependency(
    project_id: str,
    payload: DependencyRequest,
    service_name: str = Depends(authenticate_service),
    service: OCREngineService = Depends(get_engine_service),
):
    allowed_project = {"ocr-api": "ocr", "schema-api": "schema"}.get(service_name)
    if allowed_project is None:
        raise HTTPException(status_code=403, detail="Unknown service identity.")
    if project_id.strip().lower() != allowed_project:
        raise HTTPException(
            status_code=403,
            detail=f"Service '{service_name}' is not permitted to request dependencies for project_id='{project_id}'.",
        )
    outcome = service.request_dependency(
        project_id=project_id,
        capability=payload.capability,
        api_version=payload.api_version,
        capabilities=tuple(payload.capabilities) if payload.capabilities else None,
    )
    return outcome.__dict__
