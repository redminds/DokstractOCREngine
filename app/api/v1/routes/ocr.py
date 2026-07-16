from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile

from app.core.config import SETTINGS
from app.core.ocr_execution import OCRDependencyUnavailable, extract_internal_ocr_document
from app.core.security import require_internal_token
from app.core.service import OCREngineService


router = APIRouter(prefix="/api/v1/internal/ocr", tags=["ocr"], dependencies=[Depends(require_internal_token)])
_ENGINE_REQUEST_GATE = threading.BoundedSemaphore(max(1, SETTINGS.engine_max_inflight_requests))
_AUTHORIZED_PROJECT_KEY = "schema"


def get_engine_service() -> OCREngineService:
    from app.main import engine_service

    return engine_service


def _parse_capabilities(raw: str | None, fallback: tuple[str, ...]) -> tuple[str, ...]:
    value = raw if raw is not None else ",".join(fallback)
    return tuple(item.strip() for item in value.split(",") if item.strip())


@contextmanager
def _acquire_engine_request_slot():
    if not _ENGINE_REQUEST_GATE.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="OCR engine is busy. Please retry.")
    try:
        yield
    finally:
        _ENGINE_REQUEST_GATE.release()


@router.post("/extract")
@router.post("/auto")
def extract_document(
    file: UploadFile = File(...),
    project_key: str = Form("schema"),
    api_version: str = Form("v1"),
    capabilities: str | None = Form(None),
    image_processing: str = Form("false"),
    pages: str | None = Form(None),
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    x_correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
    x_job_id: str | None = Header(default=None, alias="X-Job-ID"),
    service: OCREngineService = Depends(get_engine_service),
) -> dict[str, Any]:
    requested_capabilities = _parse_capabilities(capabilities, ("ocr", "pdf"))
    primary_capability = requested_capabilities[0] if requested_capabilities else "ocr"
    normalized_project_key = project_key.strip().lower()
    if normalized_project_key != _AUTHORIZED_PROJECT_KEY:
        raise HTTPException(
            status_code=403,
            detail=f"OCR engine internal OCR access is restricted to project_key='{_AUTHORIZED_PROJECT_KEY}'.",
        )
    outcome = service.request_dependency(
        project_id=normalized_project_key,
        capability=primary_capability,
        api_version=api_version,
        capabilities=requested_capabilities,
    )

    if not outcome.allowed or not outcome.assigned_release:
        detail = {
            "message": outcome.message,
            "project_key": project_key,
            "requested_capabilities": requested_capabilities,
            "api_version": api_version,
            "manual_intervention_required": outcome.manual_intervention_required,
            "active_release": outcome.active_release,
            "compatible_release": outcome.compatible_release,
            "pending_request": outcome.pending_request,
            "blocking_projects": outcome.blocking_projects,
        }
        raise HTTPException(status_code=403, detail=detail)

    with _acquire_engine_request_slot():
        raw_bytes = file.file.read()
        try:
            payload = extract_internal_ocr_document(
                file_bytes=raw_bytes,
                filename=file.filename or "uploaded-file",
                content_type=file.content_type,
                image_processing=image_processing,
                pages=pages,
            )
        except OCRDependencyUnavailable as exc:
            raise HTTPException(status_code=503, detail="OCR engine unavailable.") from exc
        except ValueError as exc:
            message = str(exc)
            status_code = 413 if "Maximum" in message or "too large" in message.lower() else 400
            raise HTTPException(status_code=status_code, detail=message) from exc

    release = outcome.assigned_release
    payload.update(
        {
            "project_key": project_key,
            "api_version": api_version,
            "requested_capabilities": requested_capabilities,
            "request_id": x_request_id,
            "correlation_id": x_correlation_id,
            "job_id": x_job_id,
            "release_tag": release["release_tag"],
            "image_digest": release["image_digest"],
            "supported_api_versions": release["supported_api_versions"],
            "engine_capabilities": release["capabilities"],
        }
    )
    return payload
