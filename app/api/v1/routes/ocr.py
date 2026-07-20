from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import SETTINGS
from app.core.ocr_execution import OCRDependencyUnavailable, extract_internal_ocr_document
from app.core.security import authenticate_service
from app.core.service import OCREngineService


logger = logging.getLogger("dokstract.ocr_engine.routes")
router = APIRouter(prefix="/api/v1/internal/ocr", tags=["ocr"])
_OCR_CONCURRENCY_SEMAPHORE = asyncio.Semaphore(max(1, SETTINGS.ocr_max_concurrency))
_AUTHORIZED_PROJECT_KEYS = {"ocr-api": "ocr", "schema-api": "schema"}
_ROUTE_ALLOWLIST = {
    "extract": {"ocr-api", "schema-api"},
    "auto": {"schema-api"},
}


def get_engine_service() -> OCREngineService:
    from app.main import engine_service

    return engine_service


def _parse_capabilities(raw: str | None, fallback: tuple[str, ...]) -> tuple[str, ...]:
    value = raw if raw is not None else ",".join(fallback)
    return tuple(item.strip() for item in value.split(",") if item.strip())


@asynccontextmanager
async def _acquire_engine_request_slot():
    """Acquire a concurrency slot.  Waits if no slots are available."""
    await _OCR_CONCURRENCY_SEMAPHORE.acquire()
    try:
        yield
    finally:
        _OCR_CONCURRENCY_SEMAPHORE.release()


@router.post("/extract")
@router.post("/auto")
async def extract_document(
    request: Request,
    file: UploadFile = File(...),
    project_key: str = Form("schema"),
    api_version: str = Form("v1"),
    capabilities: str | None = Form(None),
    image_processing: str = Form("false"),
    pages: str | None = Form(None),
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    x_correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
    x_job_id: str | None = Header(default=None, alias="X-Job-ID"),
    service_name: str = Depends(authenticate_service),
    service: OCREngineService = Depends(get_engine_service),
) -> dict[str, Any]:
    route_name = "auto" if request.url.path.endswith("/auto") else "extract"
    allowed_services = _ROUTE_ALLOWLIST[route_name]
    if service_name not in allowed_services:
        raise HTTPException(
            status_code=403,
            detail=f"Service '{service_name}' is not permitted to call this OCR route.",
        )
    requested_capabilities = _parse_capabilities(capabilities, ("ocr", "pdf"))
    primary_capability = requested_capabilities[0] if requested_capabilities else "ocr"
    normalized_project_key = project_key.strip().lower()
    expected_project_key = _AUTHORIZED_PROJECT_KEYS.get(service_name)
    if expected_project_key is None:
        raise HTTPException(status_code=403, detail=f"Service '{service_name}' is not permitted to access OCR routes.")
    if normalized_project_key != expected_project_key:
        raise HTTPException(
            status_code=403,
            detail=f"OCR engine internal OCR access is restricted to project_key='{expected_project_key}'.",
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

    total_start = time.perf_counter()
    async with _acquire_engine_request_slot():
        raw_bytes = await file.read()
        try:
            payload = await run_in_threadpool(
                extract_internal_ocr_document,
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

    total_elapsed = time.perf_counter() - total_start
    page_count = payload.get("pages", 1)
    line_count = payload.get("lines", 0)
    logger.info(
        "OCR request completed: file=%s pages=%d lines=%d total_seconds=%.2f",
        file.filename or "uploaded-file", page_count, line_count, total_elapsed,
    )

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
