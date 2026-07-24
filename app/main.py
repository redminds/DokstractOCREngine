from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes.admin import router as admin_router
from app.api.v1.routes.engine import router as engine_router
from app.api.v1.routes.ocr import router as ocr_router
from app.core.config import SETTINGS, validate_startup_configuration
from app.core.registry import EngineRegistry
from app.core.service import OCREngineService

logger = logging.getLogger("dokstract.ocr_engine")

registry = EngineRegistry(SETTINGS.registry_path)
engine_service = OCREngineService(registry=registry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting OCR Engine Service")
    validate_startup_configuration()
    engine_service.initialize()
    # Clean up stale temporary files from previous runs
    await run_in_threadpool(_sweep_stale_temp)
    logger.info("Initializing OCR model (PaddleOCR)...")
    init_start = time.perf_counter()
    try:
        await run_in_threadpool(_warm_up_ocr_engine)
        init_elapsed = time.perf_counter() - init_start
        logger.info("OCR model initialized successfully in %.2fs", init_elapsed)
    except Exception as exc:
        logger.error("OCR model initialization failed: %s", exc)
        raise
    # Initialize cache directory
    await run_in_threadpool(_init_cache_dir)
    yield
    logger.info("Stopping OCR Engine Service")


def _warm_up_ocr_engine() -> None:
    """Preload the OCR model so the first request does not pay the init cost."""
    from app.core.ocr_execution import _get_ocr_engine
    _get_ocr_engine()


def _sweep_stale_temp() -> None:
    """Clean up stale OCR temporary directories from previous runs."""
    from app.services.ocr_cache import sweep_stale_temp
    removed = sweep_stale_temp()
    if removed:
        logger.info("Startup: removed %d stale temp directories", removed)


def _init_cache_dir() -> None:
    """Ensure the OCR result cache directory exists."""
    from pathlib import Path
    from app.core.config import SETTINGS
    Path(SETTINGS.ocr_result_cache_dir).mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Dokstract OCR Engine Service",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if SETTINGS.app_env == "development" else None,
    redoc_url="/redoc" if SETTINGS.app_env == "development" else None,
    openapi_url="/openapi.json" if SETTINGS.app_env == "development" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.allow_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(engine_router)
app.include_router(ocr_router)
app.include_router(admin_router)


@app.get("/health")
def health():
    active = engine_service.snapshot().get("active_release")
    return {
        "status": "ok",
        "service": SETTINGS.service_name,
        "active_release": active["release_tag"] if active else None,
        "image_digest": active["image_digest"] if active else None,
        "supported_api_versions": active["supported_api_versions"] if active else [],
        "capabilities": active["capabilities"] if active else [],
    }


@app.get("/health/live")
def health_live():
    return {
        "status": "ok",
        "service": SETTINGS.service_name,
        "live": True,
        "ready": True,
    }


@app.get("/health/ready")
def health_ready():
    active = engine_service.snapshot().get("active_release")
    return {
        "status": "ok" if active and active.get("health_state") == "healthy" and active.get("readiness_state") == "ready" else "degraded",
        "service": SETTINGS.service_name,
        "live": True,
        "ready": bool(active and active.get("health_state") == "healthy" and active.get("readiness_state") == "ready"),
        "active_release": active["release_tag"] if active else None,
        "image_digest": active["image_digest"] if active else None,
        "supported_api_versions": active["supported_api_versions"] if active else [],
        "capabilities": active["capabilities"] if active else [],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port)
