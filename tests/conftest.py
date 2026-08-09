from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


@pytest.fixture(autouse=True)
def reset_ocr_route_state():
    """Keep OCR route semaphores isolated between tests."""
    from app.api.v1.routes import ocr as ocr_routes
    from app.core.config import SETTINGS

    ocr_routes._OCR_QUEUE_SEMAPHORE = None
    ocr_routes._OCR_CONCURRENCY_SEMAPHORE = asyncio.Semaphore(max(1, SETTINGS.ocr_max_concurrency))
    yield
