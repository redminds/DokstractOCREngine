from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from app.core.config import SETTINGS


_SERVICE_TOKENS = {
    "ocr-api": lambda: SETTINGS.ocr_api_token,
    "schema-api": lambda: SETTINGS.schema_api_token,
    "teaching-agent": lambda: SETTINGS.teaching_ocr_engine_token,
}


def authenticate_service(
    x_service_name: str | None = Header(default=None, alias="X-Service-Name"),
    x_service_token: str | None = Header(default=None, alias="X-Service-Token"),
) -> str:
    service_name = (x_service_name or "").strip().lower()
    token = (x_service_token or "").strip()
    if not service_name or not token:
        raise HTTPException(status_code=401, detail="Missing service identity.")

    token_getter = _SERVICE_TOKENS.get(service_name)
    if token_getter is None:
        raise HTTPException(status_code=403, detail="Unknown service identity.")

    expected = token_getter().strip()
    if not expected:
        raise HTTPException(status_code=503, detail=f"{service_name} token is not configured.")

    if not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid service token.")

    return service_name


def require_admin_token(x_engine_admin_token: str | None = Header(default=None, alias="X-Engine-Admin-Token")) -> None:
    if not x_engine_admin_token or not hmac.compare_digest(x_engine_admin_token, SETTINGS.admin_token):
        raise HTTPException(status_code=401, detail="Invalid admin token.")
