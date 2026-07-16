from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from app.core.config import SETTINGS


def require_internal_token(x_engine_internal_token: str | None = Header(default=None)) -> None:
    if not x_engine_internal_token or not hmac.compare_digest(x_engine_internal_token, SETTINGS.internal_token):
        raise HTTPException(status_code=401, detail="Invalid internal token.")


def require_admin_token(x_engine_admin_token: str | None = Header(default=None)) -> None:
    if not x_engine_admin_token or not hmac.compare_digest(x_engine_admin_token, SETTINGS.admin_token):
        raise HTTPException(status_code=401, detail="Invalid admin token.")
