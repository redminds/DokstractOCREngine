from __future__ import annotations

import json as json_module
import logging
from typing import Any, Optional

import httpx

from app.core.config import SETTINGS


logger = logging.getLogger("dokstract.ocr_engine.platform_reporting")


class PlatformReportingError(Exception):
    pass


class PlatformReportingUnavailable(PlatformReportingError):
    pass


class PlatformReportingHTTPError(PlatformReportingError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class PlatformExecutionReportingClient:
    def __init__(self) -> None:
        self.base_url = SETTINGS.platform_api_url.strip().rstrip("/")
        self.token = SETTINGS.platform_api_ocr_engine_token.strip()
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(SETTINGS.platform_reporting_request_timeout_seconds))

    async def aclose(self) -> None:
        await self._client.aclose()

    def is_configured(self) -> bool:
        return bool(self.base_url and self.token)

    async def report_ocr_execution(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url:
            raise PlatformReportingUnavailable("PLATFORM_API_URL is not configured.")
        if not self.token:
            raise PlatformReportingUnavailable("PLATFORM_API_OCR_ENGINE_TOKEN is not configured.")

        headers = {
            "X-Service-Name": "ocr-engine",
            "X-Service-Token": self.token,
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/api/v1/internal/ocr-engine/events"
        try:
            response = await self._client.post(url, json=payload, headers=headers)
        except httpx.RequestError as exc:
            raise PlatformReportingUnavailable(str(exc)) from exc

        if response.status_code >= 400:
            raise PlatformReportingHTTPError(response.status_code, _parse_detail(response))
        try:
            data = response.json()
        except ValueError as exc:
            raise PlatformReportingError(f"Platform API returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise PlatformReportingError("Platform API returned an invalid response payload.")
        return data


def _parse_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except Exception:
        data = response.text

    if isinstance(data, dict):
        detail = data.get("detail") or data.get("message") or data
        return str(detail)
    return str(data)
