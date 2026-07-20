from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _csv_env(name: str, default: str = "") -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class EngineSettings:
    app_env: str = os.getenv("APP_ENV", "development").strip().lower()
    service_name: str = os.getenv("ENGINE_SERVICE_NAME", "dokstract-ocr-engine").strip()
    host: str = os.getenv("ENGINE_HOST", "0.0.0.0").strip()
    port: int = int(os.getenv("ENGINE_PORT", "8010"))
    ocr_api_token: str = os.getenv("OCR_ENGINE_OCR_API_TOKEN", "change-me-ocr-api-to-engine").strip()
    schema_api_token: str = os.getenv("OCR_ENGINE_SCHEMA_API_TOKEN", "change-me-schema-api-to-engine").strip()
    admin_token: str = os.getenv("ENGINE_ADMIN_TOKEN", "change-me-admin").strip()
    registry_db_path: str = os.getenv("ENGINE_REGISTRY_DB_PATH", "workspace_tmp/ocr-engine-registry.db").strip()
    default_release_tag: str = os.getenv("ENGINE_DEFAULT_RELEASE_TAG", "ocr-engine-2026.07.15").strip()
    default_image_digest: str = os.getenv("ENGINE_DEFAULT_IMAGE_DIGEST", "sha256:dev-placeholder").strip()
    default_supported_api_versions: tuple[str, ...] = _csv_env("ENGINE_DEFAULT_SUPPORTED_API_VERSIONS", "v1")
    default_capabilities: tuple[str, ...] = _csv_env("ENGINE_DEFAULT_CAPABILITIES", "ocr,pdf")
    allow_origins: tuple[str, ...] = _csv_env("ENGINE_CORS_ORIGINS", "*")
    ocr_lang: str = os.getenv("OCR_LANG", "en").strip()
    ocr_cpu_threads: int = int(os.getenv("OCR_CPU_THREADS", "4"))
    ocr_det_limit_side_len: int = int(os.getenv("OCR_DET_LIMIT_SIDE_LEN", "1536"))
    ocr_text_batch_size: int = int(os.getenv("OCR_TEXT_BATCH_SIZE", "2"))
    ocr_max_render_dpi: int = int(os.getenv("OCR_MAX_RENDER_DPI", "180"))
    ocr_max_image_width: int = int(os.getenv("OCR_MAX_IMAGE_WIDTH", "6000"))
    ocr_max_image_height: int = int(os.getenv("OCR_MAX_IMAGE_HEIGHT", "6000"))
    ocr_max_image_pixels: int = int(os.getenv("OCR_MAX_IMAGE_PIXELS", str(24_000_000)))
    ocr_max_decoded_image_bytes: int = int(os.getenv("OCR_MAX_DECODED_IMAGE_BYTES", str(128 * 1024 * 1024)))
    ocr_max_upload_bytes: int = int(os.getenv("OCR_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
    ocr_max_pdf_pages_per_file: int = int(os.getenv("OCR_MAX_PDF_PAGES_PER_FILE", "100"))
    ocr_max_pdf_pages_per_request: int = int(os.getenv("OCR_MAX_PDF_PAGES_PER_REQUEST", "15"))
    digital_pdf_text_threshold: int = int(os.getenv("OCR_DIGITAL_TEXT_THRESHOLD", "200"))
    engine_max_inflight_requests: int = int(os.getenv("ENGINE_MAX_INFLIGHT_REQUESTS", "2"))
    ocr_max_concurrency: int = int(os.getenv("OCR_MAX_CONCURRENCY", "2"))

    @property
    def registry_path(self) -> Path:
        return Path(self.registry_db_path)


SETTINGS = EngineSettings()


def validate_startup_configuration() -> None:
    if SETTINGS.app_env == "production":
        if SETTINGS.ocr_api_token in {"change-me-ocr-api-to-engine", "change-me", "changeme"}:
            raise RuntimeError("OCR_ENGINE_OCR_API_TOKEN must be configured in production.")
        if SETTINGS.schema_api_token in {"change-me-schema-api-to-engine", "change-me", "changeme"}:
            raise RuntimeError("OCR_ENGINE_SCHEMA_API_TOKEN must be configured in production.")
        if SETTINGS.admin_token in {"change-me-admin", "change-me", "changeme"}:
            raise RuntimeError("ENGINE_ADMIN_TOKEN must be configured in production.")
