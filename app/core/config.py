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
    ocr_enable_mkldnn: bool = os.getenv("OCR_ENABLE_MKLDNN", "false").strip().lower() == "true"
    ocr_det_limit_side_len: int = int(os.getenv("OCR_DET_LIMIT_SIDE_LEN", "1536"))
    ocr_text_batch_size: int = int(os.getenv("OCR_TEXT_BATCH_SIZE", "2"))
    ocr_max_render_dpi: int = int(os.getenv("OCR_MAX_RENDER_DPI", "180"))
    ocr_pdf_render_scale: float = float(os.getenv("OCR_PDF_RENDER_SCALE", "1.4"))
    ocr_low_content_render_scale: float = float(os.getenv("OCR_LOW_CONTENT_RENDER_SCALE", "1.2"))
    ocr_max_image_width: int = int(os.getenv("OCR_MAX_IMAGE_WIDTH", "6000"))
    ocr_max_image_height: int = int(os.getenv("OCR_MAX_IMAGE_HEIGHT", "6000"))
    ocr_max_image_pixels: int = int(os.getenv("OCR_MAX_IMAGE_PIXELS", str(24_000_000)))
    ocr_max_decoded_image_bytes: int = int(os.getenv("OCR_MAX_DECODED_IMAGE_BYTES", str(128 * 1024 * 1024)))
    ocr_max_upload_bytes: int = int(os.getenv("OCR_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
    ocr_max_pdf_pages_per_file: int = int(os.getenv("OCR_MAX_PDF_PAGES_PER_FILE", "100"))
    ocr_max_pdf_pages_per_request: int = int(os.getenv("OCR_MAX_PDF_PAGES_PER_REQUEST", "15"))
    # Image/page dimension limits
    ocr_max_image_width_pixels: int = int(os.getenv("OCR_MAX_IMAGE_WIDTH_PIXELS", "12000"))
    ocr_max_image_height_pixels: int = int(os.getenv("OCR_MAX_IMAGE_HEIGHT_PIXELS", "12000"))
    ocr_max_image_total_pixels: int = int(os.getenv("OCR_MAX_IMAGE_TOTAL_PIXELS", str(80_000_000)))
    ocr_max_total_rendered_pixels_per_request: int = int(os.getenv("OCR_MAX_TOTAL_RENDERED_PIXELS_PER_REQUEST", str(300_000_000)))
    ocr_max_page_aspect_ratio: float = float(os.getenv("OCR_MAX_PAGE_ASPECT_RATIO", "15.0"))
    # Stitched-page detection (page-level, does NOT reject wholesale)
    ocr_stitched_page_detection_enabled: bool = os.getenv("OCR_STITCHED_PAGE_DETECTION_ENABLED", "true").strip().lower() == "true"
    ocr_stitched_page_aspect_signal_threshold: float = float(os.getenv("OCR_STITCHED_PAGE_ASPECT_SIGNAL_THRESHOLD", "4.0"))
    ocr_stitched_page_confidence_threshold: float = float(os.getenv("OCR_STITCHED_PAGE_CONFIDENCE_THRESHOLD", "0.5"))
    digital_pdf_text_threshold: int = int(os.getenv("OCR_DIGITAL_TEXT_THRESHOLD", "200"))
    engine_max_inflight_requests: int = int(os.getenv("ENGINE_MAX_INFLIGHT_REQUESTS", "2"))
    ocr_max_concurrency: int = int(os.getenv("OCR_MAX_CONCURRENCY", "2"))

    # ── Temporary input lifecycle ────────────────────────────────────────
    ocr_temp_root: str = os.getenv("OCR_TEMP_ROOT", "/tmp/dokstract-ocr").strip()
    ocr_temp_cleanup_enabled: bool = os.getenv("OCR_TEMP_CLEANUP_ENABLED", "true").strip().lower() == "true"
    ocr_temp_stale_after_seconds: int = int(os.getenv("OCR_TEMP_STALE_AFTER_SECONDS", "3600"))

    # ── OCR result cache ─────────────────────────────────────────────────
    ocr_result_cache_enabled: bool = os.getenv("OCR_RESULT_CACHE_ENABLED", "true").strip().lower() == "true"
    ocr_result_cache_required: bool = os.getenv("OCR_RESULT_CACHE_REQUIRED", "true").strip().lower() == "true"
    ocr_result_cache_ttl_seconds: int = int(os.getenv("OCR_RESULT_CACHE_TTL_SECONDS", "86400"))
    ocr_result_cache_key_prefix: str = os.getenv("OCR_RESULT_CACHE_KEY_PREFIX", "dokstract:ocr:result").strip()
    ocr_result_cache_dir: str = os.getenv("OCR_RESULT_CACHE_DIR", "/app/.data/ocr-cache").strip()

    # ── Image processing ─────────────────────────────────────────────────
    ocr_image_processing_default_enabled: bool = os.getenv("OCR_IMAGE_PROCESSING_DEFAULT_ENABLED", "false").strip().lower() == "true"
    ocr_image_processing_default_profile: str = os.getenv("OCR_IMAGE_PROCESSING_DEFAULT_PROFILE", "none").strip()
    ocr_image_processing_allowed_profiles: tuple[str, ...] = _csv_env("OCR_IMAGE_PROCESSING_ALLOWED_PROFILES", "none,document_standard")
    ocr_image_processing_pipeline_version: str = os.getenv("OCR_IMAGE_PROCESSING_PIPELINE_VERSION", "1").strip()

    # ── Deskew ───────────────────────────────────────────────────────────
    ocr_deskew_enabled: bool = os.getenv("OCR_DESKEW_ENABLED", "true").strip().lower() == "true"
    ocr_deskew_min_confidence: float = float(os.getenv("OCR_DESKEW_MIN_CONFIDENCE", "0.85"))
    ocr_deskew_max_angle_degrees: float = float(os.getenv("OCR_DESKEW_MAX_ANGLE_DEGREES", "10"))

    # ── Request guard ────────────────────────────────────────────────────
    ocr_max_queued_requests: int = int(os.getenv("OCR_MAX_QUEUED_REQUESTS", "50"))
    ocr_queue_wait_timeout_seconds: int = int(os.getenv("OCR_QUEUE_WAIT_TIMEOUT_SECONDS", "300"))

    # ── Render ───────────────────────────────────────────────────────────
    ocr_render_default_dpi: int = int(os.getenv("OCR_RENDER_DEFAULT_DPI", "150"))
    ocr_render_default_scale: float = float(os.getenv("OCR_RENDER_DEFAULT_SCALE", "1.5"))

    # ── Digital PDF ──────────────────────────────────────────────────────
    ocr_digital_pdf_fast_path_enabled: bool = os.getenv("OCR_DIGITAL_PDF_FAST_PATH_ENABLED", "true").strip().lower() == "true"
    ocr_digital_pdf_min_characters: int = int(os.getenv("OCR_DIGITAL_PDF_MIN_CHARACTERS", "50"))

    # ── Blank page detection ─────────────────────────────────────────────
    ocr_blank_page_detection_enabled: bool = os.getenv("OCR_BLANK_PAGE_DETECTION_ENABLED", "true").strip().lower() == "true"
    ocr_blank_page_white_ratio_threshold: float = float(os.getenv("OCR_BLANK_PAGE_WHITE_RATIO_THRESHOLD", "0.995"))

    # ── Logging ──────────────────────────────────────────────────────────
    ocr_log_page_summaries: bool = os.getenv("OCR_LOG_PAGE_SUMMARIES", "true").strip().lower() == "true"
    ocr_log_ocr_text: bool = os.getenv("OCR_LOG_OCR_TEXT", "false").strip().lower() == "true"

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
    if SETTINGS.ocr_max_image_width_pixels <= 0:
        raise RuntimeError("OCR_MAX_IMAGE_WIDTH_PIXELS must be positive.")
    if SETTINGS.ocr_max_image_height_pixels <= 0:
        raise RuntimeError("OCR_MAX_IMAGE_HEIGHT_PIXELS must be positive.")
    if SETTINGS.ocr_max_image_total_pixels <= 0:
        raise RuntimeError("OCR_MAX_IMAGE_TOTAL_PIXELS must be positive.")
    if SETTINGS.ocr_max_total_rendered_pixels_per_request <= 0:
        raise RuntimeError("OCR_MAX_TOTAL_RENDERED_PIXELS_PER_REQUEST must be positive.")
    if SETTINGS.ocr_max_page_aspect_ratio < 1.0:
        raise RuntimeError("OCR_MAX_PAGE_ASPECT_RATIO must be >= 1.0.")
    if SETTINGS.ocr_stitched_page_aspect_signal_threshold < 1.0:
        raise RuntimeError("OCR_STITCHED_PAGE_ASPECT_SIGNAL_THRESHOLD must be >= 1.0.")
    if SETTINGS.ocr_stitched_page_aspect_signal_threshold > SETTINGS.ocr_max_page_aspect_ratio:
        raise RuntimeError("OCR_STITCHED_PAGE_ASPECT_SIGNAL_THRESHOLD must not exceed OCR_MAX_PAGE_ASPECT_RATIO.")
    if not (0.0 <= SETTINGS.ocr_stitched_page_confidence_threshold <= 1.0):
        raise RuntimeError("OCR_STITCHED_PAGE_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0.")
    if SETTINGS.ocr_max_pdf_pages_per_request > SETTINGS.ocr_max_pdf_pages_per_file:
        raise RuntimeError("OCR_MAX_PDF_PAGES_PER_REQUEST must not exceed OCR_MAX_PDF_PAGES_PER_FILE.")
    # Temp lifecycle
    if SETTINGS.ocr_temp_stale_after_seconds <= 0:
        raise RuntimeError("OCR_TEMP_STALE_AFTER_SECONDS must be positive.")
    # Cache
    if SETTINGS.ocr_result_cache_required and not SETTINGS.ocr_result_cache_enabled:
        raise RuntimeError("OCR_RESULT_CACHE_REQUIRED=true requires OCR_RESULT_CACHE_ENABLED=true.")
    # Render scale
    if SETTINGS.ocr_pdf_render_scale <= 0:
        raise RuntimeError("OCR_PDF_RENDER_SCALE must be positive.")
    if round(SETTINGS.ocr_pdf_render_scale * 72) > SETTINGS.ocr_max_render_dpi:
        raise RuntimeError(
            f"OCR_PDF_RENDER_SCALE ({SETTINGS.ocr_pdf_render_scale}) yields "
            f"{round(SETTINGS.ocr_pdf_render_scale * 72)} DPI, which exceeds "
            f"OCR_MAX_RENDER_DPI ({SETTINGS.ocr_max_render_dpi})."
        )
    # Low-content render scale
    if SETTINGS.ocr_low_content_render_scale <= 0:
        raise RuntimeError("OCR_LOW_CONTENT_RENDER_SCALE must be positive.")
    if SETTINGS.ocr_low_content_render_scale > SETTINGS.ocr_pdf_render_scale:
        raise RuntimeError(
            f"OCR_LOW_CONTENT_RENDER_SCALE ({SETTINGS.ocr_low_content_render_scale}) "
            f"must not exceed OCR_PDF_RENDER_SCALE ({SETTINGS.ocr_pdf_render_scale})."
        )
    if round(SETTINGS.ocr_low_content_render_scale * 72) > SETTINGS.ocr_max_render_dpi:
        raise RuntimeError(
            f"OCR_LOW_CONTENT_RENDER_SCALE ({SETTINGS.ocr_low_content_render_scale}) yields "
            f"{round(SETTINGS.ocr_low_content_render_scale * 72)} DPI, which exceeds "
            f"OCR_MAX_RENDER_DPI ({SETTINGS.ocr_max_render_dpi})."
        )
    if SETTINGS.ocr_result_cache_ttl_seconds <= 0:
        raise RuntimeError("OCR_RESULT_CACHE_TTL_SECONDS must be positive.")
    # Image processing
    if SETTINGS.ocr_image_processing_default_profile not in SETTINGS.ocr_image_processing_allowed_profiles:
        raise RuntimeError(f"OCR_IMAGE_PROCESSING_DEFAULT_PROFILE '{SETTINGS.ocr_image_processing_default_profile}' not in allowed profiles.")
    if SETTINGS.ocr_image_processing_allowed_profiles and "none" not in SETTINGS.ocr_image_processing_allowed_profiles:
        raise RuntimeError("OCR_IMAGE_PROCESSING_ALLOWED_PROFILES must include 'none'.")
    # Deskew
    if not (0.0 <= SETTINGS.ocr_deskew_min_confidence <= 1.0):
        raise RuntimeError("OCR_DESKEW_MIN_CONFIDENCE must be between 0.0 and 1.0.")
    if SETTINGS.ocr_deskew_max_angle_degrees <= 0:
        raise RuntimeError("OCR_DESKEW_MAX_ANGLE_DEGREES must be positive.")
