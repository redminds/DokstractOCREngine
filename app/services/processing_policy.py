"""Stable, non-secret fingerprint of the effective OCR processing policy."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.core.config import SETTINGS


def effective_processing_policy(profile: str) -> dict[str, Any]:
    """Return every configured policy input that can affect OCR output.

    Values are deliberately limited to configuration and algorithm/version
    identifiers. No paths, credentials, document contents, or OCR text are
    included.
    """
    if profile not in {"standard_auto", "enhanced_scan_recovery"}:
        raise ValueError(f"Unsupported effective processing profile: {profile}")
    return {
        "policy_schema": 1,
        "profile": profile,
        "preprocessing": {
            "algorithm": "document_standard-v1",
            "pipeline_version": SETTINGS.ocr_image_processing_pipeline_version,
            "enabled": profile == "enhanced_scan_recovery",
            "clahe": {"clip_limit": 2.0, "tile_grid": [8, 8]},
            "denoise": {"algorithm": "fastNlMeansDenoising", "h": 10, "template_window": 7, "search_window": 21},
            "deskew": {
                "enabled": SETTINGS.ocr_deskew_enabled,
                "min_confidence": SETTINGS.ocr_deskew_min_confidence,
                "max_angle_degrees": SETTINGS.ocr_deskew_max_angle_degrees,
            },
        },
        "digital_pdf": {
            "enabled": SETTINGS.ocr_digital_pdf_fast_path_enabled,
            "min_characters": SETTINGS.ocr_digital_pdf_min_characters,
            "minimum_printable_ratio": 0.7,
            "extraction_printable_ratio": 0.85,
            "algorithm": "digital-pdf-v1",
        },
        "blank_page": {
            "enabled": SETTINGS.ocr_blank_page_detection_enabled,
            "white_ratio_threshold": SETTINGS.ocr_blank_page_white_ratio_threshold,
            "dark_pixel_limit": 500,
            "edge_density_limit": 0.002,
            "variance_limit": 50,
            "algorithm": "blank-page-v1",
        },
        "rendering": {
            "default_dpi": SETTINGS.ocr_render_default_dpi,
            "pdf_scale": SETTINGS.ocr_pdf_render_scale,
            "low_content_scale": SETTINGS.ocr_low_content_render_scale,
            "max_dpi": SETTINGS.ocr_max_render_dpi,
            "det_limit_side_len": SETTINGS.ocr_det_limit_side_len,
            "max_image_width": SETTINGS.ocr_max_image_width,
            "max_image_height": SETTINGS.ocr_max_image_height,
            "max_image_pixels": SETTINGS.ocr_max_image_pixels,
            "max_total_rendered_pixels": SETTINGS.ocr_max_total_rendered_pixels_per_request,
            "max_page_aspect_ratio": SETTINGS.ocr_max_page_aspect_ratio,
            "stitched_detection_enabled": SETTINGS.ocr_stitched_page_detection_enabled,
            "stitched_aspect_signal_threshold": SETTINGS.ocr_stitched_page_aspect_signal_threshold,
            "stitched_confidence_threshold": SETTINGS.ocr_stitched_page_confidence_threshold,
            "page_classification": "adaptive-page-plan-v1",
            "tile_algorithm": "vertical-overlap-v1",
        },
        "recovery": {
            "policy_version": SETTINGS.ocr_recovery_policy_version,
            "enabled": SETTINGS.ocr_recovery_enabled,
            "min_long_side": SETTINGS.ocr_recovery_min_long_side,
            "max_items_per_megapixel": SETTINGS.ocr_recovery_max_items_per_megapixel,
            "det_limit_side_len": SETTINGS.ocr_recovery_det_limit_side_len,
            "det_db_thresh": SETTINGS.ocr_recovery_det_db_thresh,
            "det_db_box_thresh": SETTINGS.ocr_recovery_det_db_box_thresh,
            "algorithm": "sparse-recovery-v1",
        },
        "targeted_fallback": {
            "algorithm": "quality-gated-fallback-v1",
            "confidence_threshold": 0.55,
            "text_length_threshold": 24,
            "min_score_improvement": 0.05,
        },
        "model": {
            "language": SETTINGS.ocr_lang,
            "release": SETTINGS.default_release_tag,
            "cpu_threads": SETTINGS.ocr_cpu_threads,
            "mkldnn": SETTINGS.ocr_enable_mkldnn,
            "text_batch_size": SETTINGS.ocr_text_batch_size,
        },
    }


def effective_processing_policy_fingerprint(profile: str) -> str:
    policy = effective_processing_policy(profile)
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
