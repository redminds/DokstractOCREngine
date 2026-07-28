"""Configurable image processing profiles for OCR preprocessing.

Supported profiles:
  - none: No enhancement, use original image
  - document_standard: Conservative grayscale + contrast normalization + light denoise + optional deskew
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.core.config import SETTINGS

logger = logging.getLogger("dokstract.ocr_engine.image_processing")


@dataclass
class ProcessingResult:
    """Result of applying an image processing profile."""

    image: np.ndarray
    profile: str
    operations_applied: list[str] = field(default_factory=list)
    operations_skipped: list[dict[str, str]] = field(default_factory=list)
    duration_ms: float = 0.0


def resolve_profile(enabled: bool, profile_name: str | None) -> str:
    """Resolve the effective processing profile from request parameters."""
    if not enabled:
        return "none"
    name = (profile_name or SETTINGS.ocr_image_processing_default_profile).strip().lower()
    if name not in SETTINGS.ocr_image_processing_allowed_profiles:
        raise ValueError(f"Unsupported image processing profile: {name}")
    return name


def apply_profile(image: np.ndarray, profile_name: str) -> ProcessingResult:
    """Apply a named processing profile to an image."""
    t0 = time.perf_counter()

    if profile_name == "none":
        return ProcessingResult(
            image=image,
            profile="none",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
        )

    if profile_name == "document_standard":
        return _apply_document_standard(image, t0)

    raise ValueError(f"Unknown profile: {profile_name}")


def _apply_document_standard(image: np.ndarray, t0: float) -> ProcessingResult:
    """Conservative document enhancement pipeline."""
    result = image
    applied: list[str] = []
    skipped: list[dict[str, str]] = []

    # 1. Grayscale conversion
    if len(result.shape) == 3 and result.shape[2] >= 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        applied.append("grayscale")
    else:
        applied.append("grayscale")  # already grayscale

    # 2. Contrast normalization (CLAHE)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(result)
        applied.append("contrast_normalization")
    except Exception:
        skipped.append({"operation": "contrast_normalization", "reason": "clahe_failed"})

    # 3. Light denoising
    try:
        result = cv2.fastNlMeansDenoising(result, None, h=10, templateWindowSize=7, searchWindowSize=21)
        applied.append("light_denoise")
    except Exception:
        skipped.append({"operation": "light_denoise", "reason": "denoise_failed"})

    # 4. Optional deskew
    if SETTINGS.ocr_deskew_enabled:
        deskewed, angle, conf = _try_deskew(result)
        if conf >= SETTINGS.ocr_deskew_min_confidence and abs(angle) > 0.5:
            result = deskewed
            applied.append("deskew")
        else:
            skipped.append({
                "operation": "deskew",
                "reason": "confidence_below_threshold" if conf < SETTINGS.ocr_deskew_min_confidence else "angle_too_small",
            })

    # Convert back to BGR for PaddleOCR if needed
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return ProcessingResult(
        image=result,
        profile="document_standard",
        operations_applied=applied,
        operations_skipped=skipped,
        duration_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _try_deskew(gray: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Attempt to deskew a grayscale image. Returns (image, angle_degrees, confidence)."""
    try:
        # Binarize for line detection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(binary > 0))

        if len(coords) < 100:
            return gray, 0.0, 0.0

        # Find minimum area rectangle
        rect = cv2.minAreaRect(coords.astype(np.float32))
        angle = rect[2]

        # Normalize angle
        if angle < -45:
            angle = 90 + angle
        angle = max(-SETTINGS.ocr_deskew_max_angle_degrees, min(SETTINGS.ocr_deskew_max_angle_degrees, angle))

        if abs(angle) < 0.3:
            return gray, 0.0, 0.0

        # Confidence based on how much rotation is needed
        confidence = min(1.0, abs(angle) / 5.0)

        # Rotate
        h, w = gray.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return rotated, angle, confidence
    except Exception:
        return gray, 0.0, 0.0


def inspect_image_metadata(data: bytes) -> dict[str, Any]:
    """Inspect image metadata for pre-decode safety checks using Pillow."""
    try:
        from PIL import Image, ImageFile
        import io

        # Configure decompression bomb protection
        Image.MAX_IMAGE_PIXELS = SETTINGS.ocr_max_image_total_pixels

        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            fmt = img.format or "unknown"

            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions: {width}x{height}")
            if width > SETTINGS.ocr_max_image_width_pixels:
                raise ValueError(f"Image width {width} exceeds maximum {SETTINGS.ocr_max_image_width_pixels}")
            if height > SETTINGS.ocr_max_image_height_pixels:
                raise ValueError(f"Image height {height} exceeds maximum {SETTINGS.ocr_max_image_height_pixels}")
            pixels = width * height
            if pixels > SETTINGS.ocr_max_image_total_pixels:
                raise ValueError(f"Image has {pixels} pixels, exceeding maximum {SETTINGS.ocr_max_image_total_pixels}")
            aspect = max(width / max(height, 1), height / max(width, 1))
            if aspect > SETTINGS.ocr_max_page_aspect_ratio:
                raise ValueError(f"Image aspect ratio {aspect:.2f} exceeds maximum {SETTINGS.ocr_max_page_aspect_ratio}")

            return {"width": width, "height": height, "format": fmt, "pixels": pixels}
    except ImportError:
        logger.warning("Pillow not available for pre-decode metadata inspection")
        return {}
    except Exception as exc:
        raise ValueError(f"Image metadata inspection failed: {exc}") from exc
