from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np

from app.core.config import SETTINGS
from app.services.image_processing import (
    ProcessingResult,
    apply_profile,
    inspect_image_metadata,
    resolve_profile,
)
from app.services.ocr.models import OCRPage, OCRItem, OCRLine, OCRBlock, BBox
from app.services.ocr.paddle_adapter import adapt_paddle_result
from app.services.ocr.line_reconstruction import reconstruct_lines
from app.services.ocr.block_reconstruction import reconstruct_blocks
from app.services.ocr.reading_order import (
    assign_reading_order_items,
    assign_reading_order_lines,
    assign_reading_order_blocks,
)
from app.services.ocr.response_builder import build_response
from app.services.ocr.blank_detection import is_blank_page
from app.services.ocr.low_content_detection import is_low_content_page
from app.services.ocr.digital_pdf import extract_digital_page, _has_usable_digital_text
from app.services.ocr_cache import (
    CachedOCRResult,
    compute_file_hash,
    compute_request_fingerprint,
    compute_page_fingerprint,
    get_cached_result,
    put_cached_result,
    get_cached_page,
    put_cached_page,
    sweep_stale_temp,
)

logger = logging.getLogger("dokstract.ocr_engine.execution")


@dataclass
class _RawOCRResult:
    """Internal container for raw PaddleOCR output + image dimensions."""

    raw_result: list[list[Any]] | None
    image_width: float
    image_height: float


class OCRDependencyUnavailable(RuntimeError):
    pass


_OCR_ENGINE = None


def _get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        init_start = time.perf_counter()
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # pragma: no cover - environment-specific
            raise OCRDependencyUnavailable(f"OCR engine unavailable: {exc}") from exc

        try:
            _OCR_ENGINE = PaddleOCR(
                lang=SETTINGS.ocr_lang,
                use_angle_cls=False,
                det_limit_side_len=SETTINGS.ocr_det_limit_side_len,
                det_limit_type="max",
                use_dilation=True,
                text_recognition_batch_size=SETTINGS.ocr_text_batch_size,
                cpu_threads=SETTINGS.ocr_cpu_threads,
                enable_mkldnn=SETTINGS.ocr_enable_mkldnn,
            )
        except Exception as exc:  # pragma: no cover - environment-specific
            raise OCRDependencyUnavailable(f"OCR engine unavailable: {exc}") from exc
        init_elapsed = time.perf_counter() - init_start
        logger.info("PaddleOCR model initialized in %.2fs (lang=%s cpu_threads=%d mkldnn=%s)",
                     init_elapsed, SETTINGS.ocr_lang, SETTINGS.ocr_cpu_threads, SETTINGS.ocr_enable_mkldnn)
    return _OCR_ENGINE


def normalize_text(lines: list[str]) -> str:
    seen: set[str] = set()
    clean: list[str] = []

    for line in lines:
        value = line.strip()
        if not value or len(value) < 2:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append(value)

    merged: list[str] = []
    buffer = ""
    for line in clean:
        if not buffer:
            buffer = line
            continue
        if line[0].islower() or len(buffer.split()) < 3:
            buffer += " " + line
        else:
            merged.append(buffer)
            buffer = line

    if buffer:
        merged.append(buffer)

    return "\n".join(merged).strip()


def run_ocr(image: np.ndarray, enhance: bool) -> _RawOCRResult:
    """Run PaddleOCR and return raw result with image dimensions.

    The raw result is passed to the geometry pipeline for structured processing.
    """
    if enhance:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    image_height, image_width = image.shape[:2]
    result = _get_ocr_engine().ocr(image, cls=False)

    return _RawOCRResult(
        raw_result=result,
        image_width=float(image_width),
        image_height=float(image_height),
    )


def _build_structured_page(
    raw_result: _RawOCRResult,
    page_number: int,
) -> tuple[OCRPage, dict[str, float]]:
    """Run the full geometry-aware pipeline on a single page.

    Pipeline:
      PaddleOCR raw result
          → paddle_adapter → OCRItem list
          → line_reconstruction → OCRLine list
          → block_reconstruction → OCRBlock list
          → reading_order assignment
          → OCRPage

    Returns:
        Tuple of (OCRPage, phase_timings_ms_dict).
    """
    timings: dict[str, float] = {}

    # Step 1: Adapt raw PaddleOCR result to internal OCRItem models
    t0 = time.perf_counter()
    items = adapt_paddle_result(
        raw_result.raw_result,
        page_number=page_number,
        page_width=raw_result.image_width,
        page_height=raw_result.image_height,
    )
    timings["paddle_adaptation_ms"] = (time.perf_counter() - t0) * 1000.0

    if not items:
        # Empty page
        page = OCRPage(
            page_number=page_number,
            width=raw_result.image_width,
            height=raw_result.image_height,
            duration_ms=0.0,
        )
        timings["line_reconstruction_ms"] = 0.0
        timings["block_reconstruction_ms"] = 0.0
        timings["reading_order_ms"] = 0.0
        timings["geometry_total_ms"] = timings["paddle_adaptation_ms"]
        return page, timings

    # Step 2: Reconstruct lines from items (geometry-based grouping)
    t0 = time.perf_counter()
    lines = reconstruct_lines(
        items,
        page_width=raw_result.image_width,
        page_height=raw_result.image_height,
    )
    timings["line_reconstruction_ms"] = (time.perf_counter() - t0) * 1000.0

    # Step 3: Reconstruct blocks from lines
    t0 = time.perf_counter()
    blocks = reconstruct_blocks(
        lines,
        page_width=raw_result.image_width,
        page_height=raw_result.image_height,
    )
    timings["block_reconstruction_ms"] = (time.perf_counter() - t0) * 1000.0

    # Compute normalized_bbox for lines and blocks
    from app.services.ocr.geometry import normalize_bbox as _norm
    for line in lines:
        line.normalized_bbox = _norm(line.bbox, raw_result.image_width, raw_result.image_height)
    for block in blocks:
        block.normalized_bbox = _norm(block.bbox, raw_result.image_width, raw_result.image_height)
        block_lines = [l for l in lines if l.line_id in block.line_ids]
        block_confidences = [l.confidence for l in block_lines if l.confidence > 0.0]
        block.confidence = (
            sum(block_confidences) / len(block_confidences) if block_confidences else 0.0
        )

    # Step 4: Assign reading order
    t0 = time.perf_counter()
    assign_reading_order_items(items)
    assign_reading_order_lines(lines)
    assign_reading_order_blocks(blocks)
    timings["reading_order_ms"] = (time.perf_counter() - t0) * 1000.0

    # Geometry total = sum of sub-phases
    timings["geometry_total_ms"] = (
        timings["paddle_adaptation_ms"]
        + timings["line_reconstruction_ms"]
        + timings["block_reconstruction_ms"]
        + timings["reading_order_ms"]
    )

    # Step 5: Calculate page-level aggregates
    page_text = "\n".join(line.text for line in lines if line.text.strip())
    page_confidences = [item.confidence for item in items if item.confidence > 0.0]
    page_confidence = (
        sum(page_confidences) / len(page_confidences) if page_confidences else 0.0
    )

    # Back-link block_id to items via lines
    for line in lines:
        if line.block_id:
            for item in items:
                if item.line_id == line.line_id:
                    item.block_id = line.block_id

    page = OCRPage(
        page_number=page_number,
        width=raw_result.image_width,
        height=raw_result.image_height,
        items=items,
        lines=lines,
        blocks=blocks,
        text=page_text,
        confidence=page_confidence,
        duration_ms=timings["geometry_total_ms"],
    )

    return page, timings


def _validate_image_dimensions(width: int, height: int) -> None:
    """Validate image dimensions against configured limits. Raises ValueError."""
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image dimensions: {width}x{height}. "
            f"Image must have positive width and height."
        )
    if width > SETTINGS.ocr_max_image_width_pixels:
        raise ValueError(
            f"Image width {width} exceeds maximum {SETTINGS.ocr_max_image_width_pixels}."
        )
    if height > SETTINGS.ocr_max_image_height_pixels:
        raise ValueError(
            f"Image height {height} exceeds maximum {SETTINGS.ocr_max_image_height_pixels}."
        )
    pixels = width * height
    if pixels > SETTINGS.ocr_max_image_total_pixels:
        raise ValueError(
            f"Image has {pixels} pixels, exceeding maximum {SETTINGS.ocr_max_image_total_pixels}."
        )
    aspect = max(width / max(height, 1), height / max(width, 1))
    if aspect > SETTINGS.ocr_max_page_aspect_ratio:
        raise ValueError(
            f"Image aspect ratio {aspect:.2f} exceeds maximum {SETTINGS.ocr_max_page_aspect_ratio}."
        )


def _validate_render_budget(cumulative_pixels: int, page_number: int) -> None:
    """Check cumulative rendered pixels against request budget."""
    limit = SETTINGS.ocr_max_total_rendered_pixels_per_request
    if cumulative_pixels > limit:
        raise ValueError(
            f"Request pixel budget exceeded: cumulative {cumulative_pixels} pixels "
            f"at page {page_number}, maximum {limit}."
        )


def _check_stitched_page(image: np.ndarray) -> str | None:
    """Check for stitched pages using page-level classification.

    Returns:
        None if the page is normal or can be handled adaptively.
        A warning string if the page has extreme characteristics.
        Raises ValueError only when the page is UNSAFE_TO_RENDER.
    """
    if not SETTINGS.ocr_stitched_page_detection_enabled:
        return None

    from app.services.ocr.page_classification import (
        classify_page_dimensions, run_stitched_detector_on_image,
        PageSizeClassification,
    )
    height, width = image.shape[:2]
    classification = classify_page_dimensions(width, height)

    if classification.classification == PageSizeClassification.UNSAFE_TO_RENDER:
        raise ValueError(
            f"Page {width}×{height} exceeds safe rendering limits: "
            f"{'; '.join(classification.triggered_rules)}"
        )

    if classification.classification == PageSizeClassification.NORMAL:
        return None

    # Run the visual stitched detector for POSSIBLE_STITCHED pages
    if classification.classification == PageSizeClassification.POSSIBLE_STITCHED:
        is_stitched = run_stitched_detector_on_image(image)
        if not is_stitched:
            logger.info(
                "page_classified_as_possible_stitched_but_detector_cleared "
                "width=%d height=%d aspect=%.1f",
                width, height, classification.aspect_ratio,
            )
            return None
        logger.warning(
            "stitched_page_confirmed width=%d height=%d aspect=%.1f rules=%s",
            width, height, classification.aspect_ratio,
            classification.triggered_rules,
        )
        return "stitched_confirmed"

    # Large / oversized / extreme aspect — log and continue
    logger.info(
        "page_classified_as_%s width=%d height=%d aspect=%.1f recommendation=%s",
        classification.classification.value, width, height,
        classification.aspect_ratio, classification.recommendation,
    )
    return classification.classification.value


@dataclass(frozen=True)
class PdfSelection:
    page_numbers: list[int]
    total_pages: int
    defaulted_to_first_pages: bool


def open_pdf_from_bytes(pdf_bytes: bytes) -> fitz.Document:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError("Invalid or corrupted PDF") from exc

    if getattr(doc, "is_encrypted", False):
        doc.close()
        raise ValueError("Encrypted PDF files are not supported")
    if doc.page_count <= 0:
        doc.close()
        raise ValueError("PDF has no pages")
    return doc


def parse_pages(pages_str: str) -> list[int]:
    pages = set()
    parts = pages_str.split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
            except ValueError as exc:
                raise ValueError("Invalid page range format") from exc

            if start > end:
                raise ValueError("Page range start must be <= end")

            for page in range(start, end + 1):
                pages.add(page)
        else:
            try:
                pages.add(int(part))
            except ValueError as exc:
                raise ValueError("Invalid page number") from exc

    return sorted(pages)


def select_pages(total_pages: int, pages_str: str | None, max_pages: int) -> PdfSelection:
    if total_pages <= 0:
        raise ValueError("PDF has no pages")

    if pages_str and pages_str.strip():
        page_numbers = parse_pages(pages_str)
        defaulted = False
    else:
        page_numbers = list(range(1, min(total_pages, max_pages) + 1))
        defaulted = total_pages > max_pages

    if len(page_numbers) > max_pages:
        raise ValueError(f"Maximum {max_pages} pages allowed per request")

    for page in page_numbers:
        if page < 1 or page > total_pages:
            raise ValueError(f"Invalid page {page}. PDF has {total_pages} pages")

    return PdfSelection(
        page_numbers=page_numbers,
        total_pages=total_pages,
        defaulted_to_first_pages=defaulted,
    )


def enforce_pdf_page_budget(total_pages: int, selected_pages: int, max_pages_per_file: int, max_pages_per_request: int) -> None:
    if total_pages > max_pages_per_file:
        raise ValueError(f"Maximum {max_pages_per_file} pages allowed per file")
    if selected_pages > max_pages_per_request:
        raise ValueError(f"Maximum {max_pages_per_request} pages allowed per request")


def extract_pdf_text(path: str, page_numbers: list[int] | None = None) -> str:
    doc = fitz.open(path)
    texts: list[str] = []
    try:
        for index, page in enumerate(doc, start=1):
            if page_numbers and index not in page_numbers:
                continue
            text = page.get_text().strip()
            if text:
                texts.append(text)
    finally:
        doc.close()
    return "\n".join(texts)


def pdf_page_to_img(page: Any, scale: float):
    max_dpi = max(72, int(SETTINGS.ocr_max_render_dpi))
    scale = min(scale, max_dpi / 72.0)
    rect = page.rect
    width = int(rect.width * scale)
    height = int(rect.height * scale)
    if width <= 0 or height <= 0:
        raise ValueError("Invalid PDF page dimensions")
    if width > SETTINGS.ocr_max_image_width or height > SETTINGS.ocr_max_image_height:
        raise ValueError("Rendered PDF page exceeds image dimension limits")
    if width * height > SETTINGS.ocr_max_image_pixels:
        raise ValueError("Rendered PDF page exceeds pixel limits")

    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    decoded = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), 1)
    if decoded is None:
        raise ValueError("Rendered PDF page could not be decoded")
    if decoded.shape[1] > SETTINGS.ocr_max_image_width or decoded.shape[0] > SETTINGS.ocr_max_image_height:
        raise ValueError("Rendered PDF page exceeds image dimension limits")
    if decoded.shape[0] * decoded.shape[1] > SETTINGS.ocr_max_image_pixels:
        raise ValueError("Rendered PDF page exceeds pixel limits")
    if decoded.nbytes > SETTINGS.ocr_max_decoded_image_bytes:
        raise ValueError("Rendered PDF page exceeds decoded memory limits")
    return decoded


def is_digital_pdf_text(text: str, threshold: int) -> bool:
    return len(text.strip()) > threshold


def _build_results_for_image(
    image: np.ndarray, *, page: int, enhance: bool, profile_name: str = "none"
) -> tuple[OCRPage, dict[str, float], ProcessingResult | None]:
    """Run OCR with geometry pipeline on a single image.

    Uses page classification to determine adaptive strategy:
    - NORMAL → standard OCR
    - LARGE/OVERSIZED → downscale if safe, else tile
    - POSSIBLE_STITCHED/EXTREME → tile with stitched detector
    - UNSAFE → raise ValueError

    Returns:
        Tuple of (OCRPage, timing_dict, processing_result_or_None).
    """
    from app.services.ocr.page_classification import (
        classify_page_dimensions, compute_adaptive_render_plan,
        tile_image_vertical, reconstruct_page_from_tiles,
        PageSizeClassification,
    )

    height, width = image.shape[:2]
    timings: dict[str, float] = {}

    # ── Page classification ──────────────────────────────────────────
    classification = classify_page_dimensions(width, height)

    if classification.classification == PageSizeClassification.UNSAFE_TO_RENDER:
        raise ValueError(
            f"Page {page} ({width}×{height}) unsafe to render: "
            f"{'; '.join(classification.triggered_rules)}"
        )

    # ── Adaptive render plan ─────────────────────────────────────────
    plan = compute_adaptive_render_plan(width, height, classification)

    # ── Image processing ─────────────────────────────────────────────
    processing_result: ProcessingResult | None = None
    t0 = time.perf_counter()
    profile = resolve_profile(enhance, profile_name)
    processing_result = apply_profile(image, profile)
    processed_image = processing_result.image
    timings["preprocessing_ms"] = processing_result.duration_ms

    # ── Stitched check (does NOT reject) ─────────────────────────────
    stitched_warning = _check_stitched_page(processed_image)

    # ── Execute OCR ──────────────────────────────────────────────────
    if plan.strategy == "tile":
        return _build_tiled_page(
            processed_image, page=page, plan=plan,
            enhance=enhance, profile_name=profile_name,
            timings=timings, processing_result=processing_result,
        )

    # Standard or downscale: single-image OCR
    ocr_start = time.perf_counter()
    try:
        raw_result = run_ocr(processed_image, enhance=False)
    except OCRDependencyUnavailable:
        raise
    timings["ocr_inference_ms"] = (time.perf_counter() - ocr_start) * 1000.0

    # Geometry pipeline
    page_data, geom_timings = _build_structured_page(raw_result, page_number=page)
    timings.update(geom_timings)

    timings["page_total_ms"] = (
        timings["preprocessing_ms"]
        + timings["ocr_inference_ms"]
        + timings["geometry_total_ms"]
    )
    page_data.page_metrics = {
        k: round(v, 3) for k, v in timings.items()
    }
    page_data.page_metrics["classification"] = classification.classification.value
    page_data.page_metrics["render_strategy"] = plan.strategy
    if stitched_warning:
        page_data.page_metrics["stitched_warning"] = stitched_warning

    if SETTINGS.ocr_log_page_summaries:
        logger.debug(
            "Page %d: cls=%s strategy=%s ocr=%.0fms items=%d lines=%d",
            page, classification.classification.value, plan.strategy,
            timings["ocr_inference_ms"], len(page_data.items), len(page_data.lines),
        )

    return page_data, timings, processing_result


def _build_tiled_page(
    image: np.ndarray,
    page: int,
    plan,
    enhance: bool,
    profile_name: str,
    timings: dict[str, float],
    processing_result,
) -> tuple[OCRPage, dict[str, float], ProcessingResult | None]:
    """Process a page using vertical tiling with coordinate reconstruction."""
    from app.services.ocr.page_classification import (
        tile_image_vertical, reconstruct_page_from_tiles,
    )
    height, width = image.shape[:2]
    tiles = tile_image_vertical(image, plan.tile_height, plan.tile_overlap)
    tile_results: list[dict] = []
    total_ocr_ms = 0.0

    for tile_img, y_offset in tiles:
        ocr_start = time.perf_counter()
        try:
            raw = run_ocr(tile_img, enhance=False)
        except OCRDependencyUnavailable:
            raise
        total_ocr_ms += (time.perf_counter() - ocr_start) * 1000.0

        tile_page_data, _ = _build_structured_page(raw, page_number=page)
        tile_results.append({
            "items": [{
                "item_id": it.item_id, "text": it.text,
                "confidence": it.confidence,
                "bbox": {"x1": it.bbox.x1, "y1": it.bbox.y1,
                         "x2": it.bbox.x2, "y2": it.bbox.y2},
            } for it in tile_page_data.items],
            "lines": [{
                "line_id": ln.line_id, "text": ln.text,
                "confidence": ln.confidence,
                "bbox": {"x1": ln.bbox.x1, "y1": ln.bbox.y1,
                         "x2": ln.bbox.x2, "y2": ln.bbox.y2},
            } for ln in tile_page_data.lines],
            "blocks": [],
            "tile_y_offset": y_offset,
        })

    # Reconstruct canonical page
    reconstructed = reconstruct_page_from_tiles(
        tile_results, page, width, height,
    )

    timings["ocr_inference_ms"] = total_ocr_ms
    timings["tile_count"] = float(len(tiles))
    timings["geometry_total_ms"] = 0.0
    timings["page_total_ms"] = (
        timings["preprocessing_ms"] + total_ocr_ms
    )

    # Build OCRPage from reconstructed dict
    items = []
    for it in reconstructed["items"]:
        bb = it["bbox"]
        items.append(OCRItem(
            item_id=it["item_id"], page_number=page, text=it["text"],
            confidence=it.get("confidence", 0.9),
            bbox=BBox(bb["x1"], bb["y1"], bb["x2"], bb["y2"]),
        ))
    lines = []
    for ln in reconstructed["lines"]:
        lb = ln["bbox"]
        lines.append(OCRLine(
            line_id=ln["line_id"], page_number=page, text=ln["text"],
            confidence=ln.get("confidence", 0.9),
            bbox=BBox(lb["x1"], lb["y1"], lb["x2"], lb["y2"]),
        ))

    page_data = OCRPage(
        page_number=page, width=width, height=height,
        items=items, lines=lines,
    )
    page_data.page_metrics = {
        k: round(v, 3) for k, v in timings.items()
    }
    page_data.page_metrics["classification"] = "tiled"
    page_data.page_metrics["render_strategy"] = "tile"
    page_data.page_metrics["tile_count"] = len(tiles)

    logger.info(
        "Page %d: tiled=%d tiles ocr=%.0fms lines=%d",
        page, len(tiles), total_ocr_ms, len(lines),
    )

    return page_data, timings, processing_result


def _page_dict_to_ocrpage(d: dict) -> OCRPage:
    """Reconstruct an OCRPage from its cached dict representation."""
    items = []
    for item_d in d.get("items", []):
        bbox_d = item_d.get("bbox", {})
        nbbox_d = item_d.get("normalized_bbox") or {}
        items.append(OCRItem(
            item_id=item_d.get("item_id", ""),
            page_number=item_d.get("page_number", d.get("page_number", 0)),
            text=item_d.get("text", ""),
            confidence=float(item_d.get("confidence", 0.0)),
            polygon=item_d.get("polygon", []),
            bbox=BBox(**bbox_d) if bbox_d else BBox(0, 0, 0, 0),
            normalized_bbox=BBox(**nbbox_d) if nbbox_d else None,
            line_id=item_d.get("line_id"),
            block_id=item_d.get("block_id"),
            reading_order=int(item_d.get("reading_order", 0)),
        ))
    lines = []
    for line_d in d.get("lines", []):
        lb_d = line_d.get("bbox", {})
        nb_d = line_d.get("normalized_bbox") or {}
        lines.append(OCRLine(
            line_id=line_d.get("line_id", ""),
            page_number=line_d.get("page_number", d.get("page_number", 0)),
            text=line_d.get("text", ""),
            confidence=float(line_d.get("confidence", 0.0)),
            bbox=BBox(**lb_d) if lb_d else BBox(0, 0, 0, 0),
            normalized_bbox=BBox(**nb_d) if nb_d else None,
            item_ids=line_d.get("item_ids", []),
            block_id=line_d.get("block_id"),
            reading_order=int(line_d.get("reading_order", 0)),
        ))
    blocks = []
    for block_d in d.get("blocks", []):
        bb_d = block_d.get("bbox", {})
        nbb_d = block_d.get("normalized_bbox") or {}
        blocks.append(OCRBlock(
            block_id=block_d.get("block_id", ""),
            page_number=block_d.get("page_number", d.get("page_number", 0)),
            block_type=block_d.get("block_type", "text"),
            bbox=BBox(**bb_d) if bb_d else BBox(0, 0, 0, 0),
            confidence=float(block_d.get("confidence", 0.0)),
            normalized_bbox=BBox(**nbb_d) if nbb_d else None,
            line_ids=block_d.get("line_ids", []),
            reading_order=int(block_d.get("reading_order", 0)),
        ))
    return OCRPage(
        page_number=int(d.get("page_number", 0)),
        width=float(d.get("width", 0)),
        height=float(d.get("height", 0)),
        rotation=float(d.get("rotation", 0.0)),
        items=items,
        lines=lines,
        blocks=blocks,
        text=d.get("text", ""),
        confidence=float(d.get("confidence", 0.0)),
        duration_ms=float(d.get("duration_ms", 0.0)),
        page_metrics={k: (float(v) if isinstance(v, (int, float)) else v) for k, v in d.get("page_metrics", {}).items()},
    )


def _ocrpage_to_page_dict(page: OCRPage) -> dict:
    """Serialize an OCRPage to the canonical page dict for caching.

    Mirrors the per-page section of build_response().
    """
    items_out = []
    for item in page.items:
        items_out.append({
            "item_id": item.item_id,
            "page_number": item.page_number,
            "text": item.text,
            "confidence": round(item.confidence, 4),
            "polygon": [[round(p[0], 1), round(p[1], 1)] for p in item.polygon],
            "bbox": {
                "x1": round(item.bbox.x1, 1),
                "y1": round(item.bbox.y1, 1),
                "x2": round(item.bbox.x2, 1),
                "y2": round(item.bbox.y2, 1),
            },
            "normalized_bbox": {
                "x1": round(item.normalized_bbox.x1, 4),
                "y1": round(item.normalized_bbox.y1, 4),
                "x2": round(item.normalized_bbox.x2, 4),
                "y2": round(item.normalized_bbox.y2, 4),
            } if item.normalized_bbox else None,
            "line_id": item.line_id,
            "block_id": item.block_id,
            "reading_order": item.reading_order,
        })
    lines_out = []
    for line in page.lines:
        lines_out.append({
            "line_id": line.line_id,
            "page_number": line.page_number,
            "text": line.text,
            "confidence": round(line.confidence, 4),
            "bbox": {
                "x1": round(line.bbox.x1, 1),
                "y1": round(line.bbox.y1, 1),
                "x2": round(line.bbox.x2, 1),
                "y2": round(line.bbox.y2, 1),
            },
            "normalized_bbox": {
                "x1": round(line.normalized_bbox.x1, 4),
                "y1": round(line.normalized_bbox.y1, 4),
                "x2": round(line.normalized_bbox.x2, 4),
                "y2": round(line.normalized_bbox.y2, 4),
            } if line.normalized_bbox else None,
            "item_ids": line.item_ids,
            "block_id": line.block_id,
            "reading_order": line.reading_order,
        })
    blocks_out = []
    for block in page.blocks:
        blocks_out.append({
            "block_id": block.block_id,
            "page_number": block.page_number,
            "block_type": block.block_type,
            "bbox": {
                "x1": round(block.bbox.x1, 1),
                "y1": round(block.bbox.y1, 1),
                "x2": round(block.bbox.x2, 1),
                "y2": round(block.bbox.y2, 1),
            },
            "confidence": round(block.confidence, 4),
            "normalized_bbox": {
                "x1": round(block.normalized_bbox.x1, 4),
                "y1": round(block.normalized_bbox.y1, 4),
                "x2": round(block.normalized_bbox.x2, 4),
                "y2": round(block.normalized_bbox.y2, 4),
            } if block.normalized_bbox else None,
            "line_ids": block.line_ids,
            "reading_order": block.reading_order,
        })
    return {
        "page_number": page.page_number,
        "width": page.width,
        "height": page.height,
        "rotation": page.rotation,
        "items": items_out,
        "lines": lines_out,
        "blocks": blocks_out,
        "text": page.text,
        "confidence": round(page.confidence, 4),
        "duration_ms": round(page.duration_ms, 1),
        "page_metrics": {
            k: (round(v, 3) if isinstance(v, (int, float)) else v)
            for k, v in page.page_metrics.items()
        },
    }


def _cache_page_now(
    file_hash: str,
    page_number: int,
    page_fingerprint: str,
    ocrpage: OCRPage,
    failures: list[int],
) -> None:
    """Cache a single page immediately after processing.

    On cache-write failure, records the page in *failures* so the
    caller can decide whether to abort (when cache is required).
    """
    if not SETTINGS.ocr_result_cache_enabled:
        return
    page_dict = _ocrpage_to_page_dict(ocrpage)
    success = put_cached_page(file_hash, page_number, page_fingerprint, page_dict)
    if not success:
        logger.warning("Failed to cache page %d", page_number)
        failures.append(page_number)


def extract_internal_ocr_document(
    *,
    file_bytes: bytes,
    filename: str,
    content_type: str | None,
    image_processing: str = "false",
    pages: str | None = None,
    image_processing_profile: str = "none",
) -> dict[str, Any]:
    total_start = time.perf_counter()
    filename = filename or "uploaded-file"
    lower_name = filename.lower()
    file_type = "pdf" if lower_name.endswith(".pdf") else "image" if (content_type or "").startswith("image/") else None
    if file_type is None:
        raise ValueError("Unsupported file type")

    enhance = image_processing.lower() == "true"
    profile_name = resolve_profile(enhance, image_processing_profile)

    # Compute file hash and check cache
    file_hash = compute_file_hash(file_bytes)

    # Determine selected pages (preliminary, for cache key)
    if file_type == "pdf":
        doc = open_pdf_from_bytes(file_bytes)
        page_count = len(doc)
        try:
            selection = select_pages(page_count, pages, int(SETTINGS.ocr_max_pdf_pages_per_request))
            selected_pages = selection.page_numbers
        finally:
            doc.close()
    else:
        page_count = 1
        selected_pages = [1]

    scale = 1.6 if enhance else SETTINGS.ocr_pdf_render_scale
    fingerprint = compute_request_fingerprint(
        file_hash=file_hash,
        selected_pages=selected_pages,
        image_processing_enabled=enhance,
        image_processing_profile=profile_name,
        pipeline_version=SETTINGS.ocr_image_processing_pipeline_version,
        ocr_lang=SETTINGS.ocr_lang,
        engine_version=SETTINGS.default_release_tag,
        render_scale=scale,
    )

    # Check cache
    cache_hit = False
    if SETTINGS.ocr_result_cache_enabled:
        cached = get_cached_result(fingerprint)
        if cached is not None:
            logger.info("Cache hit for fingerprint %s", fingerprint[:16])
            response = dict(cached.response)
            response["processing"] = {
                "image_processing": {
                    "enabled": enhance,
                    "profile": profile_name,
                    "pipeline_version": SETTINGS.ocr_image_processing_pipeline_version,
                },
                "cache": {"status": "hit", "expires_at": cached.expires_at},
            }
            return response

    # ── Page-level partial resume ────────────────────────────────────────
    metrics: dict[str, float] = {
        "rendering_ms": 0.0, "preprocessing_ms": 0.0, "ocr_inference_ms": 0.0,
        "paddle_adaptation_ms": 0.0, "line_reconstruction_ms": 0.0,
        "block_reconstruction_ms": 0.0, "reading_order_ms": 0.0,
        "geometry_total_ms": 0.0,
    }
    # Recompute per-page by swapping in page number and actual render scale
    def _page_fp(pn: int, page_scale: float) -> str:
        return compute_page_fingerprint(
            file_hash=file_hash,
            page_number=pn,
            image_processing_enabled=enhance,
            image_processing_profile=profile_name,
            pipeline_version=SETTINGS.ocr_image_processing_pipeline_version,
            ocr_lang=SETTINGS.ocr_lang,
            engine_version=SETTINGS.default_release_tag,
            render_scale=page_scale,
        )

    pages_data: list[OCRPage] = []
    processing_meta: dict[str, Any] = {}
    cumulative_pixels = 0
    cache_failures: list[int] = []
    preview_scale = 0.5  # cheap ~36 DPI for blank/low-content detection

    try:
        with tempfile.TemporaryDirectory(prefix="dokstract-ocr-") as tmp_dir:
            safe_filename = Path(filename).name
            file_path = Path(tmp_dir) / safe_filename
            file_path.write_bytes(file_bytes)

            if file_type == "pdf":
                doc = open_pdf_from_bytes(file_bytes)
                try:
                    enforce_pdf_page_budget(
                        total_pages=page_count,
                        selected_pages=len(selected_pages),
                        max_pages_per_file=int(SETTINGS.ocr_max_pdf_pages_per_file),
                        max_pages_per_request=int(SETTINGS.ocr_max_pdf_pages_per_request),
                    )
                    for page_number in selected_pages:
                        page = doc[page_number - 1]

                        # Digital PDF fast path — no rendering needed
                        if _has_usable_digital_text(page):
                            pfp = _page_fp(page_number, scale)
                            cached_page_dict = get_cached_page(file_hash, page_number, pfp)
                            if cached_page_dict is not None:
                                pages_data.append(_page_dict_to_ocrpage(cached_page_dict))
                                logger.info("Page %d: digital, loaded from page cache", page_number)
                                continue
                            ocrpage = extract_digital_page(page, page_number, 0, 0)
                            if ocrpage is not None:
                                pages_data.append(ocrpage)
                                ocrpage.page_metrics["rendering_profile"] = "digital"
                                _cache_page_now(file_hash, page_number, pfp, ocrpage, cache_failures)
                                continue

                        # ── Cheap preview for blank + low-content detection ──
                        preview = pdf_page_to_img(page, preview_scale)
                        metrics["rendering_ms"] += 0.0  # preview cost negligible

                        # Blank check on preview
                        if is_blank_page(preview):
                            ocrpage = OCRPage(
                                page_number=page_number,
                                width=preview.shape[1],
                                height=preview.shape[0],
                            )
                            ocrpage.page_metrics = {
                                "source": "blank",
                                "rendering_profile": "blank",
                                "render_scale": 0.0,
                                "render_dpi": 0.0,
                            }
                            pages_data.append(ocrpage)
                            logger.debug("Page %d: blank, skipped OCR", page_number)
                            pfp = _page_fp(page_number, 0.0)
                            _cache_page_now(file_hash, page_number, pfp, ocrpage, cache_failures)
                            continue

                        # Low-content classification on preview
                        if is_low_content_page(preview):
                            page_scale = SETTINGS.ocr_low_content_render_scale
                            rendering_profile = "low_content"
                        else:
                            page_scale = scale
                            rendering_profile = "standard"

                        del preview  # release immediately

                        # ── Determine render dimensions ────────────────
                        rect = page.rect
                        max_dpi = max(72, int(SETTINGS.ocr_max_render_dpi))
                        eff_scale = min(page_scale, max_dpi / 72.0)
                        rw = int(rect.width * eff_scale)
                        rh = int(rect.height * eff_scale)
                        _validate_image_dimensions(rw, rh)
                        cumulative_pixels += rw * rh
                        _validate_render_budget(cumulative_pixels, page_number)

                        # ── Check page cache with actual scale ─────────
                        pfp = _page_fp(page_number, page_scale)
                        cached_page_dict = get_cached_page(file_hash, page_number, pfp)
                        if cached_page_dict is not None:
                            pages_data.append(_page_dict_to_ocrpage(cached_page_dict))
                            logger.info("Page %d: loaded from page cache (profile=%s)", page_number, rendering_profile)
                            continue

                        # ── Render at chosen scale ─────────────────────
                        render_start = time.perf_counter()
                        image = pdf_page_to_img(page, eff_scale)
                        metrics["rendering_ms"] += (time.perf_counter() - render_start) * 1000.0

                        # OCR + geometry pipeline (with per-page error handling)
                        try:
                            ocrpage, page_timings, proc_result = _build_results_for_image(
                                image, page=page_number, enhance=enhance, profile_name=profile_name,
                            )
                            for key in metrics:
                                metrics[key] += page_timings.get(key, 0.0)
                            if proc_result and not processing_meta:
                                processing_meta = {
                                    "operations_applied": proc_result.operations_applied,
                                    "operations_skipped": proc_result.operations_skipped,
                                }
                            # Add rendering metadata
                            ocrpage.page_metrics["rendering_profile"] = rendering_profile
                            ocrpage.page_metrics["render_scale"] = round(page_scale, 2)
                            ocrpage.page_metrics["render_dpi"] = round(page_scale * 72, 1)
                            pages_data.append(ocrpage)

                            # Cache page immediately
                            _cache_page_now(file_hash, page_number, pfp, ocrpage, cache_failures)
                        except ValueError as page_exc:
                            # Per-page failure — record and continue to next page
                            msg = str(page_exc)
                            logger.warning("Page %d failed: %s", page_number, msg[:200])
                            error_page = OCRPage(page_number=page_number, width=rw, height=rh)
                            error_page.page_metrics = {
                                "source": "error",
                                "error": msg[:200],
                                "rendering_profile": rendering_profile,
                            }
                            pages_data.append(error_page)
                finally:
                    doc.close()
            else:
                # Image file — use standard scale for single page
                img_scale = scale
                pfp = _page_fp(1, img_scale)
                cached_page_dict = get_cached_page(file_hash, 1, pfp)
                if cached_page_dict is not None:
                    ocrpage = _page_dict_to_ocrpage(cached_page_dict)
                    pages_data.append(ocrpage)
                    logger.info("Page 1: loaded from page cache")
                else:
                    inspect_image_metadata(file_bytes)
                    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
                    if image is None:
                        raise ValueError("Invalid image")
                    try:
                        ocrpage, page_timings, proc_result = _build_results_for_image(
                            image, page=1, enhance=enhance, profile_name=profile_name,
                        )
                        for key in metrics:
                            metrics[key] = page_timings.get(key, 0.0)
                        if proc_result:
                            processing_meta = {
                                "operations_applied": proc_result.operations_applied,
                                "operations_skipped": proc_result.operations_skipped,
                            }
                        ocrpage.page_metrics["rendering_profile"] = "standard"
                        ocrpage.page_metrics["render_scale"] = round(img_scale, 2)
                        ocrpage.page_metrics["render_dpi"] = round(img_scale * 72, 1)
                        pages_data.append(ocrpage)
                        _cache_page_now(file_hash, 1, pfp, ocrpage, cache_failures)
                    except ValueError as page_exc:
                        msg = str(page_exc)
                        logger.warning("Image page failed: %s", msg[:200])
                        error_page = OCRPage(page_number=1, width=0, height=0)
                        error_page.page_metrics = {"source": "error", "error": msg[:200]}
                        pages_data.append(error_page)
    finally:
        # Temp directory cleaned by context manager; rendered images are
        # always deleted when the temp dir is removed.
        pass

    # Fail if any required page-cache write failed
    if cache_failures and SETTINGS.ocr_result_cache_required:
        raise RuntimeError(
            f"Page cache write failed for pages {cache_failures} and cache is required"
        )

    total_duration_ms = (time.perf_counter() - total_start) * 1000.0

    response = build_response(
        pages_data=pages_data, filename=filename, file_type=file_type,
        total_pages=page_count, selected_pages=selected_pages,
        total_duration_ms=total_duration_ms, metrics=metrics,
    )

    response["processing"] = {
        "image_processing": {
            "enabled": enhance,
            "profile": profile_name,
            "pipeline_version": SETTINGS.ocr_image_processing_pipeline_version,
        },
        "cache": {"status": "miss"},
    }
    if processing_meta:
        response["processing"]["image_processing"].update(processing_meta)

    # Write full-document cache only after all pages succeeded
    if SETTINGS.ocr_result_cache_enabled:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        ttl = SETTINGS.ocr_result_cache_ttl_seconds
        try:
            from datetime import timedelta
            expires = now + timedelta(seconds=ttl)
        except Exception:
            expires = now

        cached_result = CachedOCRResult(
            cache_key=fingerprint,
            status="completed",
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
            engine_version=SETTINGS.default_release_tag,
            request_fingerprint=fingerprint,
            processing={
                "selected_pages": selected_pages,
                "image_processing_enabled": enhance,
                "image_processing_profile": profile_name,
            },
            response=response,
        )
        success = put_cached_result(fingerprint, cached_result)
        if success:
            response["processing"]["cache"] = {"status": "stored", "expires_at": expires.isoformat()}
        elif SETTINGS.ocr_result_cache_required:
            raise RuntimeError("OCR result cache write failed and cache is required")
        else:
            response["processing"]["cache"] = {"status": "unavailable"}

    return response
