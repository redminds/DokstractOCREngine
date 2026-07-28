"""Page-level dimension classification and adaptive rendering.

Replaces document-level stitched-page rejection with per-page
classification and safe adaptive handling (downscale, tile, skip).

Aspect ratio alone is never a rejection reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from app.core.config import SETTINGS

logger = logging.getLogger("dokstract.ocr_engine.page_classifier")


class PageSizeClassification(str, Enum):
    """Per-page dimension classification."""
    NORMAL = "normal"
    LARGE = "large"
    EXTREME_ASPECT_RATIO = "extreme_aspect_ratio"
    OVERSIZED_PIXEL_COUNT = "oversized_pixel_count"
    POSSIBLE_STITCHED = "possible_stitched"
    UNSAFE_TO_RENDER = "unsafe_to_render"


@dataclass
class PageClassification:
    """Result of page dimension classification."""
    classification: PageSizeClassification
    measured_width: int
    measured_height: int
    aspect_ratio: float
    pixel_count: int
    estimated_memory_mb: float
    triggered_rules: list[str] = field(default_factory=list)
    recommendation: str = ""


def classify_page_dimensions(
    width: int,
    height: int,
    *,
    render_scale: float = 1.0,
) -> PageClassification:
    """Classify a page by its rendered dimensions.

    Uses configurable thresholds.  Returns a PageClassification
    with the severity level and recommended handling.

    NEVER rejects solely on aspect ratio.
    """
    from app.core.config import SETTINGS as s

    pixel_count = width * height
    max_dim = max(width, height)
    min_dim = min(width, height)
    aspect_ratio = max_dim / max(min_dim, 1)
    # Rough memory estimate: 3 channels × float32 × 3x working copies
    estimated_memory_mb = (pixel_count * 3 * 4 * 3) / (1024 * 1024)

    rules: list[str] = []

    # ── Absolute safety ──────────────────────────────────────────────
    if width > s.ocr_max_image_width_pixels or height > s.ocr_max_image_height_pixels:
        return PageClassification(
            classification=PageSizeClassification.UNSAFE_TO_RENDER,
            measured_width=width, measured_height=height,
            aspect_ratio=aspect_ratio, pixel_count=pixel_count,
            estimated_memory_mb=estimated_memory_mb,
            triggered_rules=[f"exceeds_absolute_dimension_limit "
                             f"({width}×{height} vs {s.ocr_max_image_width_pixels}×{s.ocr_max_image_height_pixels})"],
            recommendation="reject_page",
        )

    if pixel_count > s.ocr_max_image_total_pixels:
        return PageClassification(
            classification=PageSizeClassification.UNSAFE_TO_RENDER,
            measured_width=width, measured_height=height,
            aspect_ratio=aspect_ratio, pixel_count=pixel_count,
            estimated_memory_mb=estimated_memory_mb,
            triggered_rules=[f"exceeds_total_pixel_limit "
                             f"({pixel_count} vs {s.ocr_max_image_total_pixels})"],
            recommendation="reject_page",
        )

    # ── Oversized pixel count ────────────────────────────────────────
    if pixel_count > s.ocr_max_image_pixels:
        return PageClassification(
            classification=PageSizeClassification.OVERSIZED_PIXEL_COUNT,
            measured_width=width, measured_height=height,
            aspect_ratio=aspect_ratio, pixel_count=pixel_count,
            estimated_memory_mb=estimated_memory_mb,
            triggered_rules=[f"pixel_count {pixel_count} > limit {s.ocr_max_image_pixels}"],
            recommendation="downscale_or_tile",
        )

    # ── Extreme aspect ratio ─────────────────────────────────────────
    # Aspect ratio ALONE is never a rejection reason.
    # It contributes to possible_stitched only when combined with
    # evidence from the stitched-page detector.
    if aspect_ratio > s.ocr_max_page_aspect_ratio:
        rules.append(f"aspect_ratio {aspect_ratio:.1f} > {s.ocr_max_page_aspect_ratio}")
        # Check for stitched-page signals if detector is enabled
        if s.ocr_stitched_page_detection_enabled:
            # We can't run the full detector here (no image), but
            # extreme aspect alone → possible_stitched, to be confirmed
            # by the image-level detector
            return PageClassification(
                classification=PageSizeClassification.POSSIBLE_STITCHED,
                measured_width=width, measured_height=height,
                aspect_ratio=aspect_ratio, pixel_count=pixel_count,
                estimated_memory_mb=estimated_memory_mb,
                triggered_rules=rules,
                recommendation="run_stitched_detector_then_tile_if_confirmed",
            )
        return PageClassification(
            classification=PageSizeClassification.EXTREME_ASPECT_RATIO,
            measured_width=width, measured_height=height,
            aspect_ratio=aspect_ratio, pixel_count=pixel_count,
            estimated_memory_mb=estimated_memory_mb,
            triggered_rules=rules,
            recommendation="segment_vertically",
        )

    # ── Large page (elevated but not extreme) ─────────────────────────
    if max_dim > s.ocr_max_image_width or max_dim > s.ocr_max_image_height:
        return PageClassification(
            classification=PageSizeClassification.LARGE,
            measured_width=width, measured_height=height,
            aspect_ratio=aspect_ratio, pixel_count=pixel_count,
            estimated_memory_mb=estimated_memory_mb,
            triggered_rules=[f"max_dim {max_dim} > limit"],
            recommendation="reduced_render_scale",
        )

    # ── Normal ───────────────────────────────────────────────────────
    return PageClassification(
        classification=PageSizeClassification.NORMAL,
        measured_width=width, measured_height=height,
        aspect_ratio=aspect_ratio, pixel_count=pixel_count,
        estimated_memory_mb=estimated_memory_mb,
        recommendation="standard_ocr",
    )


def run_stitched_detector_on_image(image: np.ndarray) -> bool:
    """Run the lightweight stitched-page detector on an image.

    Returns True if the image is suspected to be stitched.
    Does NOT reject — caller decides the action.
    """
    if not SETTINGS.ocr_stitched_page_detection_enabled:
        return False
    from app.services.ocr.stitched_detection import detect_stitched_page
    result = detect_stitched_page(image)
    if result.suspected_stitched_page:
        logger.info(
            "stitched_detector: suspected confidence=%.2f reasons=%s pages=%d",
            result.confidence, result.reasons, result.estimated_visual_pages,
        )
        return True
    return False


# ── Adaptive rendering ────────────────────────────────────────────────


@dataclass
class AdaptiveRenderPlan:
    """Plan for rendering a page adaptively."""
    strategy: str  # "standard", "downscale", "tile"
    render_scale: float
    tile_height: int = 0
    tile_overlap: int = 0
    tile_count: int = 0
    reason: str = ""


def compute_adaptive_render_plan(
    width: int,
    height: int,
    classification: PageClassification,
) -> AdaptiveRenderPlan:
    """Determine the adaptive render strategy for a page.

    Returns an AdaptiveRenderPlan with the strategy and parameters.
    """
    s = SETTINGS

    if classification.classification == PageSizeClassification.NORMAL:
        return AdaptiveRenderPlan(
            strategy="standard",
            render_scale=classification.aspect_ratio if False else 1.0,  # use caller's scale
            reason="normal_page",
        )

    if classification.classification == PageSizeClassification.UNSAFE_TO_RENDER:
        return AdaptiveRenderPlan(
            strategy="reject",
            render_scale=0.0,
            reason="unsafe_to_render",
        )

    # LARGE or OVERSIZED: try downscale first
    if classification.classification in (
        PageSizeClassification.LARGE,
        PageSizeClassification.OVERSIZED_PIXEL_COUNT,
    ):
        safe_scale = _compute_safe_downscale(width, height)
        if safe_scale >= 0.5:  # Minimum readable scale
            return AdaptiveRenderPlan(
                strategy="downscale",
                render_scale=safe_scale,
                reason=f"downscaled_from_{width}x{height}_scale_{safe_scale:.2f}",
            )
        # Still unsafe — fall through to tiling

    # POSSIBLE_STITCHED, EXTREME_ASPECT_RATIO, or failed downscale → tile
    tile_h = int(s.ocr_max_image_height * 0.9) if hasattr(s, 'ocr_max_image_height') else 5400
    overlap = int(tile_h * 0.15)
    tile_count = max(1, (height + tile_h - 1) // tile_h)

    return AdaptiveRenderPlan(
        strategy="tile",
        render_scale=1.0,  # caller provides actual scale
        tile_height=tile_h,
        tile_overlap=overlap,
        tile_count=tile_count,
        reason=f"tiled_{tile_count}_tiles_h_{tile_h}_overlap_{overlap}",
    )


def _compute_safe_downscale(width: int, height: int) -> float:
    """Compute a safe render scale to stay within pixel/memory limits."""
    s = SETTINGS
    pixel_limit = s.ocr_max_image_pixels
    current_pixels = width * height
    if current_pixels <= pixel_limit:
        return 1.0
    scale = (pixel_limit / current_pixels) ** 0.5
    return max(0.3, min(1.0, scale))


def tile_image_vertical(
    image: np.ndarray,
    tile_height: int,
    overlap: int,
) -> list[tuple[np.ndarray, int]]:
    """Split an image into overlapping vertical tiles.

    Returns list of (tile_image, y_offset) tuples.
    """
    height = image.shape[0]
    tiles: list[tuple[np.ndarray, int]] = []

    y = 0
    while y < height:
        y_end = min(y + tile_height, height)
        tile = image[y:y_end, :]
        if tile.shape[0] > 0 and tile.shape[1] > 0:
            tiles.append((tile, y))
        if y_end >= height:
            break
        y = y_end - overlap  # overlap region

    return tiles


def reconstruct_page_from_tiles(
    tile_results: list[dict],
    page_number: int,
    full_width: int,
    full_height: int,
) -> dict:
    """Reconstruct a canonical page result from tile OCR outputs.

    Each tile result should have: items, lines, blocks, tile_y_offset.
    Coordinates are translated to full-page space. Duplicates in
    overlap regions are removed by matching text + position.
    """
    all_items: list[dict] = []
    all_lines: list[dict] = []
    all_blocks: list[dict] = []
    seen_line_sigs: set[tuple[str, int, int]] = set()

    for tile_idx, tile_data in enumerate(tile_results):
        y_off = tile_data.get("tile_y_offset", 0)
        x_off = 0  # vertical tiling only

        # Translate items
        for item in tile_data.get("items", []):
            bbox = item.get("bbox", {})
            item["bbox"] = {
                "x1": bbox.get("x1", 0) + x_off,
                "y1": bbox.get("y1", 0) + y_off,
                "x2": bbox.get("x2", 0) + x_off,
                "y2": bbox.get("y2", 0) + y_off,
            }
            item["page_number"] = page_number
            item["item_id"] = f"p{page_number}_t{tile_idx}_{item.get('item_id', '')}"
            all_items.append(item)

        # Translate lines with dedup
        for line in tile_data.get("lines", []):
            lbbox = line.get("bbox", {})
            lx1 = lbbox.get("x1", 0) + x_off
            ly1 = lbbox.get("y1", 0) + y_off
            lx2 = lbbox.get("x2", 0) + x_off
            ly2 = lbbox.get("y2", 0) + y_off
            text = line.get("text", "").strip()

            # Dedup: same text + approximately same position
            sig = (text, int(ly1 / 10), int(lx1 / 10))
            if sig in seen_line_sigs:
                continue
            seen_line_sigs.add(sig)

            line["bbox"] = {"x1": lx1, "y1": ly1, "x2": lx2, "y2": ly2}
            line["page_number"] = page_number
            line["line_id"] = f"p{page_number}_t{tile_idx}_{line.get('line_id', '')}"
            all_lines.append(line)

        # Translate blocks
        for block in tile_data.get("blocks", []):
            bbox = block.get("bbox", {})
            block["bbox"] = {
                "x1": bbox.get("x1", 0) + x_off,
                "y1": bbox.get("y1", 0) + y_off,
                "x2": bbox.get("x2", 0) + x_off,
                "y2": bbox.get("y2", 0) + y_off,
            }
            block["page_number"] = page_number
            block["block_id"] = f"p{page_number}_t{tile_idx}_{block.get('block_id', '')}"
            all_blocks.append(block)

    return {
        "page_number": page_number,
        "width": full_width,
        "height": full_height,
        "items": all_items,
        "lines": all_lines,
        "blocks": all_blocks,
        "tiled": True,
        "tile_count": len(tile_results),
    }
