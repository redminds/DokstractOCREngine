"""Reusable geometry helpers for bounding-box operations.

All functions are stateless and work with the internal BBox and OCRItem models.
"""

from __future__ import annotations

import math

from .models import BBox, OCRItem


def polygon_to_bbox(polygon: list[list[float]]) -> BBox:
    """Convert a polygon (list of [x, y] points) to an axis-aligned bounding box.

    Args:
        polygon: List of points, e.g. [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].

    Returns:
        BBox with x1=min(x), y1=min(y), x2=max(x), y2=max(y).
    """
    if not polygon:
        return BBox(0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))


def normalize_bbox(bbox: BBox, page_width: float, page_height: float) -> BBox:
    """Normalize bounding box coordinates to [0.0, 1.0] relative to page dimensions.

    Args:
        bbox: Absolute bounding box in pixel coordinates.
        page_width: Page width in pixels.
        page_height: Page height in pixels.

    Returns:
        BBox with values clamped to [0.0, 1.0].
    """
    if page_width <= 0 or page_height <= 0:
        return BBox(0.0, 0.0, 0.0, 0.0)
    return BBox(
        x1=max(0.0, min(1.0, bbox.x1 / page_width)),
        y1=max(0.0, min(1.0, bbox.y1 / page_height)),
        x2=max(0.0, min(1.0, bbox.x2 / page_width)),
        y2=max(0.0, min(1.0, bbox.y2 / page_height)),
    )


def bbox_union(a: BBox, b: BBox) -> BBox:
    """Return the smallest bounding box that contains both A and B."""
    return BBox(
        x1=min(a.x1, b.x1),
        y1=min(a.y1, b.y1),
        x2=max(a.x2, b.x2),
        y2=max(a.y2, b.y2),
    )


def vertical_overlap(a: BBox, b: BBox) -> float:
    """Return the fraction of vertical overlap between two bounding boxes.

    The overlap is calculated as:
        overlap_height / min(height_a, height_b)

    Returns:
        A value between 0.0 (no overlap) and 1.0 (fully overlapping).
    """
    y_overlap = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    min_height = min(a.height, b.height)
    if min_height <= 0:
        return 0.0
    return y_overlap / min_height


def horizontal_overlap(a: BBox, b: BBox) -> float:
    """Return the fraction of horizontal overlap between two bounding boxes."""
    x_overlap = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    min_width = min(a.width, b.width)
    if min_width <= 0:
        return 0.0
    return x_overlap / min_width


def horizontal_distance(a: BBox, b: BBox) -> float:
    """Return the horizontal gap between two non-overlapping bounding boxes.

    Returns 0.0 if they overlap horizontally.
    """
    if a.x2 < b.x1:
        return b.x1 - a.x2
    if b.x2 < a.x1:
        return a.x1 - b.x2
    return 0.0


def center_point(bbox: BBox) -> tuple[float, float]:
    """Return the center (cx, cy) of a bounding box."""
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


def median_item_height(items: list[OCRItem]) -> float:
    """Calculate the median height of a list of OCR items.

    Returns 0.0 if the list is empty.
    """
    heights = sorted(item.bbox.height for item in items if item.bbox.height > 0)
    if not heights:
        return 0.0
    mid = len(heights) // 2
    if len(heights) % 2 == 0:
        return (heights[mid - 1] + heights[mid]) / 2.0
    return heights[mid]


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _polygon_area(polygon: list[list[float]]) -> float:
    return abs(sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )) / 2.0


def _bbox_out_of_bounds(bbox: BBox, width: float, height: float, tolerance: float) -> bool:
    return (
        bbox.x1 < -tolerance or bbox.y1 < -tolerance
        or bbox.x2 > width + tolerance or bbox.y2 > height + tolerance
    )


def validate_page_geometry(page: "OCRPage", tolerance: float = 1.0) -> dict[str, object]:
    """Validate canonical page geometry without mutating or dropping evidence.

    The result is intentionally diagnostic-only.  OCR consumers continue to
    receive the original evidence even when a warning is found.
    """
    width = float(page.width or 0.0)
    height = float(page.height or 0.0)
    invalid_polygon_count = 0
    out_of_bounds_polygon_count = 0
    zero_area_polygon_count = 0
    invalid_bbox_count = 0
    out_of_bounds_bbox_count = 0
    normalized_out_of_bounds_count = 0
    relationship_broken_count = 0
    item_ids = {item.item_id for item in page.items}
    line_ids = {line.line_id for line in page.lines}
    block_ids = {block.block_id for block in page.blocks}

    for item in page.items:
        polygon = item.polygon
        valid_polygon = isinstance(polygon, list) and len(polygon) >= 3
        if valid_polygon:
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2 or not all(_finite(v) for v in point[:2]):
                    valid_polygon = False
                    break
        if not valid_polygon:
            invalid_polygon_count += 1
        else:
            polygon = [[float(point[0]), float(point[1])] for point in polygon]
            if _polygon_area(polygon) <= 0.0:
                zero_area_polygon_count += 1
            if any(
                x < -tolerance or y < -tolerance
                or x > width + tolerance or y > height + tolerance
                for x, y in polygon
            ):
                out_of_bounds_polygon_count += 1

        bbox = item.bbox
        if not all(_finite(value) for value in bbox.to_list()) or not bbox.is_valid:
            invalid_bbox_count += 1
        elif _bbox_out_of_bounds(bbox, width, height, tolerance):
            out_of_bounds_bbox_count += 1

        normalized = item.normalized_bbox
        if normalized is None or any(
            not _finite(value) or value < -0.001 or value > 1.001
            for value in normalized.to_list()
        ):
            normalized_out_of_bounds_count += 1
        if item.line_id and item.line_id not in line_ids:
            relationship_broken_count += 1
        if item.block_id and item.block_id not in block_ids:
            relationship_broken_count += 1

    for line in page.lines:
        if any(item_id not in item_ids for item_id in line.item_ids):
            relationship_broken_count += 1
        if line.block_id and line.block_id not in block_ids:
            relationship_broken_count += 1
    for block in page.blocks:
        if any(line_id not in line_ids for line_id in block.line_ids):
            relationship_broken_count += 1

    warnings: list[str] = []
    if invalid_polygon_count or zero_area_polygon_count or invalid_bbox_count or normalized_out_of_bounds_count:
        warnings.append("GEOMETRY_INVALID")
    if out_of_bounds_polygon_count or out_of_bounds_bbox_count:
        warnings.append("GEOMETRY_OUT_OF_BOUNDS")
    if zero_area_polygon_count:
        warnings.append("GEOMETRY_ZERO_AREA")
    if relationship_broken_count:
        warnings.append("GEOMETRY_RELATIONSHIP_BROKEN")

    page_area = width * height
    coverage = 0.0
    if page_area > 0:
        coverage = min(1.0, sum(item.bbox.width * item.bbox.height for item in page.items) / page_area)
    y_edges = sorted({edge for item in page.items for edge in (item.bbox.y1, item.bbox.y2) if _finite(edge)})
    largest_empty_vertical_region = 0.0
    if height > 0 and y_edges:
        largest_empty_vertical_region = max(
            0.0,
            max((right - left) for left, right in zip(y_edges, y_edges[1:])) / height,
        )
    confidences = [float(item.confidence) for item in page.items if _finite(item.confidence)]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    low_confidence_count = sum(1 for value in confidences if value < 0.5)
    quality_signals: list[str] = list(warnings)
    if page.items and mean_confidence < 0.45:
        quality_signals.append("LOW_CONFIDENCE")
    if page.items and coverage < 0.003:
        quality_signals.append("LOW_TEXT_COVERAGE")
    if len(y_edges) >= 4 and largest_empty_vertical_region >= 0.35:
        quality_signals.append("SUSPICIOUS_EMPTY_REGION")

    return {
        "version": 1,
        "valid": not warnings,
        "ocr_item_count": len(page.items),
        "detected_region_count": len(page.items),
        "recognized_region_count": sum(1 for item in page.items if item.text.strip()),
        "mean_confidence": round(mean_confidence, 4),
        "low_confidence_count": low_confidence_count,
        "text_coverage_ratio": round(coverage, 6),
        "largest_empty_vertical_region": round(largest_empty_vertical_region, 6),
        "invalid_polygon_count": invalid_polygon_count,
        "out_of_bounds_polygon_count": out_of_bounds_polygon_count,
        "zero_area_polygon_count": zero_area_polygon_count,
        "invalid_bbox_count": invalid_bbox_count,
        "out_of_bounds_bbox_count": out_of_bounds_bbox_count,
        "normalized_out_of_bounds_count": normalized_out_of_bounds_count,
        "relationship_broken_count": relationship_broken_count,
        "warning_codes": sorted(set(warnings)),
        "quality_signals": sorted(set(quality_signals)),
    }
