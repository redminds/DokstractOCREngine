"""Reusable geometry helpers for bounding-box operations.

All functions are stateless and work with the internal BBox and OCRItem models.
"""

from __future__ import annotations

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
