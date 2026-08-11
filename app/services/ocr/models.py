"""Internal typed models for OCR items, lines, blocks, and pages.

These models are independent of any specific OCR engine (PaddleOCR, Tesseract, etc.)
and are used throughout the geometry-aware OCR pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BBox:
    """Rectangular bounding box with (x1, y1) = top-left, (x2, y2) = bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def is_valid(self) -> bool:
        return self.x1 <= self.x2 and self.y1 <= self.y2


@dataclass
class OCRItem:
    """A single recognized text item with geometry and confidence."""

    item_id: str
    page_number: int
    text: str
    confidence: float  # 0.0 – 1.0
    polygon: list[list[float]]  # original PaddleOCR polygon [[x,y], ...]
    bbox: BBox
    normalized_bbox: BBox
    line_id: str | None = None
    block_id: str | None = None
    reading_order: int = 0


@dataclass
class OCRLine:
    """A reconstructed line containing one or more OCR items."""

    line_id: str
    page_number: int
    text: str
    confidence: float  # 0.0 – 1.0
    bbox: BBox
    normalized_bbox: BBox | None = None
    item_ids: list[str] = field(default_factory=list)
    block_id: str | None = None
    reading_order: int = 0


@dataclass
class OCRBlock:
    """A generic block grouping nearby lines."""

    block_id: str
    page_number: int
    block_type: str  # "text", "table_candidate", "unknown"
    bbox: BBox
    confidence: float = 0.0  # 0.0 – 1.0
    normalized_bbox: BBox | None = None
    line_ids: list[str] = field(default_factory=list)
    reading_order: int = 0


@dataclass
class OCRPage:
    """A single page of OCR results with geometry and timing."""

    page_number: int
    width: float
    height: float
    rotation: float = 0.0
    items: list[OCRItem] = field(default_factory=list)
    lines: list[OCRLine] = field(default_factory=list)
    blocks: list[OCRBlock] = field(default_factory=list)
    text: str = ""
    confidence: float = 0.0  # 0.0 – 1.0
    duration_ms: float = 0.0  # geometry-only time (backward compat; prefer metrics)
    page_metrics: dict[str, float] = field(default_factory=dict)
    table_meta: dict | None = None  # structured table metadata from table_reconstruction
