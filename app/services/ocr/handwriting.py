"""Optional handwriting-recognition boundary.

The stable OCR pipeline remains PaddleOCR-first.  This module deliberately
contains no model dependency: handwriting engines can be supplied later as an
optional adapter without changing the canonical OCR geometry contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence


HANDWRITING_CONTRACT_VERSION = "handwriting_v1"


@dataclass(frozen=True)
class HandwritingCandidate:
    """One OCR-grounded handwriting result in page-local coordinates."""

    text: str
    confidence: float
    polygon: tuple[tuple[float, float], ...]
    source: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HandwritingRequest:
    """Input supplied to an optional recognizer adapter."""

    page_number: int
    image: object
    page_width: float
    page_height: float
    printed_items: Sequence[object] = ()
    # Optional page-local crop bounding box: x1, y1, x2, y2.
    source_bbox: tuple[float, float, float, float] | None = None


class HandwritingRecognizer(Protocol):
    """Small adapter contract for optional local handwriting engines."""

    recognizer_id: str
    contract_version: str

    def recognize(self, request: HandwritingRequest) -> list[HandwritingCandidate]:
        """Return only candidates with valid page-local geometry."""


class DisabledHandwritingRecognizer:
    """Safe default; preserves the existing printed OCR behavior exactly."""

    recognizer_id = "disabled"
    contract_version = HANDWRITING_CONTRACT_VERSION

    def recognize(self, request: HandwritingRequest) -> list[HandwritingCandidate]:
        return []


def candidates_to_ocr_items(
    candidates: Sequence[HandwritingCandidate],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    start_index: int = 0,
):
    """Convert accepted handwriting candidates to canonical OCR items.

    This preserves the existing item geometry contract and marks provenance
    distinctly.  Callers must perform overlap/deduplication against printed
    items before adding the returned items to a page.
    """

    from .geometry import normalize_bbox, polygon_to_bbox
    from .models import OCRItem

    items = []
    for index, candidate in enumerate(candidates, start_index):
        polygon = [[float(x), float(y)] for x, y in candidate.polygon]
        bbox = polygon_to_bbox(polygon)
        if not bbox.is_valid:
            continue
        bbox = type(bbox)(
            max(0.0, bbox.x1), max(0.0, bbox.y1),
            min(page_width, bbox.x2), min(page_height, bbox.y2),
        )
        if not bbox.is_valid:
            continue
        items.append(OCRItem(
            item_id=f"p{page_number}_h{index}",
            page_number=page_number,
            text=candidate.text,
            confidence=max(0.0, min(1.0, float(candidate.confidence))),
            polygon=polygon,
            bbox=bbox,
            normalized_bbox=normalize_bbox(bbox, page_width, page_height),
            provenance=f"handwriting:{candidate.source}",
        ))
    return items


def get_handwriting_recognizer(*, enabled: bool, model_path: str | None = None) -> HandwritingRecognizer:
    """Resolve the configured recognizer without importing model packages.

    Until a separately evaluated model adapter is installed and accepted,
    enabling handwriting is intentionally fail-safe rather than silently
    falling back to a different OCR engine.
    """

    if not enabled:
        return DisabledHandwritingRecognizer()
    if not model_path:
        raise RuntimeError("Handwriting model path is required when recognition is enabled")
    from .handwriting_onnx import OnnxHandwritingRecognizer
    return OnnxHandwritingRecognizer(model_path)
