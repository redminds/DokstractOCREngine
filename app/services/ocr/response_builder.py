"""Build the canonical OCR API response from internal OCRPage models.

Produces a single structured contract without duplicate timing fields.
All confidence values are in the 0.0–1.0 range.
"""

from __future__ import annotations

from typing import Any

from .models import OCRPage


def build_response(
    pages_data: list[OCRPage],
    filename: str,
    file_type: str,
    total_pages: int,
    selected_pages: list[int],
    total_duration_ms: float = 0.0,
    metrics: dict[str, float] | None = None,
    debug_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical OCR API response dict.

    Args:
        pages_data: Processed OCR pages (one per selected page).
        filename: Original uploaded filename.
        file_type: 'pdf' or 'image'.
        total_pages: Total page count in the document.
        selected_pages: List of 1-based page numbers that were processed.
        total_duration_ms: Total wall-clock duration in milliseconds.
        metrics: Timing metrics dict from the execution layer.

    Returns:
        Canonical response dict (no serialization overhead).
    """
    # ── File ────────────────────────────────────────────────────────────
    file_info: dict[str, str] = {
        "name": filename,
        "type": file_type,
    }

    # ── Document-level aggregates ────────────────────────────────────────
    all_items = []
    for page in pages_data:
        all_items.extend(page.items)

    doc_lines: list[str] = []
    for page in pages_data:
        for line in sorted(page.lines, key=lambda l: l.reading_order):
            if line.text.strip():
                doc_lines.append(line.text)
    doc_text = "\n".join(doc_lines)

    confidences = [item.confidence for item in all_items if item.confidence > 0.0]
    doc_confidence = (
        round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    )

    document: dict[str, Any] = {
        "total_pages": total_pages,
        "processed_pages": selected_pages,
        "text": doc_text,
        "confidence": doc_confidence,
        "duration_ms": round(total_duration_ms, 1),
    }

    # ── Pages ────────────────────────────────────────────────────────────
    pages_out: list[dict[str, Any]] = []

    for page in pages_data:
        # Items
        items_out: list[dict[str, Any]] = []
        for item in page.items:
            items_out.append({
                "item_id": item.item_id,
                "text": item.text,
                "confidence": round(item.confidence, 4),
                "polygon": [[round(p[0], 1), round(p[1], 1)] for p in item.polygon],
                "bbox": [
                    round(item.bbox.x1, 1),
                    round(item.bbox.y1, 1),
                    round(item.bbox.x2, 1),
                    round(item.bbox.y2, 1),
                ],
                "normalized_bbox": [
                    round(item.normalized_bbox.x1, 4),
                    round(item.normalized_bbox.y1, 4),
                    round(item.normalized_bbox.x2, 4),
                    round(item.normalized_bbox.y2, 4),
                ],
                "line_id": item.line_id,
                "block_id": item.block_id,
                "reading_order": item.reading_order,
            })

        # Lines
        lines_out: list[dict[str, Any]] = []
        for line in page.lines:
            norm = line.normalized_bbox
            lines_out.append({
                "line_id": line.line_id,
                "text": line.text,
                "confidence": round(line.confidence, 4),
                "bbox": [
                    round(line.bbox.x1, 1),
                    round(line.bbox.y1, 1),
                    round(line.bbox.x2, 1),
                    round(line.bbox.y2, 1),
                ],
                "normalized_bbox": [
                    round(norm.x1, 4),
                    round(norm.y1, 4),
                    round(norm.x2, 4),
                    round(norm.y2, 4),
                ] if norm else None,
                "item_ids": line.item_ids,
                "reading_order": line.reading_order,
            })

        # Blocks
        blocks_out: list[dict[str, Any]] = []
        for block in page.blocks:
            norm = block.normalized_bbox
            blocks_out.append({
                "block_id": block.block_id,
                "type": block.block_type,
                "confidence": round(block.confidence, 4),
                "bbox": [
                    round(block.bbox.x1, 1),
                    round(block.bbox.y1, 1),
                    round(block.bbox.x2, 1),
                    round(block.bbox.y2, 1),
                ],
                "normalized_bbox": [
                    round(norm.x1, 4),
                    round(norm.y1, 4),
                    round(norm.x2, 4),
                    round(norm.y2, 4),
                ] if norm else None,
                "line_ids": block.line_ids,
                "reading_order": block.reading_order,
            })

        pages_out.append({
            "page_number": page.page_number,
            "width": page.width,
            "height": page.height,
            "rotation": page.rotation,
            "text": page.text,
            "confidence": round(page.confidence, 4),
            "metrics": {
                k: (round(v, 3) if isinstance(v, (int, float)) else v)
                for k, v in (page.page_metrics or {}).items()
            },
            "items": items_out,
            "lines": lines_out,
            "blocks": blocks_out,
            **({"table_meta": page.table_meta} if page.table_meta else {}),
        })

    # ── Metrics ──────────────────────────────────────────────────────────
    metrics_out: dict[str, float] = {}
    if metrics:
        metrics_out.update(metrics)
    for key in (
        "rendering_ms", "preprocessing_ms", "ocr_inference_ms",
        "paddle_adaptation_ms", "line_reconstruction_ms",
        "block_reconstruction_ms", "reading_order_ms", "geometry_total_ms",
        "total_duration_ms",
    ):
        metrics_out.setdefault(key, 0.0)
    metrics_out["total_duration_ms"] = round(total_duration_ms, 3)

    return {
        "file": file_info,
        "document": document,
        "pages": pages_out,
        "metrics": {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics_out.items()},
        **({"_debug": debug_diagnostics} if debug_diagnostics else {}),
    }
