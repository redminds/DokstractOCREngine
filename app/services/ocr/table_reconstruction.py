"""Table-aware line reconstruction for PaddleOCR detections — Phase 2.

Detects table-like regions, infers rows and columns independently,
builds a row×column cell matrix, and merges fragments within cells.

Uses normalized coordinates and median-height tolerances.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from .models import OCRItem, OCRLine, BBox
from .line_reconstruction import _sync_item_line_ids
from .geometry import (
    bbox_union, center_point, horizontal_overlap,
    vertical_overlap, median_item_height,
)

logger = logging.getLogger("dokstract.ocr_engine.table_reconstruction")

_MIN_ITEMS_PER_TABLE_ROW = 4
_ROW_OVERLAP_THRESHOLD = 0.35
_ROW_PROXIMITY_FACTOR = 1.2  # fraction of median height for row proximity
_COLUMN_MERGE_FACTOR = 0.18   # tighter threshold for same-column sub-headers
_HEADER_TOP_FRACTION = 0.25   # headers expected in top 25% of table region
_SERIAL_HEADER_RE = re.compile(r"^(?:s\.?\s*no\.?|sl\.?\s*no\.?|s\.?\s*n\.?|sno\.?|slno\.?)$", re.IGNORECASE)


def _item_sorter(it: OCRItem) -> tuple[float, float]:
    cx, cy = center_point(it.bbox)
    return (round(cy, 0), round(cx, 0))


def _detect_rows(
    items: list[OCRItem],
) -> list[list[OCRItem]]:
    """Group items into logical rows using vertical center proximity.

    Uses adaptive tolerance based on median item height.
    Items within _ROW_PROXIMITY_FACTOR * median_height of each other's
    y-center are candidates for the same row.
    """
    if not items:
        return []

    med_h = median_item_height(items) or 20
    row_tol = med_h * _ROW_PROXIMITY_FACTOR

    sorted_items = sorted(items, key=_item_sorter)
    rows: list[list[OCRItem]] = []

    for item in sorted_items:
        _, cy = center_point(item.bbox)
        placed = False
        for row in rows:
            row_cy = sum(center_point(r.bbox)[1] for r in row) / len(row)
            if abs(cy - row_cy) <= row_tol:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    return rows


def _detect_columns(
    header_items: list[OCRItem],
    data_items: list[OCRItem],
    page_width: float,
) -> list[tuple[float, float, list[str]]]:
    """Infer logical columns from header items only.

    Returns list of (x1, x2, header_texts) per column.
    Data items are NOT used for column detection — only headers determine
    column structure.  Data items inform alignment validation separately.
    """
    if len(header_items) < 2:
        return []

    med_w = sum(it.bbox.width for it in header_items) / max(len(header_items), 1)
    merge_thresh = med_w * 0.25  # slightly relaxed from 0.18 for real headers

    # Sort header items by cx (center x)
    sorted_headers = sorted(header_items, key=lambda it: center_point(it.bbox)[0])

    # Group headers into columns: merge only when BOTH x-close AND y-close
    # (same-column sub-headers are vertically stacked; adjacent columns are at same y)
    column_groups: list[list[OCRItem]] = []
    for it in sorted_headers:
        cx, cy = center_point(it.bbox)
        placed = False
        for grp in column_groups:
            g_cx, g_cy = center_point(grp[-1].bbox)
            x_dist = abs(cx - g_cx)
            y_dist = abs(cy - g_cy)
            # Merge if x-close AND vertically distinct (stacked sub-headers)
            if x_dist <= merge_thresh and y_dist > med_w * 0.3:
                grp.append(it)
                placed = True
                break
        if not placed:
            column_groups.append([it])

    # Build column bands from group bounding boxes
    bands: list[tuple[float, float, list[str]]] = []
    for i, grp in enumerate(column_groups):
        all_x1 = min(it.bbox.x1 for it in grp)
        all_x2 = max(it.bbox.x2 for it in grp)
        # Left boundary
        if i == 0:
            x1 = max(0, all_x1 - med_w * 0.3)
        else:
            prev_x2 = bands[-1][1]
            mid = (prev_x2 + all_x1) / 2
            x1 = mid
        # Right boundary
        if i == len(column_groups) - 1:
            x2 = min(page_width, all_x2 + med_w * 0.5)
        else:
            next_x1 = min(it.bbox.x1 for it in column_groups[i + 1])
            mid = (all_x2 + next_x1) / 2
            x2 = mid
        # Header texts: top-to-bottom within this column group
        headers = [it.text for it in sorted(grp, key=lambda it: it.bbox.y1)]
        bands.append((x1, x2, headers))

    return bands


def _is_serial_header(texts: Sequence[str]) -> bool:
    """Return True when a header band is just a serial/row index label."""
    compact = " ".join(texts).strip()
    if not compact:
        return False
    return bool(_SERIAL_HEADER_RE.match(compact)) or compact.replace(" ", "").lower() in {
        "sno", "slno", "srno", "sno.", "slno.",
    }


def _assign_columns(
    item: OCRItem,
    bands: list[tuple[float, float, list[str]]],
) -> tuple[int, float] | None:
    """Assign item to a column. Returns (col_idx, overlap_ratio) or None."""
    cx, _ = center_point(item.bbox)
    best_col = None
    best_overlap = 0.0
    cols_touched = 0

    for col_idx, (x1, x2, _) in enumerate(bands):
        if item.bbox.x1 < x2 and item.bbox.x2 > x1:
            cols_touched += 1
            # Prefer column where most of the item's width falls
            overlap_left = max(item.bbox.x1, x1)
            overlap_right = min(item.bbox.x2, x2)
            overlap_w = max(0, overlap_right - overlap_left)
            ratio = overlap_w / max(item.bbox.width, 1)
            if ratio > best_overlap:
                best_overlap = ratio
                best_col = col_idx

    if best_col is not None and cols_touched <= 3:
        return (best_col, best_overlap)
    return None


def _build_cell_matrix(
    items: list[OCRItem],
    rows: list[list[OCRItem]],
    bands: list[tuple[float, float, list[str]]],
    page_width: float,
    page_height: float,
) -> tuple[
    list[list[list[OCRItem]]],   # cell_matrix[row][col] = list of items
    list[OCRItem],                # unassigned
    list[OCRItem],                # spanning
    list[OCRItem],                # ambiguous
    float,                        # confidence
]:
    """Build row×column cell matrix from detected rows and columns."""
    n_rows = len(rows)
    n_cols = len(bands)
    matrix: list[list[list[OCRItem]]] = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
    unassigned: list[OCRItem] = []
    spanning: list[OCRItem] = []
    ambiguous: list[OCRItem] = []

    assigned_count = 0
    total_items = len(items)

    for item in items:
        # Find row
        _, cy = center_point(item.bbox)
        med_h = median_item_height(items) or 20
        row_idx = None
        best_row_dist = float("inf")
        for r_idx, row in enumerate(rows):
            row_cy = sum(center_point(r.bbox)[1] for r in row) / len(row) if row else 0
            dist = abs(cy - row_cy)
            if dist <= med_h * _ROW_PROXIMITY_FACTOR and dist < best_row_dist:
                best_row_dist = dist
                row_idx = r_idx

        # Find column
        col_result = _assign_columns(item, bands)
        if col_result is None:
            unassigned.append(item)
            continue
        col_idx, overlap = col_result

        if row_idx is None:
            unassigned.append(item)
            continue

        # Classify
        cols_touched = sum(1 for x1, x2, _ in bands if item.bbox.x1 < x2 and item.bbox.x2 > x1)
        if cols_touched >= 4:
            spanning.append(item)
        elif overlap < 0.4:
            ambiguous.append(item)
        else:
            matrix[row_idx][col_idx].append(item)
            assigned_count += 1

    # Confidence: fraction of items cleanly assigned
    confidence = assigned_count / max(total_items, 1)

    return matrix, unassigned, spanning, ambiguous, confidence


def _merge_cell_text(items: list[OCRItem]) -> str:
    """Merge items within a cell: left-to-right, preserving original text."""
    if not items:
        return ""
    sorted_items = sorted(items, key=lambda it: (it.bbox.y1, it.bbox.x1))
    return " ".join(it.text for it in sorted_items)


def _consolidate_logical_columns(
    physical_bands: list[tuple[float, float, list[str]]],
    header_items: list[OCRItem],
    data_rows: list[list[OCRItem]],
    med_height: float,
) -> tuple[list[dict], float]:
    """Consolidate physical bands into logical columns.

    Groups adjacent physical bands that share vertical header stacking
    or consistent data-item alignment into logical columns.

    Returns:
        (logical_columns, confidence) — each logical column has:
        index, header_text, header_item_ids, bbox, source_physical_bands
    """
    n_bands = len(physical_bands)
    if n_bands <= 1:
        return [], 0.0

    # For each physical band, collect its data items and header items
    band_headers: dict[int, list[OCRItem]] = {i: [] for i in range(n_bands)}
    band_data: dict[int, list[OCRItem]] = {i: [] for i in range(n_bands)}

    med_w = sum(it.bbox.width for it in header_items) / max(len(header_items), 1) if header_items else 100

    for it in header_items:
        cx = center_point(it.bbox)[0]
        for i, (x1, x2, _) in enumerate(physical_bands):
            if x1 <= cx <= x2:
                band_headers[i].append(it)
                break

    for row in data_rows:
        for it in row:
            cx = center_point(it.bbox)[0]
            for i, (x1, x2, _) in enumerate(physical_bands):
                if x1 <= cx <= x2:
                    band_data[i].append(it)
                    break

    # Greedy merge: consolidate adjacent bands
    merged = list(range(n_bands))  # merged[i] = logical column index
    for i in range(1, n_bands):
        h_prev = band_headers.get(i - 1, [])
        h_curr = band_headers.get(i, [])
        d_prev = band_data.get(i - 1, [])
        d_curr = band_data.get(i, [])

        _, _, h_texts_prev = physical_bands[i - 1]
        _, _, h_texts_curr = physical_bands[i]
        x1_prev, x2_prev, _ = physical_bands[i - 1]
        x1_curr, x2_curr, _ = physical_bands[i]
        gap = x1_curr - x2_prev

        should_merge = False

        # Keep the row-index prefix with the first substantive column.
        # Many sale-deed horizontal tables render the serial-number band as
        # a narrow leftmost prefix rather than a standalone semantic column.
        if i == 1 and _is_serial_header(h_texts_prev):
            if gap < med_w * 1.5:
                should_merge = True

        # Rule 1: Both have header items that are vertically stacked AND horizontally close
        if h_prev and h_curr:
            prev_cx = sum(center_point(it.bbox)[0] for it in h_prev) / len(h_prev)
            curr_cx = sum(center_point(it.bbox)[0] for it in h_curr) / len(h_curr)
            prev_cy = sum(center_point(it.bbox)[1] for it in h_prev) / len(h_prev)
            curr_cy = sum(center_point(it.bbox)[1] for it in h_curr) / len(h_curr)
            if abs(prev_cx - curr_cx) < med_w * 0.5 and abs(prev_cy - curr_cy) > med_height * 0.3:
                should_merge = True

        # Rule 2: One has headers, other has data aligning beneath it AND they're very close
        if not should_merge and ((h_prev and d_curr) or (h_curr and d_prev)):
            h_items = h_prev if h_prev else h_curr
            d_items = d_curr if d_curr else d_prev
            if h_items and d_items:
                h_cx = sum(center_point(it.bbox)[0] for it in h_items) / len(h_items)
                d_cx = sum(center_point(it.bbox)[0] for it in d_items) / len(d_items)
                # Tight alignment: data must be very close to header center
                if abs(h_cx - d_cx) < med_w * 0.3 and gap < med_w * 0.8:
                    should_merge = True

        # Rule 2b: Two data-only adjacent bands that are very narrow
        if not should_merge and not h_prev and not h_curr and gap < med_w * 0.1:
            should_merge = True

        # Rule 3: Very narrow gap between adjacent bands with no conflicting headers
        if not should_merge and gap < med_w * 0.2 and not (h_prev and h_curr and len(h_texts_prev) > 0 and len(h_texts_curr) > 0):
            should_merge = True

        if should_merge:
            merged[i] = merged[i - 1]

    # Build logical columns
    logical_groups: dict[int, list[int]] = {}
    for i in range(n_bands):
        root = merged[i]
        if root not in logical_groups:
            logical_groups[root] = []
        logical_groups[root].append(i)

    logical_columns: list[dict] = []
    for root, band_indices in sorted(logical_groups.items()):
        band_indices.sort()
        phys_bands = [physical_bands[i] for i in band_indices]
        all_x1 = min(b[0] for b in phys_bands)
        all_x2 = max(b[1] for b in phys_bands)
        all_headers = []
        all_header_ids = []
        for i in band_indices:
            for it in band_headers.get(i, []):
                all_headers.append(it.text)
                all_header_ids.append(it.item_id)
        header_text = " ".join(dict.fromkeys(all_headers)) if all_headers else ""
        logical_columns.append({
            "index": len(logical_columns),
            "header_text": header_text,
            "header_item_ids": all_header_ids,
            "bbox": [round(all_x1, 1), 0, round(all_x2, 1), 0],
            "source_physical_bands": band_indices,
        })

    # Confidence: fraction of bands that participated in multi-band logical columns
    consolidated_bands = sum(1 for i in range(n_bands) if len(logical_groups.get(merged[i], [])) > 1)
    logical_conf = consolidated_bands / max(n_bands, 1) if n_bands > 1 else 1.0

    return logical_columns, round(logical_conf, 4)


def _classify_rows(
    rows: list[list[OCRItem]],
    header_items: set[int],
    med_height: float,
) -> list[dict]:
    """Classify rows as primary_header, header_continuation, or data."""
    classified = []
    for row in rows:
        row_ids = {id(it) for it in row}
        header_overlap = len(row_ids & header_items)
        row_avg_y = sum(it.bbox.y1 for it in row) / max(len(row), 1)
        item_count = len(row)

        if header_overlap >= len(row) * 0.5:
            row_type = "primary_header"
        elif header_overlap > 0 and item_count < len(header_items) * 0.6:
            row_type = "header_continuation"
        else:
            row_type = "data"

        classified.append({
            "index": len(classified),
            "type": row_type,
            "bbox": [
                round(min(it.bbox.x1 for it in row), 1) if row else 0,
                round(min(it.bbox.y1 for it in row), 1) if row else 0,
                round(max(it.bbox.x2 for it in row), 1) if row else 0,
                round(max(it.bbox.y2 for it in row), 1) if row else 0,
            ],
            "item_count": item_count,
        })
    return classified


def _assign_logical_cell(
    item: OCRItem,
    logical_columns: list[dict],
    row_idx: int,
    med_height: float,
) -> tuple[int | None, dict]:
    """Assign an item to a logical column and classify assignment quality."""
    cx = center_point(item.bbox)[0]
    best_col = None
    best_overlap = 0.0
    cols_touched = 0
    candidate_cols = []

    for col in logical_columns:
        bbox = col["bbox"]
        if item.bbox.x1 < bbox[2] and item.bbox.x2 > bbox[0]:
            cols_touched += 1
            overlap = (min(item.bbox.x2, bbox[2]) - max(item.bbox.x1, bbox[0]))
            ratio = max(0, overlap) / max(item.bbox.width, 1)
            if ratio > best_overlap:
                best_overlap = ratio
                best_col = col["index"]
            if ratio > 0.3:
                candidate_cols.append(col["index"])

    status = {}
    if cols_touched >= 4:
        status["spanning"] = True
        return None, status
    if best_overlap < 0.3:
        status["ambiguous"] = True
        status["candidate_columns"] = candidate_cols
        return best_col, status
    return best_col, {}


def flatten_for_schema(old_func_continue_on_next_token: bool = False) -> None:
    pass  # placeholder — remove if unused


def detect_table_region(
    items: Sequence[OCRItem],
    page_width: float,
    page_height: float,
) -> tuple[list[OCRLine], list[OCRLine], dict | None]:
    """Detect and reconstruct a table structure from OCR items.

    Returns:
        (table_lines, non_table_lines, table_metadata)
        table_metadata is None if no table detected.
    """
    if len(items) < 5:
        return [], list(items), None

    item_list = list(items)
    rows = _detect_rows(item_list)

    # Find candidate rows with enough items spread across the page
    candidate_rows = [r for r in rows if len(r) >= _MIN_ITEMS_PER_TABLE_ROW]
    if len(candidate_rows) < 2:
        return [], list(items), None

    # Separate header region (top 25% of candidate rows' y-range)
    all_y1 = min(min(it.bbox.y1 for it in r) for r in candidate_rows)
    all_y2 = max(max(it.bbox.y2 for it in r) for r in candidate_rows)
    header_cutoff = all_y1 + (all_y2 - all_y1) * _HEADER_TOP_FRACTION

    # Header items: only from top 20% of candidate rows
    header_items: list[OCRItem] = []
    data_items: list[OCRItem] = []
    for r in candidate_rows:
        row_avg_y = sum(it.bbox.y1 for it in r) / len(r)
        if row_avg_y < header_cutoff:
            header_items.extend(r)
        else:
            data_items.extend(r)

    if len(header_items) < 3:
        return [], list(items), None

    bands = _detect_columns(header_items, data_items, page_width)
    if len(bands) < 3:
        return [], list(items), None

    # Collect all table-region items: only items in candidate rows
    table_items: list[OCRItem] = []
    non_table_items: list[OCRItem] = []
    candidate_row_ids = set()
    for r in candidate_rows:
        candidate_row_ids.update(id(it) for it in r)

    for it in item_list:
        if id(it) in candidate_row_ids:
            table_items.append(it)
        else:
            non_table_items.append(it)

    if len(table_items) < 6:
        return [], list(items), None

    # Re-detect rows using only table items for cleaner grouping
    table_rows = _detect_rows(table_items)

    # Build cell matrix
    matrix, unassigned, spanning, ambiguous, physical_confidence = _build_cell_matrix(
        table_items, table_rows, bands, page_width, page_height,
    )

    if physical_confidence < 0.2:
        return [], list(items), None

    med_h = median_item_height(table_items) or 20

    # Logical-column consolidation
    data_rows_for_consolidation = [r for r in table_rows if len(r) >= 1]
    logical_columns, logical_confidence = _consolidate_logical_columns(
        bands, header_items, data_rows_for_consolidation, med_h,
    )

    # Row classification
    header_ids = {id(it) for it in header_items}
    row_classes = _classify_rows(table_rows, header_ids, med_h)

    # Rebuild cell matrix with logical columns if consolidation improved things
    if logical_columns and len(logical_columns) < len(bands) and len(logical_columns) >= 2:
        use_columns = logical_columns
    else:
        use_columns = [
            {"index": i, "header_text": " ".join(b2), "bbox": [b0, 0, b1, 0],
             "header_item_ids": [], "source_physical_bands": [i]}
            for i, (b0, b1, b2) in enumerate(bands)
        ]
        logical_confidence = physical_confidence

    # Rebuild logical cell matrix
    log_matrix: list[list[list[OCRItem]]] = [[[] for _ in range(len(use_columns))] for _ in range(len(table_rows))]
    log_unassigned: list[OCRItem] = []
    log_spanning: list[dict] = []
    log_ambiguous: list[dict] = []

    for row_idx, row in enumerate(table_rows):
        for it in row:
            col_idx, status = _assign_logical_cell(it, use_columns, row_idx, med_h)
            if col_idx is not None and col_idx < len(use_columns):
                log_matrix[row_idx][col_idx].append(it)
            else:
                log_unassigned.append(it)

    # Build structured output: row-major lines
    result_lines: list[OCRLine] = []
    for row_idx, row_cells in enumerate(log_matrix):
        for col_idx, cell_items in enumerate(row_cells):
            if not cell_items:
                continue
            merged_text = _merge_cell_text(cell_items)
            cell_bbox = cell_items[0].bbox
            for it in cell_items[1:]:
                cell_bbox = bbox_union(cell_bbox, it.bbox)
            avg_conf = sum(it.confidence for it in cell_items) / len(cell_items)
            item_ids = [it.item_id for it in cell_items]

            line = OCRLine(
                line_id=f"p{cell_items[0].page_number}_r{row_idx}c{col_idx}",
                page_number=cell_items[0].page_number,
                text=merged_text,
                confidence=avg_conf,
                bbox=cell_bbox,
                item_ids=item_ids,
                reading_order=0,
            )
            result_lines.append(line)

    for i, line in enumerate(result_lines):
        line.reading_order = i

    _sync_item_line_ids(items, result_lines)

    overall_confidence = round((physical_confidence + logical_confidence) / 2, 4)

    # Build metadata
    metadata = {
        "section": "table",
        "physical_bands": [
            {"index": i, "x1": round(b[0], 1), "x2": round(b[1], 1),
             "header_text": " ".join(b[2]), "support": len(band_data.get(i, []) if 'band_data' in dir() else [])}
            for i, b in enumerate(bands)
        ],
        "physical_column_count": len(bands),
        "logical_columns": use_columns,
        "logical_column_count": len(use_columns),
        "header_rows": [r for r in row_classes if r["type"] in ("primary_header", "header_continuation")],
        "data_rows": [
            {"index": row_idx,
             "bbox": [
                 round(min(it.bbox.x1 for cell in row_cells for it in cell), 1) if any(row_cells) else 0,
                 round(min(it.bbox.y1 for cell in row_cells for it in cell), 1) if any(row_cells) else 0,
                 round(max(it.bbox.x2 for cell in row_cells for it in cell), 1) if any(row_cells) else 0,
                 round(max(it.bbox.y2 for cell in row_cells for it in cell), 1) if any(row_cells) else 0,
             ],
             "cells": [
                 {
                     "column_index": col_idx,
                     "column_header": use_columns[col_idx]["header_text"] if col_idx < len(use_columns) else "",
                     "text": _merge_cell_text(cell_items),
                     "bbox": [
                         round(min(it.bbox.x1 for it in cell_items), 1) if cell_items else 0,
                         round(min(it.bbox.y1 for it in cell_items), 1) if cell_items else 0,
                         round(max(it.bbox.x2 for it in cell_items), 1) if cell_items else 0,
                         round(max(it.bbox.y2 for it in cell_items), 1) if cell_items else 0,
                     ] if cell_items else None,
                     "confidence": round(sum(it.confidence for it in cell_items) / len(cell_items), 4) if cell_items else 0,
                     "item_ids": [it.item_id for it in cell_items],
                     "spanning": False,
                     "ambiguous": False,
                 }
                 for col_idx, cell_items in enumerate(row_cells)
                 if cell_items
             ],
            }
            for row_idx, row_cells in enumerate(log_matrix)
        ],
        "unassigned_items": [{"item_id": it.item_id, "text": it.text} for it in unassigned],
        "spanning_items": [{"item_id": it.item_id, "text": it.text} for it in spanning],
        "ambiguous_items": [{"item_id": it.item_id, "text": it.text} for it in ambiguous],
        "physical_confidence": round(physical_confidence, 4),
        "logical_confidence": logical_confidence,
        "overall_confidence": overall_confidence,
        "table_items": len(table_items),
        "table_lines": len(result_lines),
    }

    return result_lines, non_table_items, metadata
