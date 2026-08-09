"""Tests for table-aware OCR line reconstruction — Phase 2."""
import json
from pathlib import Path

import pytest, math
from app.services.ocr.models import OCRItem, OCRLine, BBox
from app.services.ocr.geometry import normalize_bbox
from app.services.ocr.table_reconstruction import (
    _detect_rows, _detect_columns, _assign_columns,
    _build_cell_matrix, detect_table_region, _merge_cell_text,
)


def _item(text, x1, y1, x2, y2, idx=0, page=1, conf=0.9):
    bbox = BBox(x1, y1, x2, y2)
    nbbox = normalize_bbox(bbox, 2000, 3000)
    return OCRItem(
        item_id=f"p{page}_i{idx}", page_number=page, text=text, confidence=conf,
        bbox=bbox, normalized_bbox=nbbox,
        polygon=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
    )


# ═══════════════════════════════════════════════════════════════════════
# Row detection
# ═══════════════════════════════════════════════════════════════════════

class TestRowDetection:
    def test_groups_items_at_same_y_into_one_row(self):
        items = [_item("A", 100, 100, 200, 140, 0), _item("B", 300, 100, 400, 140, 1)]
        rows = _detect_rows(items)
        assert len(rows) == 1

    def test_separates_items_at_different_y(self):
        items = [_item("A", 100, 100, 200, 140, 0), _item("B", 100, 300, 200, 340, 1)]
        rows = _detect_rows(items)
        assert len(rows) == 2

    def test_handles_empty(self):
        assert _detect_rows([]) == []


# ═══════════════════════════════════════════════════════════════════════
# Column detection
# ═══════════════════════════════════════════════════════════════════════

class TestColumnDetection:
    def test_detects_seven_columns(self):
        headers = [
            _item("S.No.", 100, 800, 180, 860, 0),
            _item("Survey No.", 380, 800, 560, 860, 1),
            _item("Extent", 680, 800, 860, 860, 2),
            _item("Transferred", 690, 870, 870, 920, 3),
            _item("NORTH", 950, 800, 1100, 860, 4),
            _item("SOUTH", 1200, 800, 1350, 860, 5),
            _item("EAST", 1450, 800, 1600, 860, 6),
            _item("WEST", 1700, 800, 1850, 860, 7),
        ]
        bands = _detect_columns(headers, [], 2000)
        assert len(bands) >= 6, f"Expected >=6 columns, got {len(bands)}"

    def test_merges_stacked_subheaders(self):
        headers = [
            _item("S.No.", 100, 800, 180, 860, 0),
            _item("No.", 110, 870, 190, 920, 1),
            _item("Extent", 380, 800, 560, 860, 2),
            _item("Transferred", 390, 870, 570, 920, 3),
            _item("NORTH", 680, 800, 830, 860, 4),
        ]
        bands = _detect_columns(headers, [], 2000)
        # S.No.+No. merge, Extent+Transferred merge, NORTH solo = 3 bands
        assert len(bands) == 3, f"Expected 3 columns, got {len(bands)}"

    def test_keeps_north_south_separate(self):
        headers = [
            _item("NORTH", 900, 800, 1050, 860, 0),
            _item("SOUTH", 1200, 800, 1350, 860, 1),
        ]
        bands = _detect_columns(headers, [], 2000)
        assert len(bands) == 2


# ═══════════════════════════════════════════════════════════════════════
# Column assignment
# ═══════════════════════════════════════════════════════════════════════

class TestColumnAssignment:
    def test_assigns_to_correct_column(self):
        bands = [(0, 200, ["A"]), (200, 400, ["B"]), (400, 600, ["C"])]
        result = _assign_columns(_item("X", 100, 100, 150, 140, 0), bands)
        assert result is not None
        assert result[0] == 0

    def test_returns_none_for_wide_spanning_item(self):
        bands = [(0, 200, ["A"]), (200, 400, ["B"]), (400, 600, ["C"]), (600, 800, ["D"]), (800, 1000, ["E"])]
        result = _assign_columns(_item("Wide", 50, 100, 1900, 140, 0), bands)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# Cell matrix
# ═══════════════════════════════════════════════════════════════════════

class TestCellMatrix:
    def test_builds_matrix(self):
        items = [
            _item("H1", 100, 800, 180, 860, 0), _item("H2", 400, 800, 500, 860, 1),
            _item("H3", 700, 800, 800, 860, 2),
            _item("R1C1", 110, 950, 180, 1000, 3), _item("R1C2", 410, 950, 500, 1000, 4),
            _item("R1C3", 710, 950, 810, 1000, 5),
        ]
        rows = _detect_rows(items)
        bands = [(0, 266, ["H1"]), (266, 566, ["H2"]), (566, 2000, ["H3"])]
        matrix, un, sp, am, conf = _build_cell_matrix(items, rows, bands, 2000, 3000)
        assert len(matrix) >= 2
        assert conf > 0.5

    def test_spanning_items_tracked(self):
        bands = [(0, 200, []), (200, 400, []), (400, 600, []), (600, 800, []), (800, 2000, [])]
        rows = [[_item("W", 50, 100, 1900, 140, 0)]]
        items = [_item("W", 50, 100, 1900, 140, 0)]
        matrix, un, sp, am, conf = _build_cell_matrix(items, rows, bands, 2000, 3000)
        # Wide item that spans >=4 columns is unassigned (returned None by _assign_columns)
        assert len(un) >= 1 or len(sp) >= 1, "Spanning/unassigned items should be tracked"


# ═══════════════════════════════════════════════════════════════════════
# Fragment merging
# ═══════════════════════════════════════════════════════════════════════

class TestFragmentMerging:
    def test_merges_two_fragments_in_same_cell(self):
        items = [_item("0.1100", 100, 100, 180, 140, 0), _item("Ac.Gts", 110, 145, 190, 185, 1)]
        text = _merge_cell_text(items)
        assert "0.1100" in text
        assert "Ac.Gts" in text

    def test_preserves_original_text(self):
        items = [_item("SURvey", 100, 100, 180, 140, 0)]
        assert _merge_cell_text(items) == "SURvey"


# ═══════════════════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════════════════

class TestDetectTableRegion:
    def test_detects_simple_table(self):
        items = [
            _item("H1", 100, 800, 180, 860, 0), _item("H2", 400, 800, 500, 860, 1),
            _item("H3", 700, 800, 800, 860, 2), _item("H4", 1000, 800, 1100, 860, 3),
            _item("H5", 1300, 800, 1400, 860, 4),
            _item("D1C1", 110, 950, 180, 1000, 5), _item("D1C2", 410, 950, 500, 1000, 6),
            _item("D1C3", 710, 950, 810, 1000, 7), _item("D1C4", 1010, 950, 1110, 1000, 8),
            _item("D1C5", 1310, 950, 1410, 1000, 9),
        ]
        lines, non_table, meta = detect_table_region(items, 2000, 3000)
        assert meta is not None, "Should detect table"
        assert meta["logical_column_count"] >= 5

    def test_detect_table_region_keeps_item_line_ids_resolved(self):
        items = [
            _item("H1", 100, 800, 180, 860, 0), _item("H2", 400, 800, 500, 860, 1),
            _item("H3", 700, 800, 800, 860, 2), _item("H4", 1000, 800, 1100, 860, 3),
            _item("D1", 110, 950, 180, 1000, 4), _item("D2", 410, 950, 500, 1000, 5),
            _item("D3", 710, 950, 810, 1000, 6), _item("D4", 1010, 950, 1110, 1000, 7),
        ]
        lines, non_table, meta = detect_table_region(items, 2000, 3000)

        assert meta is not None
        line_ids = {line.line_id for line in lines}
        assert line_ids
        for item in items:
            assert item.line_id in line_ids
        assert all(line.line_id in line_ids for line in lines)

    def test_paragraph_not_detected_as_table(self):
        items = [_item("Single paragraph line", 50, 100, 1900, 140, 0)]
        lines, non_table, meta = detect_table_region(items, 2000, 3000)
        assert meta is None

    def test_no_mutation_of_input(self):
        import copy
        items = [_item("Test", 100, 800, 180, 860, 0), _item("Test2", 400, 800, 500, 860, 1),
                 _item("Test3", 700, 800, 800, 860, 2), _item("Test4", 1000, 800, 1100, 860, 3),
                 _item("D1C1", 110, 950, 180, 1000, 4), _item("D1C2", 410, 950, 500, 1000, 5)]
        original = copy.deepcopy([(it.text, it.bbox.x1, it.confidence) for it in items])
        detect_table_region(items, 2000, 3000)
        for i, it in enumerate(items):
            assert it.text == original[i][0]

    def test_single_data_row_still_works(self):
        items = [
            _item("H1", 100, 800, 180, 860, 0), _item("H2", 400, 800, 500, 860, 1),
            _item("H3", 700, 800, 800, 860, 2), _item("H4", 1000, 800, 1100, 860, 3),
            _item("H5", 1300, 800, 1400, 860, 4),
            _item("D1", 110, 950, 180, 1000, 5), _item("D2", 410, 950, 500, 1000, 6),
            _item("D3", 710, 950, 810, 1000, 7), _item("D4", 1010, 950, 1110, 1000, 8),
            _item("D5", 1310, 950, 1410, 1000, 9),
        ]
        lines, non_table, meta = detect_table_region(items, 2000, 3000)
        assert meta is not None

    def test_deterministic_output(self):
        items = [
            _item("A", 100, 800, 180, 860, 0), _item("B", 400, 800, 500, 860, 1),
            _item("C", 700, 800, 800, 860, 2), _item("D", 1000, 800, 1100, 860, 3),
            _item("E", 110, 950, 180, 1000, 4), _item("F", 410, 950, 500, 1000, 5),
            _item("G", 710, 950, 810, 1000, 6), _item("H", 1010, 950, 1110, 1000, 7),
        ]
        r1 = detect_table_region(items, 2000, 3000)
        r2 = detect_table_region(items, 2000, 3000)
        assert r1[2]["physical_column_count"] == r2[2]["physical_column_count"]

    def test_missing_cells_handled(self):
        items = [
            _item("H1", 100, 800, 180, 860, 0), _item("H2", 400, 800, 500, 860, 1),
            _item("H3", 700, 800, 800, 860, 2), _item("H4", 1000, 800, 1100, 860, 3),
            _item("D1C1", 110, 950, 180, 1000, 4), _item("D1C2", 410, 950, 500, 1000, 5),
            _item("D1C3", 710, 950, 810, 1000, 6), _item("D1C4", 1010, 950, 1110, 1000, 7),
            _item("D2C1", 110, 1050, 180, 1090, 8),
        ]
        lines, non_table, meta = detect_table_region(items, 2000, 3000)
        assert meta is not None

    def test_skewed_rows_tolerated(self):
        items = [
            _item("H1", 100, 800, 180, 860, 0), _item("H2", 402, 795, 502, 855, 1),
            _item("H3", 704, 805, 804, 865, 2), _item("H4", 1006, 790, 1106, 850, 3),
            _item("D1", 110, 950, 180, 1000, 4), _item("D2", 412, 948, 512, 1002, 5),
            _item("D3", 714, 952, 814, 1004, 6), _item("D4", 1016, 945, 1116, 998, 7),
        ]
        lines, non_table, meta = detect_table_region(items, 2000, 3000)
        assert meta is not None


class TestFallback:
    def test_no_table_with_insufficient_items(self):
        items = [_item("A", 100, 100, 160, 140, 0), _item("B", 300, 100, 360, 140, 1)]
        lines, non_table, meta = detect_table_region(items, 2000, 3000)
        assert meta is None

    def test_no_table_with_only_wide_items(self):
        items = [_item("Wide", 50, 100, 1950, 140, 0)]
        lines, non_table, meta = detect_table_region(items, 2000, 3000)
        assert meta is None


class TestGeometryRegression:
    def test_page3_geometry_table_reconstructs_expected_shape(self):
        fixture_path = Path(".data/geometry-investigation/page3_items.json")
        data = json.loads(fixture_path.read_text())
        items = [
            OCRItem(
                item_id=f"p3_i{i}",
                page_number=3,
                text=entry["text"],
                confidence=entry["confidence"],
                polygon=entry["polygon"],
                bbox=BBox(entry["x1"], entry["y1"], entry["x2"], entry["y2"]),
                normalized_bbox=BBox(entry["nx1"], entry["ny1"], entry["nx2"], entry["ny2"]),
            )
            for i, entry in enumerate(data["items"])
        ]

        lines, non_table, meta = detect_table_region(items, data["page_w"], data["page_h"])

        assert meta is not None
        assert meta["physical_column_count"] == 17
        assert meta["logical_column_count"] == 5
        assert len(meta["data_rows"]) == 11
        assert meta["table_items"] == 64
        assert meta["table_lines"] == 35
        assert not any(it.line_id and it.line_id not in {ln.line_id for ln in lines} for it in items)


class TestLineReferenceIntegrity:
    """Ensure no stale item→line references after table reconstruction.

    Regression: Madera page 6 had a right-column 'table' of 4 items whose
    line_ids were synced to table lines, but the original line reconstruction
    then rebuilt non-table lines, creating line_id mismatches.
    """

    def test_items_in_table_and_nontable_are_all_synced(self):
        """Mixed page: some items detected as table, others not.

        After table reconstruction, ALL items must reference an existing line.
        No item may hold a stale reference from the first reconstruction pass.
        """
        left_items = [
            _item("Left line 1", 100,  50, 600,  80, 0),
            _item("Left line 2", 100, 100, 600, 130, 1),
            _item("Left line 3", 100, 150, 600, 180, 2),
        ]
        right_items = [
            _item("Cell A1", 700,  50, 800,  80, 3),
            _item("Cell B1", 850,  50, 950,  80, 4),
            _item("Cell A2", 700, 100, 800, 130, 5),
            _item("Cell B2", 850, 100, 950, 130, 6),
            _item("Cell A3", 700, 150, 800, 180, 7),
            _item("Cell B3", 850, 150, 950, 180, 8),
        ]
        items = left_items + right_items
        lines, non_table, meta = detect_table_region(items, 1100, 300)

        line_ids = {ln.line_id for ln in lines}
        for it in items:
            if it.line_id is not None:
                assert it.line_id in line_ids, (
                    f"{it.item_id} references missing line {it.line_id}"
                )

    def test_table_detection_with_orphan_items_is_clean(self):
        """Table items must reference the correct reconstructed line."""
        header = [
            _item("Col1",  100, 50, 250, 80, 0),
            _item("Col2",  300, 50, 450, 80, 1),
        ]
        data = [
            _item("A", 100, 100, 250, 130, 2),
            _item("B", 300, 100, 450, 130, 3),
            _item("C", 100, 150, 250, 180, 4),
            _item("D", 300, 150, 450, 180, 5),
            _item("E", 100, 200, 250, 230, 6),
            _item("F", 300, 200, 450, 230, 7),
        ]
        items = header + data
        lines, non_table, meta = detect_table_region(items, 600, 300)
        line_ids = {ln.line_id for ln in lines}
        for it in items:
            assert it.line_id is None or it.line_id in line_ids

    def test_non_table_items_after_table_split_are_valid(self):
        """When table items are removed, non-table items get new valid lines."""
        non_table = [
            _item("Paragraph text one",  100,  50, 600,  80, 0),
            _item("Paragraph text two",  100, 100, 600, 130, 1),
        ]
        table_header = [
            _item("H1", 100, 200, 250, 230, 2),
            _item("H2", 300, 200, 450, 230, 3),
            _item("H3", 500, 200, 650, 230, 4),
        ]
        table_data = [
            _item("d1", 100, 250, 250, 280, 5),
            _item("d2", 300, 250, 450, 280, 6),
            _item("d3", 500, 250, 650, 280, 7),
        ]
        items = non_table + table_header + table_data
        lines, non_table, meta = detect_table_region(items, 800, 400)
        line_ids = {ln.line_id for ln in lines}
        for it in items:
            if it.line_id is not None:
                assert it.line_id in line_ids, (
                    f"Stale ref: {it.item_id} -> {it.line_id}"
                )
