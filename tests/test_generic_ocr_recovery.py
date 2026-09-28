"""Regression tests for generic sparse-page OCR recovery."""

from app.core.config import SETTINGS
from app.core.ocr_execution import (
    _merge_recovery_items,
    _should_attempt_recovery,
)
from app.services.ocr.models import BBox, OCRItem, OCRPage


def _item(item_id: str, text: str, x: float, y: float, provenance: str = "baseline") -> OCRItem:
    polygon = [[x, y], [x + 100, y], [x + 100, y + 20], [x, y + 20]]
    return OCRItem(
        item_id=item_id,
        page_number=7,
        text=text,
        confidence=0.9,
        polygon=polygon,
        bbox=BBox(x, y, x + 100, y + 20),
        normalized_bbox=BBox(x / 1000, y / 1000, (x + 100) / 1000, (y + 20) / 1000),
        provenance=provenance,
    )


def test_sparse_low_resolution_page_triggers_generic_recovery():
    page = OCRPage(page_number=7, width=857, height=1109, items=[_item("p7_i0", "heading", 10, 10)])
    assert _should_attempt_recovery(
        __import__("numpy").zeros((1109, 857, 3), dtype="uint8"),
        page,
        enhance=False,
        rendering_profile="standard",
    ) is True


def test_dense_or_large_page_does_not_trigger_recovery():
    page = OCRPage(
        page_number=1,
        width=2000,
        height=3000,
        items=[_item(f"i{i}", "text", i % 50 * 10, i // 50 * 20) for i in range(200)],
    )
    assert _should_attempt_recovery(
        __import__("numpy").zeros((3000, 2000, 3), dtype="uint8"),
        page,
        enhance=False,
        rendering_profile="standard",
    ) is False


def test_recovery_merges_nonduplicates_without_replacing_baseline_geometry():
    baseline = [_item("p7_i0", "heading", 10, 10)]
    recovery = [
        _item("p7_i0", "heading", 10, 10, "baseline"),
        _item("p7_i1", "unclassified paragraph text", 10, 40, "baseline"),
    ]
    merged, added = _merge_recovery_items(baseline, recovery, 7)

    assert added == 1
    assert [item.item_id for item in merged] == ["p7_i0", "p7_r1"]
    assert merged[0].text == "heading"
    assert merged[0].bbox.to_list() == [10, 10, 110, 30]
    assert merged[1].text == "unclassified paragraph text"
    assert merged[1].provenance.startswith("recovery:")
    assert merged[1].polygon
