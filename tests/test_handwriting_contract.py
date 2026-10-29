from __future__ import annotations

import pytest

from app.services.ocr.handwriting import (
    HANDWRITING_CONTRACT_VERSION,
    DisabledHandwritingRecognizer,
    HandwritingRequest,
    get_handwriting_recognizer,
    HandwritingCandidate,
    candidates_to_ocr_items,
)


def test_handwriting_is_disabled_without_model_dependency():
    recognizer = get_handwriting_recognizer(enabled=False)

    assert isinstance(recognizer, DisabledHandwritingRecognizer)
    assert recognizer.recognizer_id == "disabled"
    assert recognizer.contract_version == HANDWRITING_CONTRACT_VERSION
    assert recognizer.recognize(
        HandwritingRequest(page_number=1, image=object(), page_width=100, page_height=200)
    ) == []


def test_enabled_handwriting_fails_closed_until_adapter_is_accepted():
    with pytest.raises(RuntimeError, match="model path"):
        get_handwriting_recognizer(enabled=True)


def test_enabled_handwriting_requires_a_model_path():
    with pytest.raises(RuntimeError, match="model path"):
        get_handwriting_recognizer(enabled=True)


def test_handwriting_candidates_map_to_canonical_geometry_and_provenance():
    items = candidates_to_ocr_items([
        HandwritingCandidate(
            text="written",
            confidence=0.8,
            polygon=((10, 20), (60, 20), (60, 40), (10, 40)),
            source="test-recognizer",
        )
    ], page_number=2, page_width=100, page_height=100)

    assert items[0].item_id == "p2_h0"
    assert items[0].normalized_bbox.x1 == 0.1
    assert items[0].provenance == "handwriting:test-recognizer"
