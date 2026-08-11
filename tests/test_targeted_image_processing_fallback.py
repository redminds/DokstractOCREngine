from __future__ import annotations

import numpy as np

from app.core import ocr_execution
from app.services.ocr.models import OCRPage


def _make_page(text: str, confidence: float) -> OCRPage:
    return OCRPage(page_number=1, width=100.0, height=100.0, text=text, confidence=confidence)


def test_targeted_fallback_skips_when_quality_is_good(monkeypatch):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    calls: list[bool] = []

    def fake_build_results_for_image(*args, **kwargs):
        calls.append(bool(kwargs["enhance"]))
        return _make_page("clear survey text with enough length", 0.91), {"ocr_inference_ms": 1.0}, None

    monkeypatch.setattr(ocr_execution, "_build_results_for_image", fake_build_results_for_image)

    page, timings, proc_result, fallback_meta = ocr_execution._run_page_with_targeted_fallback(
        image,
        page=1,
        enhance=False,
        profile_name="none",
        rendering_profile="standard",
    )

    assert calls == [False]
    assert page.text == "clear survey text with enough length"
    assert timings == {"ocr_inference_ms": 1.0}
    assert proc_result is None
    assert fallback_meta is None


def test_targeted_fallback_accepts_better_enhanced_result(monkeypatch):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    calls: list[bool] = []

    def fake_build_results_for_image(*args, **kwargs):
        enhance = bool(kwargs["enhance"])
        calls.append(enhance)
        if enhance:
            return _make_page("Sub-Division No.21/1", 0.86), {"ocr_inference_ms": 2.0}, None
        return _make_page("06", 0.18), {"ocr_inference_ms": 1.0}, None

    monkeypatch.setattr(ocr_execution, "_build_results_for_image", fake_build_results_for_image)

    page, timings, proc_result, fallback_meta = ocr_execution._run_page_with_targeted_fallback(
        image,
        page=1,
        enhance=False,
        profile_name="none",
        rendering_profile="low_content",
    )

    assert calls == [False, True]
    assert page.text == "Sub-Division No.21/1"
    assert timings == {"ocr_inference_ms": 2.0}
    assert proc_result is None
    assert fallback_meta is not None
    assert fallback_meta["applied"] is True
    assert fallback_meta["fallback"]["profile"] in {"document_standard", "none"}


def test_targeted_fallback_keeps_original_when_retry_is_not_better(monkeypatch):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    calls: list[bool] = []

    def fake_build_results_for_image(*args, **kwargs):
        enhance = bool(kwargs["enhance"])
        calls.append(enhance)
        if enhance:
            return _make_page("06", 0.20), {"ocr_inference_ms": 2.0}, None
        return _make_page("06", 0.19), {"ocr_inference_ms": 1.0}, None

    monkeypatch.setattr(ocr_execution, "_build_results_for_image", fake_build_results_for_image)

    page, timings, proc_result, fallback_meta = ocr_execution._run_page_with_targeted_fallback(
        image,
        page=1,
        enhance=False,
        profile_name="none",
        rendering_profile="low_content",
    )

    assert calls == [False, True]
    assert page.text == "06"
    assert timings == {"ocr_inference_ms": 1.0}
    assert proc_result is None
    assert fallback_meta is None
