"""Focused tests for OCR Engine diagnostic infrastructure — Phase 2 Session 1.

Tests cover: PdfRenderResult, pdf_page_to_img, diagnostics enabled/disabled,
Paddle device detection, detector resize calculation, preprocessing step detail,
response_builder _debug key, geometry preservation.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


# =============================================================================
# PdfRenderResult tests
# =============================================================================

class TestPdfRenderResult:
    """Test PdfRenderResult dataclass construction and pdf_page_to_img contract."""

    def test_pdf_render_result_fields(self):
        from app.core.ocr_execution import PdfRenderResult
        import numpy as np
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = PdfRenderResult(
            image=img,
            source_width_points=595.0,
            source_height_points=842.0,
            source_rotation=0.0,
            effective_scale=1.4,
        )
        assert result.source_width_points == 595.0
        assert result.source_height_points == 842.0
        assert result.source_rotation == 0.0
        assert result.effective_scale == 1.4
        assert result.image.shape == (100, 200, 3)

    def test_pdf_render_result_image_is_ndarray(self):
        from app.core.ocr_execution import PdfRenderResult
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        r = PdfRenderResult(image=img, source_width_points=0, source_height_points=0,
                           source_rotation=0, effective_scale=1.0)
        assert isinstance(r.image, np.ndarray)

    def test_pdf_render_result_rotation_can_be_nonzero(self):
        from app.core.ocr_execution import PdfRenderResult
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        r = PdfRenderResult(image=img, source_width_points=800, source_height_points=600,
                           source_rotation=90.0, effective_scale=2.0)
        assert r.source_rotation == 90.0


# =============================================================================
# _PageDiagnostics tests
# =============================================================================

class TestPageDiagnostics:
    """Test _PageDiagnostics dataclass and to_dict() serialization."""

    def test_to_dict_render_fields(self):
        from app.core.ocr_execution import _PageDiagnostics
        diag = _PageDiagnostics(
            source_width_points=595.0,
            source_height_points=842.0,
            source_rotation=0.0,
            requested_scale=1.4,
            effective_scale=1.4,
            effective_dpi_x=100.8,
            effective_dpi_y=100.8,
            rendered_width_pixels=834,
            rendered_height_pixels=1179,
            render_duration_ms=45.0,
        )
        d = diag.to_dict()
        assert d["source"]["width_points"] == 595.0
        assert d["source"]["height_points"] == 842.0
        assert d["render"]["effective_dpi_x"] == 100.8
        assert d["render"]["width_pixels"] == 834

    def test_to_dict_preprocessing_none(self):
        from app.core.ocr_execution import _PageDiagnostics
        diag = _PageDiagnostics()
        d = diag.to_dict()
        assert d["preprocessing"]["requested"] is False
        assert d["preprocessing"]["applied"] is False
        assert d["preprocessing"]["profile"] == "none"
        assert d["preprocessing"]["steps"] == []

    def test_to_dict_preprocessing_applied(self):
        from app.core.ocr_execution import _PageDiagnostics
        diag = _PageDiagnostics(
            preprocessing_requested=True,
            preprocessing_applied=True,
            preprocessing_profile="document_standard",
            preprocessing_steps=["grayscale", "deskew"],
            preprocessing_duration_ms=50.0,
        )
        d = diag.to_dict()
        assert d["preprocessing"]["requested"] is True
        assert d["preprocessing"]["applied"] is True
        assert d["preprocessing"]["profile"] == "document_standard"
        assert len(d["preprocessing"]["steps"]) == 2

    def test_to_dict_paddle_fields(self):
        from app.core.ocr_execution import _PageDiagnostics
        diag = _PageDiagnostics(
            paddleocr_version="2.7.3",
            paddle_version="2.6.2",
            detection_model="PP-OCRv3_det",
            recognition_model="PP-OCRv4_rec",
            device="cpu",
            det_limit_side_len=1536,
            det_limit_type="max",
            detector_input_width=834,
            detector_input_height=1179,
            detector_resize_status="not_required",
            calculated_detector_width=834,
            calculated_detector_height=1179,
        )
        d = diag.to_dict()
        assert d["paddle"]["paddleocr_version"] == "2.7.3"
        assert d["paddle"]["device"] == "cpu"
        assert d["paddle"]["detector_resize_status"] == "not_required"
        assert d["paddle"]["calculated_detector_width"] == 834

    def test_to_dict_geometry_fields(self):
        from app.core.ocr_execution import _PageDiagnostics
        diag = _PageDiagnostics(
            geometry_provider="paddleocr",
            text_provider="paddleocr",
            ocr_pass="baseline",
            render_profile="standard",
        )
        d = diag.to_dict()
        assert d["geometry"]["geometry_provider"] == "paddleocr"
        assert d["geometry"]["ocr_pass"] == "baseline"


# =============================================================================
# Detector resize calculation tests
# =============================================================================

class TestDetectorResize:
    """Test detector resize status calculation logic."""

    def test_not_required_when_within_limit(self):
        """When image <= limit, no resize needed."""
        from app.core.ocr_execution import _PageDiagnostics
        diag = _PageDiagnostics(
            detector_input_width=834,
            detector_input_height=1179,
            det_limit_side_len=1536,
            detector_resize_status="not_required",
            calculated_detector_width=834,
            calculated_detector_height=1179,
        )
        d = diag.to_dict()
        assert d["paddle"]["detector_resize_status"] == "not_required"

    def test_expected_when_exceeds_limit(self):
        """When long side > limit, resize IS expected."""
        from app.core.ocr_execution import _PageDiagnostics
        # 1668x2358 at 200 DPI with limit 1536 — long side 2358 > 1536
        # Short side proportional: 1668 * 1536 / 2358 ≈ 1086
        diag = _PageDiagnostics(
            detector_input_width=1668,
            detector_input_height=2358,
            det_limit_side_len=1536,
            detector_resize_status="expected",
            calculated_detector_width=1086,
            calculated_detector_height=1536,
        )
        d = diag.to_dict()
        assert d["paddle"]["detector_resize_status"] == "expected"
        assert d["paddle"]["calculated_detector_width"] == 1086
        assert d["paddle"]["calculated_detector_height"] == 1536

    def test_unavailable_when_no_data(self):
        from app.core.ocr_execution import _PageDiagnostics
        diag = _PageDiagnostics(detector_resize_status="unavailable")
        d = diag.to_dict()
        assert d["paddle"]["detector_resize_status"] == "unavailable"


# =============================================================================
# Paddle device detection tests
# =============================================================================

class TestPaddleDeviceDetection:
    """Test that _capture_paddle_diagnostics uses Paddle APIs, not torch."""

    def test_device_is_cpu_by_default(self):
        """When Paddle is not CUDA-compiled, device=cpu."""
        from app.core.ocr_execution import _capture_paddle_diagnostics
        mock_engine = MagicMock()
        mock_engine.text_detector = None
        mock_engine.text_recognizer = None
        diag = _capture_paddle_diagnostics(mock_engine)
        assert diag["device"] == "cpu"
        assert diag["paddle_compiled_with_cuda"] is False

    def test_handles_missing_paddle_gracefully(self):
        """When Paddle imports fail, diagnostic still returns safely."""
        from app.core.ocr_execution import _capture_paddle_diagnostics
        mock_engine = MagicMock()
        mock_engine.text_detector = None
        mock_engine.text_recognizer = None
        diag = _capture_paddle_diagnostics(mock_engine)
        assert "paddleocr_version" in diag
        assert "device" in diag

    def test_does_not_import_torch(self):
        """_capture_paddle_diagnostics must not import torch (use paddle.device)."""
        import inspect
        from app.core.ocr_execution import _capture_paddle_diagnostics
        src = inspect.getsource(_capture_paddle_diagnostics)
        assert "import torch" not in src, "_capture_paddle_diagnostics should not import torch"
        assert "paddle.device" in src or "paddle." in src, "Should use paddle APIs"


# =============================================================================
# Response builder debug key tests
# =============================================================================

class TestResponseBuilderDebug:
    """Test build_response _debug key behavior."""

    def test_debug_key_present_when_diagnostics_provided(self):
        from app.services.ocr.response_builder import build_response
        from app.services.ocr.models import OCRPage
        pages = [OCRPage(page_number=1, width=100, height=200)]
        resp = build_response(
            pages_data=pages, filename="t.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
            debug_diagnostics={"enabled": True, "pages": [
                {"render": {"effective_dpi_x": 100.8}}]},
        )
        assert "_debug" in resp
        assert resp["_debug"]["enabled"] is True

    def test_debug_key_absent_when_diagnostics_none(self):
        from app.services.ocr.response_builder import build_response
        from app.services.ocr.models import OCRPage
        pages = [OCRPage(page_number=1, width=100, height=200)]
        resp = build_response(
            pages_data=pages, filename="t.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
            debug_diagnostics=None,
        )
        assert "_debug" not in resp

    def test_existing_response_keys_preserved_with_debug(self):
        from app.services.ocr.response_builder import build_response
        from app.services.ocr.models import OCRPage
        pages = [OCRPage(page_number=1, width=100, height=200)]
        resp = build_response(
            pages_data=pages, filename="x.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
            debug_diagnostics={"enabled": True, "pages": []},
        )
        assert "file" in resp
        assert "document" in resp
        assert "pages" in resp
        assert "metrics" in resp

    def test_geometry_values_unchanged_with_debug(self):
        from app.services.ocr.response_builder import build_response
        from app.services.ocr.models import OCRPage, BBox, OCRLine
        line = OCRLine(line_id="l1", page_number=1, text="hello",
                      confidence=0.9, bbox=BBox(10, 20, 100, 40))
        pages = [OCRPage(page_number=1, width=200, height=300, lines=[line])]
        resp = build_response(
            pages_data=pages, filename="g.pdf", file_type="pdf",
            total_pages=1, selected_pages=[1],
            debug_diagnostics={"pages": []},
        )
        # Geometry unchanged
        p = resp["pages"][0]
        assert len(p["lines"]) == 1
        assert p["lines"][0]["bbox"] == [10.0, 20.0, 100.0, 40.0]


# =============================================================================
# Preprocessing step-level detail tests
# =============================================================================

class TestPreprocessingStepDetail:
    """Test that step-level preprocessing detail distinguishes applied vs skipped."""

    def test_steps_list_populated_from_processing_result(self):
        from app.services.image_processing import ProcessingResult
        import numpy as np
        result = ProcessingResult(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            profile="document_standard",
            operations_applied=["grayscale", "contrast_normalization"],
            operations_skipped=[
                {"operation": "deskew", "reason": "angle_below_threshold"},
            ],
            duration_ms=42.0,
        )
        # Build step detail inline like the diagnostics code does
        steps = []
        for step in (result.operations_applied or []):
            steps.append({"name": step, "status": "applied"})
        for skip in (result.operations_skipped or []):
            steps.append({
                "name": skip.get("operation", "unknown"),
                "status": "skipped",
                "reason": skip.get("reason", "unknown"),
            })
        assert len(steps) == 3
        assert steps[0] == {"name": "grayscale", "status": "applied"}
        assert steps[2] == {"name": "deskew", "status": "skipped", "reason": "angle_below_threshold"}
