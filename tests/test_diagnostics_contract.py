from app.services.ocr.diagnostics import (
    OCRFailureCode,
    build_failure_bundle,
    build_ocr_health_summary,
    classify_failure,
    document_fingerprint,
)


def test_failure_bundle_is_sanitized_and_stable():
    error = RuntimeError("Tensor holds no memory for /customer/private.pdf")
    bundle = build_failure_bundle(
        execution_id="exec-1",
        document_hash=document_fingerprint(b"private document"),
        engine_release="ocr-test",
        pipeline_version="2",
        recovery_version="1",
        failure_stage="ocr_execution",
        exc=error,
        page_number=3,
    )
    assert bundle["failure_code"] == OCRFailureCode.PREDICTOR_STATE_CORRUPTION
    assert "private.pdf" not in str(bundle)
    assert "Tensor holds" not in str(bundle)
    assert bundle["sanitized_exception"] == {
        "type": "RuntimeError",
        "code": "PREDICTOR_STATE_CORRUPTION",
    }


def test_health_summary_contains_aggregate_signals_only():
    response = {
        "pages": [{
            "page_number": 1,
            "confidence": 0.91,
            "items": [{"item_id": "i1"}],
            "lines": [{"line_id": "l1"}],
            "metrics": {
                "geometry_diagnostics": {"quality_signals": ["LOW_CONFIDENCE"]},
                "recovery": {"attempted": True},
            },
        }],
    }
    summary = build_ocr_health_summary(
        response,
        execution_id="exec-1",
        engine_release="ocr-test",
        pipeline_version="2",
        recovery_version="1",
    )
    assert summary["status"] == "needs_review"
    assert summary["warning_codes"] == ["LOW_CONFIDENCE", "RECOVERY_USED"]
    assert summary["pages"][0]["item_count"] == 1
    assert "items" not in str(summary)


def test_failure_taxonomy_maps_primitive_and_render_errors():
    assert classify_failure(RuntimeError("could not execute a primitive")) == OCRFailureCode.PREDICTOR_RUNTIME_FAILURE
    assert classify_failure(ValueError("bad image"), "render") == OCRFailureCode.RENDER_FAILURE
