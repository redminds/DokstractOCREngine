from app.services.ocr.quality_assessment import build_quality_assessment


def _page(number=1, *, source=None, recovery=None, warning=None, rendering_profile=None):
    metrics = {}
    if source:
        metrics["source"] = source
    if recovery is not None:
        metrics["recovery"] = recovery
    if warning:
        metrics["stitched_warning"] = True
    if rendering_profile:
        metrics["rendering_profile"] = rendering_profile
    return {
        "page_number": number,
        "confidence": 0.42,
        "items": [{"item_id": f"p{number}-i1"}],
        "lines": [],
        "metrics": metrics,
    }


def test_recovery_is_recovered_only_when_evidence_was_added():
    assessment = build_quality_assessment(
        {"pages": [_page(recovery={"attempted": True, "added_items": 2})]},
        selected_pages=[1],
    )
    assert assessment["version"] == 1
    assert assessment["outcome"] == "recovered"
    assert assessment["reasons"] == [{
        "code": "recovery_added_items",
        "severity": "info",
        "page_numbers": [1],
        "safe_summary": "Generic OCR recovery added additional OCR evidence.",
    }]


def test_failure_and_warning_precede_recovery_and_have_reasons():
    assessment = build_quality_assessment(
        {"pages": [
            _page(1, source="error"),
            _page(2, recovery={"attempted": True, "added_items": 0}, warning=True),
        ]},
        selected_pages=[1, 2, 3],
    )
    assert assessment["outcome"] == "failed"
    assert {reason["code"] for reason in assessment["reasons"]} == {
        "page_processing_failed",
        "recovery_attempted_no_items",
        "stitched_page_warning",
        "selected_page_processing_incomplete",
    }


def test_blank_and_digital_pages_are_informational_not_quality_failures():
    blank = build_quality_assessment({"pages": [_page(source="blank")]}, selected_pages=[1])
    digital = build_quality_assessment({"pages": [_page(rendering_profile="digital")]}, selected_pages=[1])
    assert blank["outcome"] == "passed"
    assert digital["outcome"] == "passed"
    assert blank["reasons"][0]["severity"] == "info"
    assert digital["reasons"][0]["severity"] == "info"


def test_low_confidence_alone_is_not_called_inaccurate_without_a_calibrated_signal():
    assessment = build_quality_assessment(
        {"pages": [_page()]}, selected_pages=[1]
    )
    assert assessment["outcome"] == "passed"
    assert not any(reason["severity"] == "warning" for reason in assessment["reasons"])
