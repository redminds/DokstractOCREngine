import json

from app.core.ocr_execution import _build_ocr_run_manifest, _build_ocr_run_summary_for_outbox, _build_ocr_diagnostics_snapshot_for_outbox


def _response(*, recovery=False, warning=False):
    metrics = {"recovery": {"attempted": recovery, "added_items": 2 if recovery else 0}}
    if warning:
        metrics["stitched_warning"] = "page quality warning"
    return {
        "processing": {"execution": {"execution_id": "exec-1", "ocr_version": "pipe-1-exec"}},
        "pages": [{
            "page_number": 7,
            "confidence": 0.81,
            "items": [{"item_id": "p7-i1"}],
            "lines": [{"line_id": "p7-l1"}],
            "metrics": metrics,
        }],
    }


def test_manifest_contains_provenance_pages_fingerprints_and_stable_outcome():
    manifest = _build_ocr_run_manifest(
        response=_response(recovery=True),
        source_file_hash="sha256-source",
        selected_pages=[7],
        processing_profile="none",
        cache_mode="refresh",
        cache_outcome="refreshed",
        request_id="req-1",
        correlation_id="corr-1",
        pipeline_fingerprint="pipe-fp",
        recovery_policy_fingerprint="recovery-fp",
        rendering_fingerprint="render-fp",
    )

    assert manifest["manifest_version"] == 1
    assert manifest["engine_execution_id"] == "exec-1"
    assert manifest["source_file_hash"] == "sha256-source"
    assert manifest["selected_pages"] == [7]
    assert manifest["processed_pages"] == [7]
    assert manifest["cache"] == {"mode": "refresh", "outcome": "refreshed"}
    assert manifest["recovery"] == {"attempted": True, "pages": [7], "items_added": 2}
    assert manifest["quality"]["status"] == "recovered"
    assert manifest["quality"]["page_summaries"][0]["item_count"] == 1


def test_manifest_quality_status_uses_only_stable_values():
    allowed = {"passed", "recovered", "needs_review", "failed"}
    assert _build_ocr_run_manifest(
        response=_response(), source_file_hash="s", selected_pages=[7],
        processing_profile="none", cache_mode="reuse", cache_outcome="hit",
        request_id=None, correlation_id=None, pipeline_fingerprint="p",
        recovery_policy_fingerprint="r", rendering_fingerprint="g",
    )["quality"]["status"] == "passed"
    assert _build_ocr_run_manifest(
        response=_response(warning=True), source_file_hash="s", selected_pages=[7],
        processing_profile="none", cache_mode="bypass", cache_outcome="bypassed",
        request_id=None, correlation_id=None, pipeline_fingerprint="p",
        recovery_policy_fingerprint="r", rendering_fingerprint="g",
    )["quality"]["status"] in allowed


def test_outbox_summary_is_an_allow_listed_projection_of_terminal_manifest():
    manifest = _build_ocr_run_manifest(
        response=_response(recovery=True),
        source_file_hash="source-secret",
        selected_pages=[7, 8],
        processing_profile="document_standard",
        cache_mode="refresh",
        cache_outcome="refreshed",
        request_id="request-secret",
        correlation_id="correlation-secret",
        pipeline_fingerprint="pipeline-secret",
        recovery_policy_fingerprint="recovery-secret",
        rendering_fingerprint="rendering-secret",
    )

    summary = _build_ocr_run_summary_for_outbox(
        manifest=manifest,
        execution_status="success",
        engine_release="ocr-engine-test",
        completed_at="2026-09-28T10:00:00Z",
    )

    assert summary == {
        "schema_version": 1,
        "engine_execution_id": "exec-1",
        "ocr_version": "pipe-1-exec",
        "run_manifest_version": 1,
        "processing_profile": "document_standard",
        "cache": {"mode": "refresh", "outcome": "refreshed"},
        "pages": {"selected_count": 2, "processed_count": 1},
        "recovery": {"outcome": "recovered"},
        "quality": {"outcome": "recovered"},
        "engine_release": "ocr-engine-test",
        "completed_at": "2026-09-28T10:00:00Z",
    }

    forbidden = {
        "source_file_hash", "document_name", "document_text", "ocr_text", "labels",
        "extracted_values", "request_id", "correlation_id", "selected_pages",
        "processed_pages", "page_number", "page_numbers", "item_id", "line_id",
        "bbox", "polygon", "page_image", "rendering_fingerprint", "detector_fingerprint",
        "preprocessing_fingerprint", "pipeline_fingerprint", "recovery_policy_fingerprint",
        "model_fingerprint", "storage_path", "artifact_path", "url", "headers",
        "token", "credentials", "exception_trace",
    }

    def assert_safe(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for nested in value.values():
                assert_safe(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_safe(nested)

    assert_safe(summary)
    assert "source-secret" not in json.dumps(summary)


def test_diagnostics_snapshot_projects_page_health_without_customer_content():
    response = _response(recovery=True)
    response["pages"][0].update({"width": 2480, "height": 3508, "rotation": 0, "text": "PRIVATE OCR TEXT"})
    response["pages"][0]["metrics"]["render_dpi"] = 300
    response["pages"][0]["metrics"]["geometry_diagnostics"] = {
        "detected_region_count": 42,
        "recognized_region_count": 38,
        "mean_confidence": 0.7,
        "text_coverage_ratio": 0.12,
        "invalid_polygon_count": 1,
        "warning_codes": ["GEOMETRY_WARNING"],
        "quality_signals": ["GEOMETRY_WARNING"],
    }
    manifest = _build_ocr_run_manifest(
        response=response, source_file_hash="document-hash", selected_pages=[7],
        processing_profile="none", cache_mode="reuse", cache_outcome="hit",
        request_id="request", correlation_id="correlation", pipeline_fingerprint="p",
        recovery_policy_fingerprint="r", rendering_fingerprint="g",
    )
    snapshot = _build_ocr_diagnostics_snapshot_for_outbox(
        response=response, manifest=manifest, execution_status="success", engine_release="release-1",
    )
    assert snapshot["health"] == "DEGRADED"
    page = snapshot["page_diagnostics"][0]
    assert page["render_width"] == 2480
    assert page["geometry_warning_count"] == 1
    assert page["recovery_result"] == "SUCCESS"
    assert "PRIVATE OCR TEXT" not in json.dumps(snapshot)
    assert "text" not in snapshot


def test_outbox_summary_preserves_all_cache_modes_and_safe_outcomes():
    for mode, outcome in (("reuse", "hit"), ("reuse", "miss"), ("refresh", "refreshed"), ("bypass", "bypassed")):
        manifest = _build_ocr_run_manifest(
            response=_response(), source_file_hash="s", selected_pages=[7],
            processing_profile="none", cache_mode=mode, cache_outcome=outcome,
            request_id=None, correlation_id=None, pipeline_fingerprint="p",
            recovery_policy_fingerprint="r", rendering_fingerprint="g",
        )
        summary = _build_ocr_run_summary_for_outbox(
            manifest=manifest, execution_status="success",
            engine_release="release", completed_at="2026-09-28T10:00:00Z",
        )
        assert summary["cache"] == {"mode": mode, "outcome": outcome}


def test_outbox_summary_omits_incomplete_or_failed_terminal_events():
    manifest = _build_ocr_run_manifest(
        response=_response(), source_file_hash="s", selected_pages=[7],
        processing_profile="none", cache_mode="reuse", cache_outcome="miss",
        request_id=None, correlation_id=None, pipeline_fingerprint="p",
        recovery_policy_fingerprint="r", rendering_fingerprint="g",
    )
    assert _build_ocr_run_summary_for_outbox(
        manifest=manifest, execution_status="failed",
        engine_release="release", completed_at="2026-09-28T10:00:00Z",
    ) is None
    incomplete = dict(manifest)
    incomplete.pop("engine_ocr_version")
    assert _build_ocr_run_summary_for_outbox(
        manifest=incomplete, execution_status="success",
        engine_release="release", completed_at="2026-09-28T10:00:00Z",
    ) is None
