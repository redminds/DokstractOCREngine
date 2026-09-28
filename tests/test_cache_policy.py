from app.services.ocr_cache import (
    cache_mode_allows_read,
    cache_mode_allows_write,
    compute_page_fingerprint,
    compute_request_fingerprint,
)


def test_cache_modes_have_unambiguous_read_write_semantics():
    assert cache_mode_allows_read("reuse") is True
    assert cache_mode_allows_write("reuse") is True
    assert cache_mode_allows_read("refresh") is False
    assert cache_mode_allows_write("refresh") is True
    assert cache_mode_allows_read("bypass") is False
    assert cache_mode_allows_write("bypass") is False


def test_cache_identity_includes_scope_profile_and_recovery_policy():
    kwargs = dict(
        file_hash="sha256:file",
        selected_pages=[1, 7],
        image_processing_enabled=False,
        image_processing_profile="none",
        pipeline_version="2",
        ocr_lang="en",
        engine_version="release-a",
        render_scale=1.4,
    )
    assert compute_request_fingerprint(**kwargs) != compute_request_fingerprint(**{**kwargs, "selected_pages": [1]})
    assert compute_request_fingerprint(**kwargs) != compute_request_fingerprint(**{**kwargs, "image_processing_profile": "document_standard"})
    assert compute_request_fingerprint(**kwargs) != compute_request_fingerprint(**{**kwargs, "recovery_policy_version": "2"})
    assert compute_request_fingerprint(**kwargs) != compute_request_fingerprint(**{**kwargs, "rendering_config_fingerprint": "dpi=240"})
    page_kwargs = {k: v for k, v in kwargs.items() if k != "selected_pages"}
    assert compute_page_fingerprint(**page_kwargs, page_number=7) != compute_page_fingerprint(**page_kwargs, page_number=8)
