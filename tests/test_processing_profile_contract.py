from dataclasses import replace

import pytest

from app.services.image_processing import (
    internal_profile_for_effective,
    resolve_processing_profile,
)
from app.services.ocr_cache import compute_request_fingerprint
from app.services.processing_policy import (
    effective_processing_policy,
    effective_processing_policy_fingerprint,
)


@pytest.mark.parametrize(
    ("normalized", "legacy_enabled", "legacy_name", "requested", "effective"),
    [
        (None, "false", None, "legacy:image_processing=false", "standard_auto"),
        (None, "false", "none", "legacy:image_processing_profile=none", "standard_auto"),
        (None, "false", "document_standard", "legacy:image_processing_profile=document_standard", "enhanced_scan_recovery"),
        (None, "true", None, "legacy:image_processing=true", "enhanced_scan_recovery"),
        ("standard_auto", "true", "document_standard", "standard_auto", "standard_auto"),
        ("enhanced_scan_recovery", "false", "none", "enhanced_scan_recovery", "enhanced_scan_recovery"),
    ],
)
def test_effective_profile_mapping(normalized, legacy_enabled, legacy_name, requested, effective):
    assert resolve_processing_profile(normalized, legacy_enabled, legacy_name) == (requested, effective)


@pytest.mark.parametrize("value", ["diagnostic", "", "unknown"])
def test_unsupported_normalized_profile_is_rejected(value):
    if value == "":
        with pytest.raises(ValueError):
            resolve_processing_profile(value, "false", "unknown")
    else:
        with pytest.raises(ValueError, match="Unsupported processing_profile"):
            resolve_processing_profile(value, "false", None)


def test_unsupported_legacy_profile_is_rejected():
    with pytest.raises(ValueError, match="Unsupported image processing profile"):
        resolve_processing_profile(None, "false", "diagnostic")


def test_normalized_profiles_map_to_existing_internal_pipeline():
    assert internal_profile_for_effective("standard_auto") == (False, "none")
    assert internal_profile_for_effective("enhanced_scan_recovery") == (True, "document_standard")


def test_legacy_aliases_share_cache_identity_when_effective_policy_matches():
    base = dict(
        file_hash="sha256:file",
        selected_pages=[1, 7],
        image_processing_enabled=False,
        image_processing_profile="standard_auto",
        pipeline_version="2",
        ocr_lang="en",
        engine_version="release-a",
        render_scale=1.4,
        effective_processing_policy_fingerprint=effective_processing_policy_fingerprint("standard_auto"),
    )
    assert compute_request_fingerprint(**base) == compute_request_fingerprint(**base)
    assert compute_request_fingerprint(
        **{**base, "effective_processing_policy_fingerprint": effective_processing_policy_fingerprint("enhanced_scan_recovery")}
    ) != compute_request_fingerprint(**base)


@pytest.mark.parametrize(
    "field",
    [
        "ocr_digital_pdf_min_characters",
        "ocr_blank_page_white_ratio_threshold",
        "ocr_deskew_min_confidence",
        "ocr_pdf_render_scale",
        "ocr_recovery_min_long_side",
    ],
)
def test_policy_fingerprint_changes_for_output_affecting_setting(monkeypatch, field):
    import app.services.processing_policy as policy_module

    original = policy_module.SETTINGS
    value = getattr(original, field)
    changed_value = value + 1 if isinstance(value, int) else value + 0.01
    monkeypatch.setattr(
        policy_module,
        "SETTINGS",
        replace(original, **{field: changed_value}),
    )
    changed = effective_processing_policy_fingerprint("standard_auto")
    monkeypatch.setattr(policy_module, "SETTINGS", original)
    assert changed != effective_processing_policy_fingerprint("standard_auto")


def test_policy_is_sorted_non_secret_and_profile_specific():
    policy = effective_processing_policy("standard_auto")
    assert policy["profile"] == "standard_auto"
    assert "document_standard" not in str(policy)
    assert "ocr_result_cache_dir" not in str(policy)
    assert effective_processing_policy_fingerprint("standard_auto") != effective_processing_policy_fingerprint(
        "enhanced_scan_recovery"
    )
