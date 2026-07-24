"""Filesystem-based OCR result cache with configurable TTL.

Stores completed canonical OCR responses as JSON files under a cache directory.
The cache directory should be on a Docker volume for durability across restarts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import SETTINGS

logger = logging.getLogger("dokstract.ocr_engine.cache")


@dataclass
class CachedOCRResult:
    cache_key: str
    status: str  # "completed"
    created_at: str
    expires_at: str
    engine_version: str
    request_fingerprint: str
    processing: dict[str, Any]
    response: dict[str, Any]


def _make_cache_dir() -> Path:
    path = Path(SETTINGS.ocr_result_cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_file_path(cache_key: str) -> Path:
    safe_key = hashlib.sha256(cache_key.encode()).hexdigest()[:64]
    return _make_cache_dir() / f"{safe_key}.json"


def compute_file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_request_fingerprint(
    file_hash: str,
    selected_pages: list[int],
    image_processing_enabled: bool,
    image_processing_profile: str,
    pipeline_version: str,
    ocr_lang: str,
    engine_version: str,
    render_scale: float,
) -> str:
    """Build a deterministic cache key from all output-affecting inputs."""
    normalised_pages = ",".join(str(p) for p in sorted(set(selected_pages)))
    components = [
        file_hash,
        normalised_pages,
        str(image_processing_enabled),
        image_processing_profile,
        pipeline_version,
        ocr_lang,
        engine_version,
        str(round(render_scale, 2)),
    ]
    raw = "|".join(components)
    return hashlib.sha256(raw.encode()).hexdigest()


def compute_page_fingerprint(
    file_hash: str,
    page_number: int,
    image_processing_enabled: bool,
    image_processing_profile: str,
    pipeline_version: str,
    ocr_lang: str,
    engine_version: str,
    render_scale: float,
) -> str:
    """Build a deterministic cache key for a single page's output.

    Must include every input that affects what a page produces:
    file hash, page number, OCR model/version, language, render scale,
    image-processing profile/version, digital-PDF path version,
    blank/low-content detector version, geometry pipeline version.

    All pipeline versions are captured through engine_version.
    """
    components = [
        file_hash,
        str(page_number),
        str(image_processing_enabled),
        image_processing_profile,
        pipeline_version,
        ocr_lang,
        engine_version,
        str(round(render_scale, 2)),
    ]
    raw = "|".join(components)
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached_result(fingerprint: str) -> CachedOCRResult | None:
    """Retrieve a cached OCR result if it exists and is not expired."""
    if not SETTINGS.ocr_result_cache_enabled:
        return None
    try:
        path = _cache_file_path(fingerprint)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        expires_at = data.get("expires_at", "")
        if expires_at:
            expiry = _parse_iso(expires_at)
            if expiry and time.time() > expiry:
                logger.debug("Cache entry expired: %s", fingerprint[:16])
                _safe_delete(path)
                return None
        return CachedOCRResult(
            cache_key=data.get("cache_key", fingerprint),
            status=data.get("status", "completed"),
            created_at=data.get("created_at", ""),
            expires_at=expires_at,
            engine_version=data.get("engine_version", ""),
            request_fingerprint=data.get("request_fingerprint", ""),
            processing=data.get("processing", {}),
            response=data.get("response", {}),
        )
    except Exception:
        logger.warning("Failed to read cache entry", exc_info=True)
        return None


def put_cached_result(fingerprint: str, result: CachedOCRResult) -> bool:
    """Write a completed OCR result to the cache. Returns True on success."""
    if not SETTINGS.ocr_result_cache_enabled:
        return False
    try:
        path = _cache_file_path(fingerprint)
        data = {
            "cache_key": result.cache_key,
            "status": result.status,
            "created_at": result.created_at,
            "expires_at": result.expires_at,
            "engine_version": result.engine_version,
            "request_fingerprint": result.request_fingerprint,
            "processing": result.processing,
            "response": result.response,
        }
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"), default=str)
        os.replace(tmp_path, path)
        logger.info("Cached OCR result: key=%s", fingerprint[:16])
        return True
    except Exception:
        logger.warning("Failed to write cache entry", exc_info=True)
        return False


def delete_cached_result(fingerprint: str) -> None:
    path = _cache_file_path(fingerprint)
    _safe_delete(path)


def _safe_delete(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _parse_iso(ts: str) -> float | None:
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def sweep_stale_temp() -> int:
    """Remove stale OCR temporary directories. Returns count removed."""
    if not SETTINGS.ocr_temp_cleanup_enabled:
        return 0
    try:
        root = Path(SETTINGS.ocr_temp_root)
        if not root.exists():
            return 0
        cutoff = time.time() - SETTINGS.ocr_temp_stale_after_seconds
        removed = 0
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            if not entry.name.startswith("dokstract-ocr-"):
                continue
            try:
                mtime = entry.stat().st_mtime
                if mtime < cutoff:
                    _rmtree(entry)
                    removed += 1
            except Exception:
                pass
        if removed:
            logger.info("Cleaned up %d stale temp directories", removed)
        return removed
    except Exception:
        logger.warning("Stale temp sweep failed", exc_info=True)
        return 0


def _rmtree(path: Path) -> None:
    import shutil
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


# ── Page-level cache ────────────────────────────────────────────────────

def _page_cache_dir() -> Path:
    path = Path(SETTINGS.ocr_result_cache_dir) / "pages"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _page_cache_path(file_hash: str, page_number: int, pipeline_fp: str) -> Path:
    raw = f"{file_hash}:{page_number}:{pipeline_fp}"
    safe = hashlib.sha256(raw.encode()).hexdigest()[:64]
    return _page_cache_dir() / f"{safe}.json"


def get_cached_page(file_hash: str, page_number: int, pipeline_fp: str) -> dict | None:
    """Retrieve a cached page result."""
    if not SETTINGS.ocr_result_cache_enabled:
        return None
    try:
        path = _page_cache_path(file_hash, page_number, pipeline_fp)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        expires_at = data.get("expires_at", "")
        if expires_at:
            expiry = _parse_iso(expires_at)
            if expiry and time.time() > expiry:
                _safe_delete(path)
                return None
        return data.get("page_data")
    except Exception:
        return None


def put_cached_page(file_hash: str, page_number: int, pipeline_fp: str, page_data: dict, ttl: int | None = None) -> bool:
    """Write a completed page result to cache."""
    if not SETTINGS.ocr_result_cache_enabled:
        return False
    try:
        ttl = ttl or SETTINGS.ocr_result_cache_ttl_seconds
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=ttl)
        data = {
            "page_number": page_number,
            "expires_at": expires.isoformat(),
            "page_data": page_data,
        }
        path = _page_cache_path(file_hash, page_number, pipeline_fp)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"), default=str)
        os.replace(tmp, path)
        return True
    except Exception:
        logger.warning("Failed to write page cache for page %d", page_number, exc_info=True)
        return False
