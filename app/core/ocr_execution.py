from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np

from app.core.config import SETTINGS

logger = logging.getLogger("dokstract.ocr_engine.execution")


@dataclass(frozen=True)
class OCRLineResult:
    text: str
    accuracy: float


class OCRDependencyUnavailable(RuntimeError):
    pass


_OCR_ENGINE = None


def _get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # pragma: no cover - environment-specific
            raise OCRDependencyUnavailable(f"OCR engine unavailable: {exc}") from exc

        try:
            _OCR_ENGINE = PaddleOCR(
                lang=SETTINGS.ocr_lang,
                use_angle_cls=False,
                det_limit_side_len=SETTINGS.ocr_det_limit_side_len,
                det_limit_type="max",
                use_dilation=True,
                text_recognition_batch_size=SETTINGS.ocr_text_batch_size,
                cpu_threads=SETTINGS.ocr_cpu_threads,
            )
        except Exception as exc:  # pragma: no cover - environment-specific
            raise OCRDependencyUnavailable(f"OCR engine unavailable: {exc}") from exc
    return _OCR_ENGINE


def normalize_text(lines: list[str]) -> str:
    seen: set[str] = set()
    clean: list[str] = []

    for line in lines:
        value = line.strip()
        if not value or len(value) < 2:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append(value)

    merged: list[str] = []
    buffer = ""
    for line in clean:
        if not buffer:
            buffer = line
            continue
        if line[0].islower() or len(buffer.split()) < 3:
            buffer += " " + line
        else:
            merged.append(buffer)
            buffer = line

    if buffer:
        merged.append(buffer)

    return "\n".join(merged).strip()


def run_ocr(image: np.ndarray, enhance: bool) -> OCRLineResult:
    if enhance:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    result = _get_ocr_engine().ocr(image, cls=False)
    boxes = []
    scores = []

    if result:
        for line in result:
            if not line:
                continue
            for word in line:
                if not word or len(word) < 2 or not word[1] or len(word[1]) < 2:
                    continue
                boxes.append(word)
                scores.append(word[1][1] * 100)

    boxes = sorted(boxes, key=lambda item: (item[0][0][1], item[0][0][0]))
    lines = [word[1][0] for word in boxes if word[1][0].strip()]
    text = normalize_text(lines)
    accuracy = round(sum(scores) / len(scores), 2) if scores else 0
    return OCRLineResult(text=text, accuracy=accuracy)


@dataclass(frozen=True)
class PdfSelection:
    page_numbers: list[int]
    total_pages: int
    defaulted_to_first_pages: bool


def open_pdf_from_bytes(pdf_bytes: bytes) -> fitz.Document:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError("Invalid or corrupted PDF") from exc

    if getattr(doc, "is_encrypted", False):
        doc.close()
        raise ValueError("Encrypted PDF files are not supported")
    if doc.page_count <= 0:
        doc.close()
        raise ValueError("PDF has no pages")
    return doc


def parse_pages(pages_str: str) -> list[int]:
    pages = set()
    parts = pages_str.split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
            except ValueError as exc:
                raise ValueError("Invalid page range format") from exc

            if start > end:
                raise ValueError("Page range start must be <= end")

            for page in range(start, end + 1):
                pages.add(page)
        else:
            try:
                pages.add(int(part))
            except ValueError as exc:
                raise ValueError("Invalid page number") from exc

    return sorted(pages)


def select_pages(total_pages: int, pages_str: str | None, max_pages: int) -> PdfSelection:
    if total_pages <= 0:
        raise ValueError("PDF has no pages")

    if pages_str and pages_str.strip():
        page_numbers = parse_pages(pages_str)
        defaulted = False
    else:
        page_numbers = list(range(1, min(total_pages, max_pages) + 1))
        defaulted = total_pages > max_pages

    if len(page_numbers) > max_pages:
        raise ValueError(f"Maximum {max_pages} pages allowed per request")

    for page in page_numbers:
        if page < 1 or page > total_pages:
            raise ValueError(f"Invalid page {page}. PDF has {total_pages} pages")

    return PdfSelection(
        page_numbers=page_numbers,
        total_pages=total_pages,
        defaulted_to_first_pages=defaulted,
    )


def enforce_pdf_page_budget(total_pages: int, selected_pages: int, max_pages_per_file: int, max_pages_per_request: int) -> None:
    if total_pages > max_pages_per_file:
        raise ValueError(f"Maximum {max_pages_per_file} pages allowed per file")
    if selected_pages > max_pages_per_request:
        raise ValueError(f"Maximum {max_pages_per_request} pages allowed per request")


def extract_pdf_text(path: str, page_numbers: list[int] | None = None) -> str:
    doc = fitz.open(path)
    texts: list[str] = []
    try:
        for index, page in enumerate(doc, start=1):
            if page_numbers and index not in page_numbers:
                continue
            text = page.get_text().strip()
            if text:
                texts.append(text)
    finally:
        doc.close()
    return "\n".join(texts)


def pdf_page_to_img(page: Any, scale: float):
    max_dpi = max(72, int(SETTINGS.ocr_max_render_dpi))
    scale = min(scale, max_dpi / 72.0)
    rect = page.rect
    width = int(rect.width * scale)
    height = int(rect.height * scale)
    if width <= 0 or height <= 0:
        raise ValueError("Invalid PDF page dimensions")
    if width > SETTINGS.ocr_max_image_width or height > SETTINGS.ocr_max_image_height:
        raise ValueError("Rendered PDF page exceeds image dimension limits")
    if width * height > SETTINGS.ocr_max_image_pixels:
        raise ValueError("Rendered PDF page exceeds pixel limits")

    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    decoded = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), 1)
    if decoded is None:
        raise ValueError("Rendered PDF page could not be decoded")
    if decoded.shape[1] > SETTINGS.ocr_max_image_width or decoded.shape[0] > SETTINGS.ocr_max_image_height:
        raise ValueError("Rendered PDF page exceeds image dimension limits")
    if decoded.shape[0] * decoded.shape[1] > SETTINGS.ocr_max_image_pixels:
        raise ValueError("Rendered PDF page exceeds pixel limits")
    if decoded.nbytes > SETTINGS.ocr_max_decoded_image_bytes:
        raise ValueError("Rendered PDF page exceeds decoded memory limits")
    return decoded


def is_digital_pdf_text(text: str, threshold: int) -> bool:
    return len(text.strip()) > threshold


def _build_results_for_image(image: np.ndarray, *, page: int, enhance: bool) -> tuple[list[dict[str, Any]], float]:
    try:
        result = run_ocr(image, enhance)
    except OCRDependencyUnavailable:
        raise

    results: list[dict[str, Any]] = []
    if result.text:
        for text in result.text.split("\n"):
            cleaned = text.strip()
            if cleaned:
                results.append({"page": page, "text": cleaned})
    return results, float(result.accuracy)


def extract_internal_ocr_document(
    *,
    file_bytes: bytes,
    filename: str,
    content_type: str | None,
    image_processing: str = "false",
    pages: str | None = None,
) -> dict[str, Any]:
    filename = filename or "uploaded-file"
    lower_name = filename.lower()
    file_type = "pdf" if lower_name.endswith(".pdf") else "image" if (content_type or "").startswith("image/") else None
    if file_type is None:
        raise ValueError("Unsupported file type")

    enhance = image_processing.lower() == "true"
    with tempfile.TemporaryDirectory() as tmp_dir:
        safe_filename = Path(filename).name
        file_path = Path(tmp_dir) / safe_filename
        file_path.write_bytes(file_bytes)

        results: list[dict[str, Any]] = []
        all_confidences: list[float] = []
        page_count = 1

        if file_type == "pdf":
            doc = open_pdf_from_bytes(file_bytes)
            try:
                page_count = len(doc)
                selection = select_pages(
                    page_count,
                    pages,
                    int(SETTINGS.ocr_max_pdf_pages_per_request),
                )
                enforce_pdf_page_budget(
                    total_pages=page_count,
                    selected_pages=len(selection.page_numbers),
                    max_pages_per_file=int(SETTINGS.ocr_max_pdf_pages_per_file),
                    max_pages_per_request=int(SETTINGS.ocr_max_pdf_pages_per_request),
                )

                direct_text = extract_pdf_text(str(file_path), selection.page_numbers)
                direct_text_pages = is_digital_pdf_text(direct_text, SETTINGS.digital_pdf_text_threshold)
                if direct_text_pages and not pages:
                    direct_lines = [line.strip() for line in direct_text.split("\n") if line.strip()]
                    return {
                        "file": filename,
                        "file_type": file_type,
                        "pages": page_count,
                        "lines": len(direct_lines),
                        "overall_confidence": 100.0,
                        "combined_text": direct_text,
                        "results": [{"page": selection.page_numbers[0], "text": line} for line in direct_lines],
                        "selected_pages": selection.page_numbers,
                    }

                for page_number in selection.page_numbers:
                    page = doc[page_number - 1]
                    image = pdf_page_to_img(page, 1.6 if enhance else 1.4)
                    results_for_page, accuracy = _build_results_for_image(image, page=page_number, enhance=enhance)
                    results.extend(results_for_page)
                    if accuracy:
                        all_confidences.append(accuracy)
            finally:
                doc.close()
        else:
            image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
            if image is None:
                raise ValueError("Invalid image")
            results_for_page, accuracy = _build_results_for_image(image, page=1, enhance=enhance)
            results.extend(results_for_page)
            if accuracy:
                all_confidences.append(accuracy)

        all_lines = [item["text"] for item in results]
        return {
            "file": filename,
            "file_type": file_type,
            "pages": page_count,
            "lines": len(all_lines),
            "overall_confidence": round(sum(all_confidences) / len(all_confidences), 2) if all_confidences else 0.0,
            "combined_text": "\n".join(all_lines),
            "results": results,
        }
