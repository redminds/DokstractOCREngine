"""Optional ONNX CTC handwriting recognizer adapter.

This module is intentionally imported lazily.  ``onnxruntime`` is not part of
the stable OCR image; installing it is an explicit optional deployment choice.
The adapter is suitable for word/crop recognition and does not claim full-page
handwriting detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .handwriting import (
    HANDWRITING_CONTRACT_VERSION,
    HandwritingCandidate,
    HandwritingRequest,
)


class OnnxHandwritingRecognizer:
    recognizer_id = "titanocr-hw-ctc-c1-clean"
    contract_version = HANDWRITING_CONTRACT_VERSION
    input_height = 48
    input_width = 320

    def __init__(self, model_path: str | Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional environment
            raise RuntimeError(
                "onnxruntime is required only when the optional handwriting recognizer is enabled"
            ) from exc
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Handwriting model not found: {self.model_path}")
        self._session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        metadata = self._session.get_modelmeta().custom_metadata_map
        self._characters = self._read_characters(metadata)

    @staticmethod
    def _read_characters(metadata: dict[str, str]) -> list[str]:
        raw = metadata.get("character") or metadata.get("characters")
        if not raw:
            raise ValueError("Handwriting ONNX model is missing character metadata")
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            # PaddleOCR exports character dictionaries as one character per
            # line.  Keeping the lines is important for whitespace and for
            # non-Latin Unicode characters.
            value = raw.splitlines()
        if isinstance(value, str):
            return list(value)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
        raise ValueError("Handwriting ONNX character metadata has an unsupported format")

    def _preprocess(self, image: Any) -> np.ndarray:
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Handwriting crop must have positive dimensions")
        target_width = max(1, round(width * self.input_height / height))
        target_width = min(target_width, self.input_width)
        image = image.resize((target_width, self.input_height), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR
        array = (array / 255.0 - 0.5) / 0.5
        padded = np.zeros((self.input_height, self.input_width, 3), dtype=np.float32)
        padded[:, :target_width, :] = array
        return np.transpose(padded, (2, 0, 1))[None, ...].astype(np.float32)

    def recognize(self, request: HandwritingRequest) -> list[HandwritingCandidate]:
        image = self._preprocess(request.image)
        output = self._session.run(None, {self._input_name: image})[0]
        logits = np.asarray(output)[0]
        if logits.ndim != 2:
            raise ValueError(f"Unexpected handwriting model output shape: {output.shape}")
        probabilities = _softmax(logits)
        token_ids = np.argmax(probabilities, axis=1)
        blank_id = 0
        decoded: list[str] = []
        confidences: list[float] = []
        previous = None
        for token_id, row in zip(token_ids.tolist(), probabilities):
            if token_id == blank_id or token_id == previous:
                previous = token_id
                continue
            character_index = token_id - 1  # CTC blank occupies class zero.
            if character_index < 0 or character_index >= len(self._characters):
                previous = token_id
                continue
            decoded.append(self._characters[character_index])
            confidences.append(float(row[token_id]))
            previous = token_id
        text = "".join(decoded).strip()
        if not text:
            return []
        polygon = _source_polygon(request)
        return [HandwritingCandidate(
            text=text,
            confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            polygon=polygon,
            source=self.recognizer_id,
            metadata={
                "recognizer": self.recognizer_id,
                "contract_version": self.contract_version,
                "model_path": str(self.model_path),
            },
        )]


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(values)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def _source_polygon(request: HandwritingRequest) -> tuple[tuple[float, float], ...]:
    bbox = request.source_bbox or (0.0, 0.0, request.page_width, request.page_height)
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 <= x2 <= request.page_width and 0 <= y1 <= y2 <= request.page_height):
        raise ValueError("Handwriting source_bbox must be within page-local coordinates")
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
