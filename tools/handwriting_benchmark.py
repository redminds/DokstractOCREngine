"""Evaluate the optional ONNX handwriting recognizer on RxHandBD test crops."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.ocr.handwriting import HandwritingRequest
from app.services.ocr.handwriting_onnx import OnnxHandwritingRecognizer


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    recognizer = OnnxHandwritingRecognizer(args.model)
    labels = list(csv.DictReader((args.dataset / "Test_Labels.csv").open(newline="")))[:args.limit]
    total_chars = total_words = char_errors = word_errors = 0
    elapsed = []
    examples = []
    for row in labels:
        truth = row["Text"].strip()
        image_path = args.dataset / "Test_Set" / row["Images"]
        start = time.perf_counter()
        candidates = recognizer.recognize(HandwritingRequest(
            page_number=1,
            image=Image.open(image_path),
            page_width=Image.open(image_path).width,
            page_height=Image.open(image_path).height,
            source_bbox=(0.0, 0.0, float(Image.open(image_path).width), float(Image.open(image_path).height)),
        ))
        elapsed.append(time.perf_counter() - start)
        prediction = candidates[0].text if candidates else ""
        char_errors += edit_distance(prediction, truth)
        word_errors += int(prediction != truth)
        total_chars += len(truth)
        total_words += 1
        if len(examples) < 5:
            examples.append({"image": row["Images"], "truth": truth, "prediction": prediction})

    result = {
        "dataset": "RxHandBD",
        "split": "Test_Set",
        "samples": total_words,
        "cer": char_errors / total_chars if total_chars else 0.0,
        "wer": word_errors / total_words if total_words else 0.0,
        "mean_ms": (sum(elapsed) / len(elapsed) * 1000) if elapsed else 0.0,
        "p95_ms": (float(np.percentile(elapsed, 95)) * 1000) if elapsed else 0.0,
        "examples": examples,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
