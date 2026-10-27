# Optional handwriting benchmark

This directory documents the disposable benchmark sources. The images and
transcriptions are intentionally kept outside the repository and outside
Golden Data.

## RxHandBD

- Mendeley version 3: https://data.mendeley.com/datasets/dsb5r6vskg/3
- DOI: `10.17632/dsb5r6vskg.3`
- Published description: 5,578 cropped word images with train/test labels.
- Mendeley page license: CC BY 4.0.
- Disposable archive used: Zenodo record `10.5281/zenodo.18478741`,
  `RxHandBD.zip`, 32,464,605 bytes, MD5
  `c096859fcccb33655edaf34e4c1e7a06`, SHA-256
  `25d31c7bc1ceab4b3707d5e6d2205e518cadbd75d2f99efe1492a1aee5ef1951`.
- The Zenodo record reports a different MIT license; this discrepancy must be
  resolved before any commercial training or redistribution.

## African Medical Records

- Source: https://huggingface.co/datasets/Nigeria-Health-data-OCR-pipeline/African-Medical-Records
- License: CC-BY-4.0.
- Published release: 62 matched handwritten image/transcription pairs.
- Disposable pair: `AMR_001_HTR.png` and `truth/AMR_001.txt`.
- SHA-256 image:
  `58cd7bb9e0dba28b9958d745ce12420d4a1fc56c899e295a9ba57570ee40d951`.
- SHA-256 transcription:
  `8f3aef40938341b0db66c9c9f679542576ffe1a3ccbdfc7eaf43ae40609561da`.

## Model

- Apache-2.0 ONNX model:
  https://huggingface.co/aravsaxena884/titanocr-handwriting-ppocrv5-ctc
- File: `titanocr-hw-ctc-c1-clean.onnx`.
- SHA-256:
  `871cea67c01ef1b8da079794746b1b2de7de7a8f5b599a8b0c7aeca7a5f31cf7`.
- Scope: word/crop recognition only; not accepted for full-page recognition.
