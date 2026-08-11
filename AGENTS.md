# Agent Contract — Dokstract OCR Engine

## Required: Use the OCR benchmark entry point first

Before any OCR investigation or quality work, use the existing benchmark runner:

```bash
python tools/ocr_benchmark.py fast              # quick production validation
python tools/ocr_benchmark.py fast --case <id>  # single case
python tools/ocr_benchmark.py recog --case <id> # crop-variant investigation
```

Or via Makefile:

```bash
make benchmark-fast [BENCHMARK_CASE=madera-party]
make benchmark-recog [BENCHMARK_CASE=rajkumar-party]
```

## Do NOT begin OCR investigation with

- `docker exec` — use the benchmark runner instead
- `docker cp` — the benchmark copies required PDFs automatically
- Temporary Python scripts copied into containers — the recognition investigation profile handles crop variants
- Raw `curl` to OCR endpoints — use `benchmark-fast`

Use direct Docker commands only when the benchmark itself reports an unsupported capability or fails to execute.

## Structured extraction benchmark (Schema API)

The extraction benchmark lives in the Schema API repository:

```bash
# From DokstractSchemaAPI:
python tools/extraction_benchmark.py prepare     # Cache OCR for all documents
python tools/extraction_benchmark.py smoke       # Run one extraction case
python tools/extraction_benchmark.py full        # Run all approved cases
python tools/extraction_benchmark.py compare A B # Diff two saved runs
```

All six sale deeds are registered in `resources/extraction_benchmark/cases.json`.
Ground truth templates are in `resources/extraction_benchmark/ground_truth/` (all `pending_review`).
Use filtered commands: `--case madera-srinivasulu`, `--refresh-ocr`.

## Benchmark profiles

| Profile | Command | Purpose |
|---------|---------|---------|
| `fast` | `make benchmark-fast` | Validate production OCR, check target strings |
| `recog` | `make benchmark-recog` | Crop padding/preprocessing variants, artifact saving |

## Output structure

```
artifacts/ocr-benchmarks/<run-id>/
├── summary.json        # machine-readable results
├── report.md          # human-readable markdown summary
└── <case-id>/
    ├── polygon.png    # rendered page with detection overlay
    ├── crop_R0.png .. crop_R3.png  # padding variants
    ├── crop_gray.png, crop_clahe.png  # preprocessing variants
    └── summary.json   # per-case detailed results
```

## Known limitations

- `ARJUN` recognized as `AR.HIN`: PP-OCRv4 SVTR_LCNet italic serif model limitation. No fix via padding/preprocessing/DPI.
- `M.ALIVELU` spacing: Recognizer-sensitive to crop boundaries. Preserved at baseline (R0).
- `WEST 20`: Fixed by 2% horizontal crop padding (benchmark-only finding; no production fallback implemented).

## Container references

- OCR Engine: `dokstract-ocr-engine-local-ocr-engine-1`
- Schema API: `dokstract-schema-api-local-schema-api-1`

## Related repositories

- Schema API agent contract: `../DokstractSchemaAPI/AGENTS.md`
- Schema API operator: `../DokstractSchemaAPI/tools/dokstract.ps1`
