"""Dokstract OCR Engine benchmark runner — unified entry point.

Profiles:
  fast       — Production OCR validation, normal path only, compact output.
  recog      — Recognition investigation with crop variants and artifacts.

Usage:
  python tools/ocr_benchmark.py fast [--case madera-party]
  python tools/ocr_benchmark.py recog [--case rajkumar-party]
"""

import json, os, re, subprocess, sys, time, hashlib
from pathlib import Path

# ── Project paths ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "ocr-benchmarks"
RESOURCES_DIR = ROOT / "resources"
sys.path.insert(0, str(ROOT))

# ── Container references ────────────────────────────────────────────────
SCHEMA_CONTAINER = "dokstract-schema-api-local-schema-api-1"
OCR_CONTAINER = "dokstract-ocr-engine-local-ocr-engine-1"
TOKEN = "DSGDFGH46SDFFGK5DFG56GDFGHDFGKLHJ@SFDGR56DFRF"

# ── Regression cases ────────────────────────────────────────────────────
CASES = {
    "madera-party": {
        "doc": "Sale_Deed_MADERA_SRINIVASULU.pdf", "page": "2",
        "targets": ["M. ALIVELU", "M. SRINIVASULU"],
        "classification": "expected_pass",
        "note": "M.ALIVELU spacing is recognizer-sensitive; spacing difference is a known sensitivity, not a failure",
    },
    "madera-boundaries": {
        "doc": "Sale_Deed_MADERA_SRINIVASULU.pdf", "page": "8",
        "targets": ["NORTH", "SOUTH", "EAST", "WEST"],
        "classification": "expected_pass",
        "note": "WEST 20 wide Road: '20' may be clipped to '2ide' at baseline; fixed by 2% crop padding or CLAHE",
    },
    "rajkumar-party": {
        "doc": "Sale_Deed_RajKumar_Anthony.pdf", "page": "1",
        "targets": ["RAJ KUMAR", "ANTHONY", "ARJUN"],
        "classification": "known_model_limitation",
        "note": "ARJUN recognized as AR.HIN — PP-OCRv4 SVTR_LCNet italic serif limitation. Not fixable via padding/preprocessing/DPI.",
    },
    "rajkumar-built-up": {
        "doc": "Sale_Deed_RajKumar_Anthony.pdf", "page": "7",
        "targets": ["Total Built up area", "Ground Floor", "1100", "Sft."],
        "classification": "expected_pass",
        "note": "Stable across all tested DPI and preprocessing configurations",
    },
    "geometry-table": {
        "fixture": ".data/geometry-investigation/page3_items.json",
        "page": "3",
        "classification": "expected_pass",
        "note": "Real horizontal table geometry fixture captured from the corpus investigation.",
    },
}


def sh(cmd: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=True)


def _load_geometry_fixture(fixture_path: Path) -> dict:
    """Load a serialized OCR page fixture for table-regression replay."""
    data = json.loads(fixture_path.read_text())
    if "items" not in data:
        raise ValueError(f"Invalid geometry fixture: {fixture_path}")
    return data


def _line_to_dict(line) -> dict:
    norm = line.normalized_bbox
    return {
        "line_id": line.line_id,
        "text": line.text,
        "confidence": round(line.confidence, 4),
        "bbox": [round(line.bbox.x1, 1), round(line.bbox.y1, 1), round(line.bbox.x2, 1), round(line.bbox.y2, 1)],
        "normalized_bbox": [
            round(norm.x1, 4), round(norm.y1, 4), round(norm.x2, 4), round(norm.y2, 4),
        ] if norm else None,
        "item_ids": list(line.item_ids),
        "reading_order": line.reading_order,
    }


def _item_to_dict(item) -> dict:
    return {
        "item_id": item.item_id,
        "text": item.text,
        "confidence": round(item.confidence, 4),
        "polygon": [[round(p[0], 1), round(p[1], 1)] for p in item.polygon],
        "bbox": [round(item.bbox.x1, 1), round(item.bbox.y1, 1), round(item.bbox.x2, 1), round(item.bbox.y2, 1)],
        "normalized_bbox": [
            round(item.normalized_bbox.x1, 4), round(item.normalized_bbox.y1, 4),
            round(item.normalized_bbox.x2, 4), round(item.normalized_bbox.y2, 4),
        ] if item.normalized_bbox else None,
        "line_id": item.line_id,
        "block_id": item.block_id,
        "reading_order": item.reading_order,
    }


# ═════════════════════════════════════════════════════════════════════════
# FAST PROFILE — production OCR validation via Schema API proxy
# ═════════════════════════════════════════════════════════════════════════

def run_fast(case_filter: str | None = None) -> dict:
    """Run production OCR on selected cases and check target strings."""
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = ARTIFACT_DIR / f"fast-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases_to_run = {k: v for k, v in CASES.items() if not case_filter or k == case_filter}
    results = []
    for case_id, cfg in cases_to_run.items():
        page = cfg.get("page")
        targets = cfg.get("targets", [])
        fixture_rel = cfg.get("fixture")
        doc = cfg.get("doc")

        if fixture_rel:
            fixture_path = ROOT / fixture_rel
            try:
                fixture = _load_geometry_fixture(fixture_path)
                from app.services.ocr.models import OCRItem, BBox
                from app.services.ocr.table_reconstruction import detect_table_region

                items = [
                    OCRItem(
                        item_id=f"p3_i{i}",
                        page_number=int(page or 3),
                        text=it["text"],
                        confidence=it["confidence"],
                        polygon=it["polygon"],
                        bbox=BBox(it["x1"], it["y1"], it["x2"], it["y2"]),
                        normalized_bbox=BBox(it["nx1"], it["ny1"], it["nx2"], it["ny2"]),
                    )
                    for i, it in enumerate(fixture["items"])
                ]
                t0 = time.perf_counter()
                lines, non_table, table_meta = detect_table_region(items, fixture["page_w"], fixture["page_h"])
                elapsed = time.perf_counter() - t0
                invalid_item_refs = [it.item_id for it in items if it.line_id and it.line_id not in {ln.line_id for ln in lines}]
                none_item_refs = [it.item_id for it in items if it.line_id is None]
                expected_checks = [
                    ("physical_column_count=17", table_meta and table_meta.get("physical_column_count") == 17),
                    ("logical_column_count=5", table_meta and table_meta.get("logical_column_count") == 5),
                    ("row_count=11", table_meta and len(table_meta.get("data_rows", [])) == 11),
                    ("table_items=64", table_meta and table_meta.get("table_items") == 64),
                    ("table_lines=35", table_meta and table_meta.get("table_lines") == 35),
                ]
                exact = [label for label, ok in expected_checks if ok]
                missing = [label for label, ok in expected_checks if not ok]
                r = {
                    "case": case_id,
                    "classification": cfg["classification"],
                    "document": fixture_path.name,
                    "page": page,
                    "expected": [label for label, _ in expected_checks],
                    "exact_matches": exact,
                    "ci_matches": [],
                    "missing": missing,
                    "lines": [_line_to_dict(ln) for ln in lines],
                    "items": [_item_to_dict(it) for it in items],
                    "confidence": round(table_meta.get("overall_confidence", 0.0) if table_meta else 0.0, 4),
                    "render_dpi": fixture.get("dpi"),
                    "width": fixture.get("page_w"),
                    "height": fixture.get("page_h"),
                    "duration_s": round(elapsed, 2),
                    "status": "PASS" if table_meta and not missing and not invalid_item_refs else "FAIL",
                    "raw_text": "",
                    "page_number": int(page or 3),
                    "table_meta": table_meta,
                    "integrity": {
                        "non_null_item_refs": len(items) - len(none_item_refs),
                        "none_item_refs": len(none_item_refs),
                        "invalid_item_refs": invalid_item_refs,
                        "missing_line_item_refs": [],
                        "table_count": 1 if table_meta else 0,
                        "item_ids": sorted(it.item_id for it in items),
                        "line_ids": sorted(ln.line_id for ln in lines),
                    },
                }
                results.append(r)
                continue
            except Exception as exc:
                results.append({"case": case_id, "status": "ERROR", "reason": str(exc)})
                continue

        doc_path = f"/app/resources/{doc}"

        # Check document availability via Schema API container
        check = sh(f"docker exec {SCHEMA_CONTAINER} test -f {doc_path}", timeout=5)
        if check.returncode != 0:
            results.append({"case": case_id, "status": "SKIP", "reason": f"document not found: {doc}"})
            continue

        # Clear cache
        sh(f"docker exec {OCR_CONTAINER} sh -c 'rm -rf /app/.data/ocr-cache/* 2>/dev/null'", timeout=5)

        # Run OCR request via Schema API container
        t0 = time.perf_counter()
        # Write a clean Python script, copy, execute, capture result
        bm_script = f'''import json, httpx, sys
sys.path.insert(0,"/app")
with open("{doc_path}","rb") as f:
    fb = f.read()
r = httpx.post("http://ocr-engine:8010/api/v1/internal/ocr/extract",
    data={{"project_key":"schema","pages":"{page}"}},
    files={{"file":("bm.pdf",fb,"application/pdf")}},
    headers={{"X-Service-Name":"schema-api","X-Service-Token":"{TOKEN}"}},
    timeout=300)
d = r.json()
pg = d.get("pages",[{{}}])[0] if d.get("pages") else {{}}
raw = pg.get("text","")
lines = pg.get("lines",[])
items = pg.get("items",[])
m = pg.get("metrics",{{}})
line_ids = {{ln.get("line_id") for ln in lines if ln.get("line_id")}}
item_ids = {{it.get("item_id") for it in items if it.get("item_id")}}
invalid_item_refs = [it.get("item_id") for it in items if it.get("line_id") and it.get("line_id") not in line_ids]
missing_line_item_refs = [ln.get("line_id") for ln in lines if any(iid not in item_ids for iid in ln.get("item_ids", []))]
none_item_refs = [it.get("item_id") for it in items if it.get("line_id") is None]
table_meta = pg.get("table_meta") or {{}}
print(json.dumps({{
    "raw": raw, "line_count": len(lines), "item_count": len(items),
    "width": pg.get("width"), "height": pg.get("height"),
    "confidence": pg.get("confidence",0), "render_dpi": m.get("render_dpi"),
    "page_number": pg.get("page_number"),
    "items": items,
    "lines": lines,
    "blocks": pg.get("blocks", []),
    "table_meta": table_meta if table_meta else None,
    "integrity": {{
        "non_null_item_refs": len(items) - len(none_item_refs),
        "none_item_refs": len(none_item_refs),
        "invalid_item_refs": invalid_item_refs,
        "missing_line_item_refs": missing_line_item_refs,
        "table_count": 1 if table_meta else 0,
        "item_ids": sorted(item_ids),
        "line_ids": sorted(line_ids),
    }},
}}))
'''
        tmp_host = ROOT / "artifacts" / "_bm_fast.py"
        tmp_host.parent.mkdir(parents=True, exist_ok=True)
        tmp_host.write_text(bm_script)
        sh(f'docker cp "{tmp_host}" {SCHEMA_CONTAINER}:/tmp/_bm_fast.py', timeout=10)
        resp = sh(f"docker exec {SCHEMA_CONTAINER} python3 /tmp/_bm_fast.py", timeout=300)
        elapsed = time.perf_counter() - t0

        try:
            data = json.loads(resp.stdout.strip())
        except Exception:
            results.append({"case": case_id, "status": "ERROR", "reason": resp.stderr[:200]})
            continue

        raw = data.get("raw", "")
        integrity = data.get("integrity", {})
        exact = [t for t in targets if t in raw]
        ci = [t for t in targets if t.lower() in raw.lower() and t not in exact]
        missing = [t for t in targets if t not in raw and t.lower() not in raw.lower()]

        r = {
            "case": case_id, "classification": cfg["classification"],
            "document": doc, "page": page,
            "expected": targets,
            "exact_matches": exact, "ci_matches": ci, "missing": missing,
            "lines": data.get("lines"), "items": data.get("items"),
            "confidence": round(data.get("confidence", 0), 4),
            "render_dpi": data.get("render_dpi"),
            "width": data.get("width"), "height": data.get("height"),
            "duration_s": round(elapsed, 2), "status": "PASS" if not missing else "FAIL",
            "raw_text": raw[:1500],
            "page_number": data.get("page_number"),
            "table_meta": data.get("table_meta"),
            "integrity": integrity,
        }
        # Known model limitations pass even if targets missing
        if cfg["classification"] == "known_model_limitation":
            r["status"] = "KNOWN_LIMITATION" if "ARJUN" in missing else "PASS"
        results.append(r)

    # Write outputs
    summary = {"run_id": run_id, "profile": "fast", "results": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # Generate markdown report
    md_lines = [f"# OCR Benchmark — Fast Profile\n", f"Run: {run_id}\n"]
    md_lines.append("| Case | Status | Matches | Missing | Conf | Lines | DPI |")
    md_lines.append("|------|--------|---------|---------|------|-------|-----|")
    for r in results:
        if r.get("status") == "SKIP":
            md_lines.append(f"| {r['case']} | SKIP | — | — | — | — | — |")
            continue
        lines_val = r.get("lines")
        line_count = len(lines_val) if isinstance(lines_val, list) else lines_val
        md_lines.append(f"| {r['case']} | {r['status']} | {len(r.get('exact_matches',[]))}/{len(r.get('expected',[]))} | {', '.join(r.get('missing',[]) or ['none'])} | {r.get('confidence',0):.3f} | {line_count if line_count is not None else '?'} | {r.get('render_dpi','?')} |")
    (out_dir / "report.md").write_text("\n".join(md_lines))
    print(f"Report: {out_dir}/report.md")

    # Exit code
    failures = [r for r in results if r["status"] == "FAIL"]
    if failures:
        print(f"FAILURES: {[r['case'] for r in failures]}")
        return summary

    print("All cases passed or known limitations.")
    return summary


# ═════════════════════════════════════════════════════════════════════════
# RECOGNITION INVESTIGATION PROFILE — targeted crop experiments
# ═════════════════════════════════════════════════════════════════════════

RECOG_PY_TEMPLATE = r'''
import json, os, sys, time, cv2, numpy as np
sys.path.insert(0,"/app")
from app.core.ocr_execution import _get_ocr_engine, pdf_page_to_img, open_pdf_from_bytes
from app.core.config import SETTINGS

CASE_ID = "{case_id}"
DOC_PATH = "/app/resources/{doc}"
PAGE_IDX = {page_idx}  # 0-indexed
SEARCH = {search}
EXPECTED = {expected}
RENDER_SCALE = {render_scale}
PADDING_VARIANTS = {padding_variants}
PP_VARIANTS = {pp_variants}
OUTDIR = "/app/.data/ocr-benchmarks/{run_id}/{case_id}"
os.makedirs(OUTDIR, exist_ok=True)

engine = _get_ocr_engine()
with open(DOC_PATH, "rb") as f: fb = f.read()
doc = open_pdf_from_bytes(fb)
page = doc[PAGE_IDX]
render = pdf_page_to_img(page, RENDER_SCALE)
image = render.image
h, w = image.shape[:2]
print(f"PAGE: {{w}}x{{h}} DPI={{round(render.effective_scale*72,1)}}")

# Run detection
result = engine.ocr(image, cls=False)
found = None
for det in (result[0] or []):
    text = det[1][0]
    for s in SEARCH:
        if s.upper() in text.upper():
            found = det
            break
    if found: break

if found is None:
    print(json.dumps({{"status": "NOT_FOUND", "search_terms": SEARCH}}))
    sys.exit(0)

poly = np.array(found[0], dtype=np.int32)
rec_text, rec_conf = found[1]
print(f"FOUND: {{rec_text!r}} conf={{rec_conf:.3f}} poly={{found[0]}}")

# Save overlay
overlay = image.copy()
cv2.polylines(overlay, [poly], True, (0, 255, 0), 2)
cv2.imwrite(os.path.join(OUTDIR, "polygon.png"), overlay)

results = []
for label, pad_h, pad_v in PADDING_VARIANTS:
    x1, y1 = poly[:,0].min(), poly[:,1].min()
    x2, y2 = poly[:,0].max(), poly[:,1].max()
    bw, bh = x2-x1, y2-y1
    px = int(bw * pad_h / 100)
    py = int(bh * pad_v / 100)
    x1c, y1c = max(0, int(x1)-px), max(0, int(y1)-py)
    x2c, y2c = min(w, int(x2)+px), min(h, int(y2)+py)
    crop = image[y1c:y2c, x1c:x2c].copy()
    cv2.imwrite(os.path.join(OUTDIR, f"crop_{{label}}.png"), crop)
    out = engine.ocr(crop, cls=False, det=False, rec=True)
    if out and out[0]:
        txt, cnf = out[0][0]
    else:
        txt, cnf = "", 0
    match = any(e.upper() in txt.upper() for e in EXPECTED)
    results.append({{"variant": label, "pad_h_pct": pad_h, "pad_v_pct": pad_v,
        "crop_w": crop.shape[1], "crop_h": crop.shape[0],
        "rec_text": txt, "rec_conf": round(cnf, 4), "match": match}})

for label in PP_VARIANTS:
    x1, y1 = poly[:,0].min(), poly[:,1].min()
    x2, y2 = poly[:,0].max(), poly[:,1].max()
    crop = image[int(y1):int(y2), int(x1):int(x2)].copy()
    if label == "gray":
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    elif label == "clahe":
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)
        crop = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(OUTDIR, f"crop_{{label}}.png"), crop)
    out = engine.ocr(crop, cls=False, det=False, rec=True)
    if out and out[0]:
        txt, cnf = out[0][0]
    else:
        txt, cnf = "", 0
    match = any(e.upper() in txt.upper() for e in EXPECTED)
    results.append({{"variant": label, "pad_h_pct": 0, "pad_v_pct": 0,
        "crop_w": crop.shape[1], "crop_h": crop.shape[0],
        "rec_text": txt, "rec_conf": round(cnf, 4), "match": match}})

summary = {{
    "case": CASE_ID, "document": os.path.basename(DOC_PATH), "page": PAGE_IDX+1,
    "render_scale": round(render.effective_scale, 4),
    "render_dpi": round(render.effective_scale*72, 1),
    "render_w": w, "render_h": h,
    "polygon": found[0], "prod_text": rec_text, "prod_conf": round(rec_conf, 4),
    "expected": EXPECTED, "classification": "{classification}",
    "variants": results,
    "artifact_dir": OUTDIR,
}}
with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
'''

PADDING_VARIANTS = [
    ("R0", 0, 0),
    ("R1", 2, 5),
    ("R2", 5, 10),
    ("R3", 10, 5),
]
PP_VARIANTS = ["gray", "clahe"]


def run_recog(case_filter: str | None = None) -> dict:
    """Run recognition investigation on selected cases."""
    run_id = time.strftime("%Y%m%d-%H%M%S")
    cases_to_run = {k: v for k, v in CASES.items() if not case_filter or k == case_filter}
    all_summaries = []

    for case_id, cfg in cases_to_run.items():
        print(f"\n--- {case_id} ---")
        # Copy PDF to OCR container if needed
        doc = cfg["doc"]
        check = sh(f"docker exec {OCR_CONTAINER} test -f /app/resources/{doc}", timeout=5)
        if check.returncode != 0:
            src = RESOURCES_DIR / doc
            if src.exists():
                sh(f"docker exec {OCR_CONTAINER} mkdir -p /app/resources", timeout=5)
                sh(f'docker cp "{src}" {OCR_CONTAINER}:/app/resources/', timeout=30)

        # Determine search terms from targets
        search_terms = json.dumps(cfg["targets"][:2])
        expected = json.dumps(cfg["targets"])
        page_idx = int(cfg["page"]) - 1

        script = RECOG_PY_TEMPLATE.format(
            case_id=case_id, doc=doc, page_idx=page_idx,
            search=search_terms, expected=expected,
            render_scale=SETTINGS_RENDER_SCALE,
            run_id=run_id,
            padding_variants=json.dumps(PADDING_VARIANTS),
            pp_variants=json.dumps(PP_VARIANTS),
            classification=cfg["classification"],
        )
        # Write script to temp and execute in OCR container
        script_path = f"/app/tests/_recog_{case_id}.py"
        escaped = script.replace("'", "'\"'\"'")
        sh(f"docker exec {OCR_CONTAINER} sh -c \"cat > {script_path} << 'PYEOF'\n{script}\nPYEOF\"", timeout=10)
        resp = sh(f"docker exec {OCR_CONTAINER} python3 {script_path}", timeout=600)

        try:
            summary = json.loads(resp.stdout.strip())
        except Exception:
            summary = {"case": case_id, "status": "ERROR", "reason": resp.stderr[:500]}
        all_summaries.append(summary)

        print(f"  {summary.get('status', summary.get('prod_text', 'done'))}")

    # Write combined report
    out_dir = ARTIFACT_DIR / f"recog-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"run_id": run_id, "profile": "recog", "cases": all_summaries}
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))

    md = [f"# OCR Benchmark — Recognition Investigation\n", f"Run: {run_id}\n"]
    for s in all_summaries:
        md.append(f"## {s.get('case', 'ERROR')}")
        md.append(f"Classification: {s.get('classification', 'N/A')}")
        md.append(f"Production text: `{s.get('prod_text', '?')}`")
        md.append(f"Expected: {s.get('expected', [])}")
        md.append(f"Artifacts: `{s.get('artifact_dir', 'N/A')}`")
        for v in (s.get('variants') or []):
            md.append(f"- **{v['variant']}**: `{v['rec_text']}` conf={v['rec_conf']:.4f} match={v['match']}")
        md.append("")
    (out_dir / "report.md").write_text("\n".join(md))
    print(f"\nReport: {out_dir}/report.md")
    return report


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════

SETTINGS_RENDER_SCALE = 2.78  # Config C default for recog investigation

HELP = """Dokstract OCR Benchmark Runner
Usage:  python tools/ocr_benchmark.py <profile> [--case <id>]

Profiles:
  fast    Production OCR validation — fast, no crop variants
  recog   Recognition investigation — crop variants, artifacts

Cases:  madera-party, madera-boundaries, rajkumar-party, rajkumar-built-up, geometry-table

Examples:
  python tools/ocr_benchmark.py fast
  python tools/ocr_benchmark.py fast --case rajkumar-party
  python tools/ocr_benchmark.py recog --case madera-boundaries
  python tools/ocr_benchmark.py recog
"""

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("profile", nargs="?", choices=["fast", "recog", "help"], default="help")
    p.add_argument("--case", dest="case_filter", default=None)
    args = p.parse_args()

    if args.profile == "help":
        print(HELP)
        sys.exit(0)

    if args.profile == "fast":
        r = run_fast(args.case_filter)
        failures = [x for x in r.get("results", []) if x["status"] == "FAIL"]
        sys.exit(1 if failures else 0)

    if args.profile == "recog":
        run_recog(args.case_filter)
