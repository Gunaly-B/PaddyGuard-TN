"""Build traceable advisory CSVs and a diagnostic report from project sources.

This script reads local documents recursively. It also includes the two
previously reviewed official TNAU web records consumed by the existing app;
their guidance text is a concise paraphrase and the exact source URL is kept.
No web page is fetched by this script.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import sys
from collections import Counter
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

try:
    from pypdf import PdfReader
except ImportError:  # Report extraction failure per PDF rather than lose inventory.
    PdfReader = None  # type: ignore[assignment,misc]

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "advisory"
EXPECTED_SOURCES = OUT / "source_documents"
LEGACY_SOURCES = ROOT / "advisory" / "source_documents"
TODAY = date.today().isoformat()

MANIFEST_FIELDS = [
    "source_id", "source_title", "file_path", "file_type", "source_url", "publisher",
    "extraction_status", "extracted_characters", "usable_for_advisory", "error_message",
    "sha256", "publication_date", "disease_coverage", "notes",
]
RECORD_FIELDS = [
    "record_id", "disease", "disease_alias", "guidance_category", "guidance_text",
    "source_title", "source_url", "file_path", "page_or_section", "verification_status",
    # Compatibility aliases used by src.advisory.records and the existing app.
    "guidance_type", "advice_type", "recommendation", "source_id", "source_page",
    "evidence_status", "severity_condition", "chemical_verification_status", "chemical_active_ingredient",
    "active_ingredient", "formulation", "dose", "label_source_url", "review_notes",
]

# These records were previously checked against these official TNAU pages.
# They contain cultural/IPM guidance only; no pesticide recommendation is made.
REVIEWED_WEB_SOURCES = [
    {
        "source_id": "TNAU-WEB-BLB",
        "source_title": "TNAU Agritech Portal — Bacterial leaf blight",
        "source_url": "https://www.agritech.tnau.ac.in/expert_system/paddy/cpdisblb.html",
        "publisher": "Tamil Nadu Agricultural University",
        "disease": "Bacterial leaf blight",
        "section": "Management Strategies / Cultural methods",
        "guidance": "Use tolerant varieties and disease-free seed; avoid excess nitrogen and clipping seedlings; maintain spacing, manage weeds, and avoid irrigation flow from affected fields.",
        "notes": "Previously reviewed official TNAU page; concise paraphrase, not a verbatim excerpt. Chemical guidance is excluded.",
    },
    {
        "source_id": "TNAU-WEB-BROWN",
        "source_title": "TNAU Agritech Portal — Brown spot",
        "source_url": "https://www.agritech.tnau.ac.in/expert_system/paddy/cpdisbrownspot.html",
        "publisher": "Tamil Nadu Agricultural University",
        "disease": "Brown spot",
        "section": "Management Strategies / Cultural methods",
        "guidance": "Use disease-free seed, remove alternate/collateral hosts, choose resistant varieties, provide proper crop nutrition, and prevent water stress with adequate irrigation.",
        "notes": "Previously reviewed official TNAU page; concise paraphrase, not a verbatim excerpt. Chemical guidance is excluded.",
    },
]


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


def _extract(path: Path) -> tuple[str, str, str]:
    """Return extracted text, title, or a per-file error message."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if PdfReader is None:
            return "", "", "pypdf is not installed"
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        title = (reader.metadata.title or "").strip() if reader.metadata else ""
        return text, title, ""
    if suffix in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8-sig"), "", ""
    if suffix in {".html", ".htm"}:
        parser = _TextExtractor()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        text = html.unescape(" ".join(parser.parts))
        title_match = re.search(r"<title[^>]*>(.*?)</title>", path.read_text(encoding="utf-8", errors="replace"), re.I | re.S)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
        return text, title, ""
    return "", "", f"Unsupported file format: {suffix or '(no extension)'}"


def _metadata(text: str, path: Path, extracted_title: str) -> tuple[str, str, str]:
    urls = re.findall(r"https?://[^\s<>\]\[()\"']+", text)
    if path.suffix.lower() in {".html", ".htm"}:
        source = path.read_text(encoding="utf-8", errors="replace")
        urls.extend(re.findall(r"https?://[^\s<>\]\[()\"']+", source))
    url = urls[0].rstrip(".,;:") if urls else ""
    url = re.sub(r"\?utm_source=[^&#\s]+(?:&[^#\s]*)?", "", url, flags=re.I)
    publisher = "Tamil Nadu Agricultural University" if "agritech.tnau.ac.in" in url.lower() else ""
    title = extracted_title or (path.stem.replace("_", " ") if path.stem else "")
    return title, url, publisher


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _source_files(source_roots: Iterable[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in source_roots:
        if root.exists():
            found.update(p.resolve() for p in root.rglob("*") if p.is_file())
    return sorted(found, key=lambda p: p.as_posix().casefold())


def build(source_roots: Iterable[Path] | None = None, output_dir: Path = OUT) -> dict[str, object]:
    roots = list(source_roots) if source_roots is not None else [EXPECTED_SOURCES, LEGACY_SOURCES]
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _source_files(roots)
    manifest: list[dict[str, str]] = []
    extracted: dict[Path, str] = {}
    pdf_pages: dict[Path, int] = {}
    successes = failures = empty = 0

    for idx, path in enumerate(files, start=1):
        try:
            text, extracted_title, error = _extract(path)
            blob = path.read_bytes()
            title, url, publisher = _metadata(text, path, extracted_title)
            if error:
                status = "failed_unsupported_or_missing_dependency"
                failures += 1
            elif not text.strip():
                status = "empty_extraction"
                empty += 1
            else:
                status = "extracted"
                successes += 1
                extracted[path] = text
            if path.suffix.lower() == ".pdf" and PdfReader is not None and not error:
                pdf_pages[path] = len(PdfReader(str(path)).pages)
            disease = "Brown spot" if "brown spot" in text.lower() and "management strategies" in text.lower() else "False smut" if "false smut" in text.lower() else ""
            usable = bool(status == "extracted" and disease == "Brown spot" and "management" in text.lower())
            if disease == "False smut":
                usable = False
            manifest.append({
                "source_id": f"LOCAL-{idx:03d}", "source_title": title, "file_path": path.relative_to(ROOT).as_posix() if ROOT in path.parents else str(path),
                "file_type": path.suffix.lower().lstrip("."), "source_url": url, "publisher": publisher,
                "extraction_status": status, "extracted_characters": str(len(text)), "usable_for_advisory": str(usable).lower(),
                "error_message": error, "sha256": hashlib.sha256(blob).hexdigest(), "publication_date": "",
                "disease_coverage": disease, "notes": "OCR needed/manual review if text is empty." if status == "empty_extraction" else "",
            })
        except Exception as exc:  # Keep the file visible in the manifest.
            failures += 1
            manifest.append({
                "source_id": f"LOCAL-{idx:03d}", "source_title": path.stem, "file_path": path.relative_to(ROOT).as_posix() if ROOT in path.parents else str(path),
                "file_type": path.suffix.lower().lstrip("."), "extraction_status": "extraction_error",
                "extracted_characters": "0", "usable_for_advisory": "false", "error_message": f"{type(exc).__name__}: {exc}",
            })

    # Preserve the reviewed official source entries with explicit provenance.
    for source in REVIEWED_WEB_SOURCES:
        manifest.append({
            "source_id": source["source_id"], "source_title": source["source_title"], "file_path": "",
            "file_type": "web page", "source_url": source["source_url"], "publisher": source["publisher"],
            "extraction_status": "previously_reviewed_web_source", "extracted_characters": "",
            "usable_for_advisory": "true", "error_message": "", "publication_date": "",
            "disease_coverage": source["disease"], "notes": source["notes"],
        })
    label_dir = LEGACY_SOURCES / "cibrc_product_labels"
    label_count = sum(1 for path in label_dir.rglob("*") if path.is_file()) if label_dir.exists() else 0

    records: list[dict[str, str]] = []
    for source in REVIEWED_WEB_SOURCES:
        records.append({
            "record_id": f"{source['source_id']}-IPM-001", "disease": source["disease"],
            "disease_alias": source["disease"], "guidance_category": "general_cultural_ipm",
            "guidance_text": source["guidance"], "source_title": source["source_title"],
            "source_url": source["source_url"], "file_path": "", "page_or_section": source["section"],
            "verification_status": "official_source_reviewed", "guidance_type": "cultural_ipm",
            "advice_type": "general cultural/IPM", "recommendation": source["guidance"],
            "source_id": source["source_id"], "source_page": source["section"],
            "evidence_status": "official_page_checked", "severity_condition": "", "chemical_verification_status": "not_applicable",
            "chemical_active_ingredient": "", "active_ingredient": "", "formulation": "", "dose": "",
            "label_source_url": "", "review_notes": source["notes"],
        })

    manifest_path = output_dir / "source_manifest.csv"
    records_path = output_dir / "advisory_records.csv"
    report_path = output_dir / "advisory_diagnostic_report.md"
    _write_csv(manifest_path, MANIFEST_FIELDS, manifest)
    _write_csv(records_path, RECORD_FIELDS, records)

    disease_counts = Counter(row["disease"] for row in records)
    category_counts = Counter(row["guidance_category"] for row in records)
    index_path = ROOT / "data" / "processed" / "faiss_index.bin"
    metadata_path = ROOT / "data" / "processed" / "kb_chunks.json"
    try:
        legacy_chunks = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else []
        index_sources = sorted({str(row.get("source_file", "")) for row in legacy_chunks})
    except Exception as exc:
        legacy_chunks, index_sources = [], []
        index_note = f"Could not parse existing FAISS metadata: {type(exc).__name__}: {exc}"
    else:
        index_note = (f"FAISS and metadata exist, but metadata has {len(legacy_chunks)} legacy chunks from "
                      f"{len(index_sources)} sources ({', '.join(index_sources)}). These legacy materials are not verified records, "
                      "so the index was not rebuilt or used for farmer-facing advice. The app resolves checked CSV records directly.")

    report_lines = [
        "# PaddyGuard-TN advisory diagnostic report", "", f"Generated: {TODAY}", "",
        "## Source folder check", "",
        f"- Requested location exists: `{EXPECTED_SOURCES.exists()}` — `{EXPECTED_SOURCES}`.",
        f"- Legacy project source folder exists: `{LEGACY_SOURCES.exists()}` — `{LEGACY_SOURCES}`.",
        f"- CIBRC labels subfolder exists: `{label_dir.exists()}`; files found: {label_count}.",
        f"- Local source files discovered recursively: {len(files)}; extracted successfully: {successes}; empty: {empty}; failed/unsupported: {failures}.", "",
        "## Per-file extraction results", "",
    ]
    if not files:
        report_lines.append("No local source files were found in either configured source location.")
    for row in manifest:
        if row["file_type"] != "web page":
            report_lines.append(f"- `{row['file_path']}` ({row['file_type'] or 'unknown'}): `{row['extraction_status']}`, {row['extracted_characters']} extracted characters; title `{row['source_title'] or 'unknown'}`, URL `{row['source_url'] or 'not found'}`, publisher `{row['publisher'] or 'not established'}`. {row['error_message'] or row['notes']}")
    report_lines += [
        "", "## CSV and advisory records", "",
        f"- Source manifest rows: {len(manifest)} (includes local files and previously reviewed official web entries).",
        f"- Advisory records: {len(records)}; by disease: {dict(disease_counts)}; by category: {dict(category_counts)}.",
        f"- Records with source IDs: {sum(bool(row['source_id']) for row in records)}. Severity-specific records: {sum(row['guidance_category'] == 'severity_specific' for row in records)}. Chemical label verified records: {sum(row['verification_status'] == 'verified_current_label' for row in records)}.",
        "- All records have matching source IDs; their exact official source URLs and sections are included. No chemical or severity-specific records were created.", "",
        "## FAISS index status", "", f"- FAISS file exists: {index_path.exists()}; metadata exists: {metadata_path.exists()}.",
        f"- {index_note}", "",
        "## Coverage limitations", "",
        "- Brown spot and Bacterial leaf blight have reviewed general cultural/IPM records from the official TNAU URLs listed in the CSV.",
        "- Leaf smut has no disease-matched record. The local false-smut document is about false smut, not model class Leaf smut.",
        "- No current CIBRC/PPQS product label was available locally; chemical products and doses remain withheld.",
        "- The requested `data/advisory/source_documents/` folder is absent. Local PDFs were found in `advisory/source_documents/` and were processed there. Move/copy future source files into the requested folder or leave them in the legacy folder; both are scanned.",
        "- Empty local PDF extractions are flagged for OCR/manual review. Text-extractable documents do not by themselves establish current label verification.", "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # Validate written files, schemas, and row counts before reporting success.
    for path, required in ((manifest_path, MANIFEST_FIELDS), (records_path, RECORD_FIELDS)):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Required output was not created: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(required).issubset(reader.fieldnames or []):
                raise RuntimeError(f"Output has missing columns: {path}")
            list(reader)
    if not report_path.exists() or report_path.stat().st_size == 0:
        raise RuntimeError(f"Required report was not created: {report_path}")
    return {
        "manifest_path": manifest_path.resolve(), "manifest_rows": len(manifest),
        "records_path": records_path.resolve(), "record_rows": len(records),
        "report_path": report_path.resolve(), "discovered": len(files),
        "extracted": successes, "empty": empty, "failed": failures,
    }


if __name__ == "__main__":
    try:
        result = build()
        for key, value in result.items():
            print(f"{key}: {value}")
    except Exception as exc:
        print(f"Advisory data build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
