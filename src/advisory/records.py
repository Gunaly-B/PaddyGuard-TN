"""Load source-checked advisory records and fail closed for chemicals."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RECORDS_PATH = ROOT / "data" / "advisory" / "advisory_records.csv"


def load_advisory_records(path: Path = RECORDS_PATH) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def guidance_for_disease(disease: str, reliable: bool = True,
                         path: Path = RECORDS_PATH) -> list[dict[str, str]]:
    """Return only official, disease-matched cultural/IPM guidance.

    Uncertain predictions never receive disease-specific records. Chemical
    records are excluded unless current label evidence has been explicitly
    verified for the exact crop, disease and formulation.
    """
    if not reliable:
        return []
    output = []
    for row in load_advisory_records(path):
        if row.get("disease") != disease or row.get("evidence_status") != "official_page_checked":
            continue
        if row.get("guidance_type") == "chemical":
            if not chemical_advice_allowed(row):
                continue
        elif row.get("guidance_type") != "cultural_ipm":
            continue
        output.append(row)
    return output


def chemical_advice_allowed(row: dict[str, Any]) -> bool:
    required_evidence = (
        "disease", "target_crop", "active_ingredient", "formulation", "dose",
        "pre_harvest_interval_days", "label_issue_date", "label_source_url",
    )
    return (
        row.get("guidance_type") == "chemical"
        and row.get("evidence_status") == "official_label_checked"
        and row.get("chemical_verification_status") == "verified_current_label"
        and all(str(row.get(field, "")).strip() for field in required_evidence)
    )
