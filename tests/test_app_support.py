import csv
import io
import zipfile
from pathlib import Path

from PIL import Image

from src.app_support import (
    GUIDANCE, TEXT, disease_info, farmer_report, make_result_row,
    process_batch, results_csv, result_is_reliable, severity_band_text, tr, validate_image_bytes,
    read_zip_images,
)
from src.advisory.records import chemical_advice_allowed, guidance_for_disease


def make_png(color=(20, 150, 30)):
    image = Image.new("RGB", (16, 12), color)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def sample_result(name="sample.png", disease="Brown spot", confidence=0.9,
                  second=0.2, ood=False):
    return {
        "uid": "abc",
        "filename": name,
        "reliable": confidence >= 0.7 and confidence-second >= 0.15 and not ood,
        "report": {
            "predicted_class": disease,
            "confidence": confidence,
            "top3_predictions": [
                {"class_name": disease, "probability": confidence, "percentage": confidence*100},
                {"class_name": "Other", "probability": second, "percentage": second*100},
            ],
            "ood_detection": {"is_ood": ood, "status": "OOD_SUSPECTED" if ood else "IN_DISTRIBUTION"},
            "severity_assessment": {"lesion_percentage": 12.5, "irri_ses_score": 5},
        },
    }


def test_bilingual_dictionary_and_review_note_are_present():
    assert all(len(value) == 2 and value[0] and value[1] for value in TEXT.values())
    assert tr("language_review", "தமிழ்")
    assert "(Brown spot)" in disease_info("Brown spot", "தமிழ்")["label"]
    assert "false smut" in disease_info("Leaf smut", "English")["meaning"]
    assert severity_band_text("Severe", "தமிழ்") == "தீவிரம்"


def test_image_validation_accepts_png_and_rejects_corrupt_oversized_and_wrong_suffix():
    payload = make_png()
    image, reason = validate_image_bytes(payload, "leaf.png")
    assert image is not None and image.size == (16, 12) and reason is None
    assert validate_image_bytes(b"bad", "leaf.png")[1] == "invalid"
    assert validate_image_bytes(payload, "leaf.gif")[1] == "unsupported"
    assert validate_image_bytes(payload, "leaf.png", max_bytes=2)[1] == "oversize"


def test_safety_gate_rejects_low_confidence_low_margin_and_ood():
    assert result_is_reliable(sample_result()["report"])
    assert not result_is_reliable(sample_result(confidence=0.69, second=0.1)["report"])
    assert not result_is_reliable(sample_result(confidence=0.8, second=0.7)["report"])
    assert not result_is_reliable(sample_result(ood=True)["report"])


def test_bulk_processing_continues_after_one_image_fails():
    seen = []
    def analyzer(uid, item):
        seen.append(uid)
        if uid == "bad":
            raise ValueError("corrupt image")
        return {"uid": uid, "filename": item["filename"], "report": {"ok": True}}
    result = process_batch([("one", {"filename": "one.png"}),
                            ("bad", {"filename": "bad.png"}),
                            ("three", {"filename": "three.jpg"})], analyzer)
    assert seen == ["one", "bad", "three"]
    assert result[1]["error"].startswith("ValueError")
    assert result[2]["filename"] == "three.jpg"


def test_unverified_leaf_smut_and_uncertain_results_have_no_source_or_advice():
    uncertain = sample_result(ood=True)
    row = make_result_row(uncertain, "தமிழ்")
    assert not row[tr("table_source", "தமிழ்")]
    assert row[tr("advisory_status", "தமிழ்")] == tr("unavailable", "தமிழ்")
    leaf_smut = sample_result(disease="Leaf smut")
    row = make_result_row(leaf_smut, "English")
    assert row[tr("advisory_status", "English")] == tr("unavailable", "English")
    assert not row[tr("table_source", "English")]


def test_reports_include_date_disclaimer_and_only_verified_links():
    good = sample_result()
    csv_text = results_csv([good], "தமிழ்", "2026-10-21T10:00:00+05:30").decode("utf-8-sig")
    assert "2026-10-21T10:00:00+05:30" in csv_text
    assert tr("report_disclaimer", "தமிழ்") in csv_text
    assert GUIDANCE["Brown spot"]["url"] in csv_text
    uncertain = sample_result(ood=True)
    row = make_result_row(uncertain, "English")
    assert not row[tr("table_source", "English")]
    report = farmer_report(uncertain, "தமிழ்", "2026-10-21")
    assert tr("uncertain", "தமிழ்") in report
    assert "Mancozeb" not in report and "kg/ha" not in report


def test_zip_upload_reads_images_without_extracting_and_rejects_bad_archives():
    payload = make_png()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("field/leaf.png", payload)
        bundle.writestr("notes.txt", "ignored")
    images, warnings_out = read_zip_images(archive.getvalue(), "photos.zip")
    assert images == [("leaf.png", payload)]
    assert not warnings_out
    assert read_zip_images(b"broken", "photos.zip")[1] == ["invalid_archive"]


def test_guidance_records_are_disease_matched_and_fail_closed_for_chemicals(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text(
        "record_id,disease,guidance_type,guidance_text,source_url,evidence_status,chemical_verification_status,label_source_url\n"
        "a,Brown spot,cultural_ipm,Use disease-free seed,https://example.test,official_page_checked,not_applicable,\n"
        "b,Brown spot,chemical,Spray product,https://example.test,official_page_checked,pending,\n"
        "c,Leaf smut,cultural_ipm,Other disease,https://example.test,official_page_checked,not_applicable,\n",
        encoding="utf-8")
    assert [row["record_id"] for row in guidance_for_disease("Brown spot", True, path)] == ["a"]
    assert guidance_for_disease("Brown spot", False, path) == []
    assert not chemical_advice_allowed({"guidance_type": "chemical", "chemical_verification_status": "pending", "label_source_url": ""})
    label_evidence = {"guidance_type": "chemical", "evidence_status": "official_label_checked",
        "chemical_verification_status": "verified_current_label", "disease": "Brown spot", "target_crop": "Rice",
        "active_ingredient": "Example", "formulation": "WP", "dose": "Example rate",
        "pre_harvest_interval_days": "10", "label_issue_date": "2026-01-01", "label_source_url": "https://official.test/label"}
    assert chemical_advice_allowed(label_evidence)
    label_evidence.pop("pre_harvest_interval_days")
    assert not chemical_advice_allowed(label_evidence)


def test_english_advice_shown_in_ui_matches_the_cited_record():
    info = disease_info("Brown spot", "English")
    assert info["records"]
    assert info["ipm"] == info["records"][0]["guidance_text"]
    assert info["url"] == info["records"][0]["source_url"]


def test_folder_upload_label_and_source_report_are_present():
    assert tr("folder_upload", "English") == "Or choose a folder of leaf photos"
    assert tr("folder_upload", "தமிழ்")
    report = Path(__file__).resolve().parents[1] / "data" / "advisory" / "advisory_diagnostic_report.md"
    text = report.read_text(encoding="utf-8")
    assert "false smut, not model class Leaf smut" in text
    assert "Brown spot" in text and "Bacterial leaf blight" in text
    assert "Severity-specific records: 0" in text
    assert "Chemical label verified records: 0" in text


def test_advisory_csv_has_traceable_extended_fields():
    from src.advisory.records import RECORDS_PATH, load_advisory_records
    records = load_advisory_records()
    assert len(records) == 2
    assert all(row["source_id"] and row["recommendation"] for row in records)
    assert all(row["severity_condition"] == "" for row in records)
    assert {row["advice_type"] for row in records} == {"general cultural/IPM"}
