"""Farmer-facing Streamlit UI reusing PaddyGuard's existing inference pipeline."""
from __future__ import annotations

import cv2
import hashlib
import io
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from predict import PaddyPredictor
from src.app_support import (
    LANGUAGES, disease_info, farmer_report, make_result_row, process_batch,
    read_zip_images, results_csv, result_is_reliable, severity_band_text,
    tr, validate_image_bytes,
)
from src.severity.analyzer import create_severity_overlay
from src.utils.safety import prediction_is_reliable

st.set_page_config(page_title="PaddyGuard-TN", page_icon="🌾", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 1rem; padding-bottom: 2rem;}
.pg-header {background:linear-gradient(110deg,#e9f4e8,#f7fbf4);border:1px solid #d6e6d1;border-radius:16px;padding:1.1rem 1.4rem;margin:.6rem 0 1rem}
.pg-action {background:#f0f7ed;border-left:5px solid #3a7d44;padding:1rem 1.2rem;border-radius:8px}
div[data-testid="stMetric"] {background:#fbfdf9;border:1px solid #e3ece0;padding:.65rem;border-radius:10px}
[data-testid="stSidebar"] {background:#f4f8f1;border-right:1px solid #dce8d7}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding:1.1rem 1rem 1.5rem}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {gap:.2rem}
[data-testid="stSidebar"] [data-testid="stRadio"] label {box-sizing:border-box;min-height:2.65rem;padding:.55rem .7rem;margin:.12rem 0;border:1px solid transparent;border-radius:10px;align-items:center}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {background:#e4f0df;border-color:#cfdfc9;color:#205b31;font-weight:650}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {background:#edf5e9}
.pg-brand {font-size:1.2rem;font-weight:750;color:#245c32;padding:.75rem 0 .25rem;line-height:1.4}
.pg-menu-title {font-size:.78rem;letter-spacing:.08em;color:#5e7460;font-weight:700;margin:.2rem 0 .55rem}
[data-testid="stSidebar"] hr {margin:.85rem 0}
[data-testid="stMain"] .st-key-home_diagnose_button button {background:#28683a;color:white;border:1px solid #205b31;border-radius:10px;min-height:3rem;padding:.65rem 1.1rem;font-weight:650}
[data-testid="stMain"] .st-key-home_diagnose_button button:hover {background:#205b31;color:white;border-color:#205b31}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor() -> PaddyPredictor:
    project_root = Path(__file__).resolve().parent
    checkpoint = project_root / "results" / "checkpoints" / "mobilenetv3_small_100" / "repeat_0_fold_1_best.pt"
    splits = project_root / "data" / "processed" / "splits.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Required PaddyGuard checkpoint is missing: {checkpoint}")
    if not splits.is_file():
        raise FileNotFoundError(f"Required class mapping is missing: {splits}")
    return PaddyPredictor(checkpoint_path=checkpoint, splits_path=splits)


def _uid(filename: str, data: bytes) -> str:
    return hashlib.sha256(filename.encode("utf-8") + data).hexdigest()[:16]


def _store_uploads() -> None:
    selected = st.session_state.setdefault("selected_images", {})
    removed = st.session_state.setdefault("removed_upload_ids", set())
    uploads = list(st.session_state.get("image_upload", []) or [])
    uploads += list(st.session_state.get("folder_upload", []) or [])
    for uploaded in uploads:
        data = uploaded.getvalue()
        uid = _uid(uploaded.name, data)
        if uid not in removed:
            selected[uid] = {"filename": uploaded.name, "data": data}
    zipped = st.session_state.get("zip_upload")
    if zipped is not None:
        entries, warnings_out = read_zip_images(zipped.getvalue(), zipped.name)
        st.session_state.zip_warnings = warnings_out
        for filename, data in entries:
            uid = _uid(filename, data)
            if uid not in removed:
                selected[uid] = {"filename": filename, "data": data}


def _png_bytes(rgb_array) -> bytes:
    output = io.BytesIO()
    Image.fromarray(rgb_array.astype("uint8")).save(output, format="PNG")
    return output.getvalue()


def _run_one(predictor: PaddyPredictor, uid: str, item: dict) -> dict:
    filename, data = item["filename"], item["data"]
    _, invalid_reason = validate_image_bytes(data, filename)
    if invalid_reason:
        return {"uid": uid, "filename": filename, "error": invalid_reason, "reliable": False}
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix.lower(), delete=False) as handle:
        handle.write(data)
        image_path = Path(handle.name)
    try:
        report = predictor.predict_image(image_path, include_visuals=True)
        visuals = report.pop("_visuals", {})
        top = report.get("top3_predictions", [])
        margin = top[0]["probability"] - top[1]["probability"] if len(top) > 1 else 0.0
        reliable = result_is_reliable(report) and prediction_is_reliable(
            report.get("confidence", 0), margin, report.get("ood_detection", {}).get("is_ood", True))
        # Avoid presenting legacy RAG text whose source provenance has not been verified.
        report.pop("advisory", None)
        cam = visuals.get("cam_overlay")
        bgr = cv2.imread(str(image_path))
        if cam is not None and bgr is not None:
            severity = report["severity_assessment"]
            overlay = create_severity_overlay(bgr, visuals.get("leaf_mask"), visuals.get("lesion_mask"),
                severity["lesion_percentage"], severity["irri_ses_score"], severity["irri_ses_band"])
            visuals = {"cam_png": _png_bytes(cam), "severity_png": _png_bytes(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))}
        else:
            visuals = {}
        return {"uid": uid, "filename": filename, "thumbnail": data, "report": report,
                "reliable": reliable, "visuals": visuals,
                "analysis_date": datetime.now().astimezone().isoformat(timespec="seconds")}
    except Exception as exc:
        return {"uid": uid, "filename": filename, "thumbnail": data,
                "error": f"{type(exc).__name__}: {exc}", "reliable": False,
                "analysis_date": datetime.now().astimezone().isoformat(timespec="seconds")}
    finally:
        image_path.unlink(missing_ok=True)


def _render_result(result: dict, language: str) -> None:
    with st.expander(result["filename"], expanded=False):
        if result.get("thumbnail"):
            image, _ = validate_image_bytes(result["thumbnail"], result["filename"])
            if image:
                st.image(image, use_container_width=True)
        if result.get("error"):
            key = result["error"] if result["error"] in {"oversize", "unsupported", "invalid"} else "error"
            st.error(tr(key, language))
            return
        report = result["report"]
        info = disease_info(report.get("predicted_class", ""), language)
        st.subheader(tr("possible_disease", language))
        st.markdown(f"### {info['label'] if info else report.get('predicted_class', '—')}")
        st.caption(f"{tr('score', language)}: {report.get('confidence', 0):.1%} · {tr('prototype', language)}")
        if result.get("reliable"):
            st.success(tr("reliable_caveat", language))
        else:
            st.error(tr("uncertain", language))
        severity = report.get("severity_assessment", {})
        st.subheader(tr("severity", language))
        left, right = st.columns(2)
        left.metric(tr("lesion_pct", language), f"{severity.get('lesion_percentage', 0):.2f}%")
        right.metric(tr("ses", language), f"{severity.get('irri_ses_score', '—')} ({severity_band_text(str(severity.get('irri_ses_band', '—')), language)})")
        st.caption(tr("severity_caveat", language))
        if result.get("visuals", {}).get("severity_png"):
            st.image(result["visuals"]["severity_png"], use_container_width=True)
        st.subheader(tr("advisory", language))
        if not result.get("reliable"):
            st.info(tr("uncertain", language))
        elif info and info.get("records"):
            st.markdown(f"**{tr('meaning', language)}**  \n{info['meaning']}")
            st.markdown(f"**{tr('symptoms', language)}**  \n{info['symptoms']}")
            st.markdown(f"**{tr('ipm', language)}**  \n{info['ipm']}")
            st.markdown(f"**{tr('treatment', language)}**  \n{tr('chemical_withheld', language)}")
            for record in info["records"]:
                st.caption(f"{record['record_id']} · {record['source_page']}")
            st.markdown(f"[{tr('source_note', language)} — TNAU]({info['url']})")
        else:
            st.info(tr("unavailable", language))
        st.download_button(tr("download_report", language), farmer_report(result, language, result.get("analysis_date")),
            file_name=f"{Path(result['filename']).stem}_PaddyGuard.txt", mime="text/plain; charset=utf-8",
            key=f"report_{result['uid']}_{language}")
        if st.toggle(tr("tech", language), key=f"tech_{result['uid']}_{language}"):
            st.write({"Top predictions": report.get("top3_predictions"), "OOD": report.get("ood_detection"),
                      "Advisory": "withheld" if not result.get("reliable") else "record-gated"})
            if result.get("visuals", {}).get("cam_png"):
                st.image(result["visuals"]["cam_png"], caption=tr("gradcam", language), use_container_width=True)


def _render_home(language: str) -> None:
    st.markdown(f"### {tr('home', language)}")
    st.write(tr("home_intro", language))
    st.button("🌱  " + ("Check Paddy Leaf" if language == "English" else "நெல் இலையைப் பரிசோதிக்கவும்"),
        type="primary", key="home_diagnose_button", on_click=_navigate_to_diagnose)
    st.info(tr("prototype", language))
    st.markdown(tr("home_steps", language))
    st.warning(tr("home_caveat", language))


def _navigate_to_diagnose() -> None:
    """Streamlit invokes this callback before rerunning and building the nav widget."""
    st.session_state["navigation"] = "diagnose"


def _render_diagnose(language: str) -> None:
    st.subheader(tr("diagnose", language))
    st.write(tr("photo_tip", language))
    st.file_uploader(tr("upload", language), type=["jpg", "jpeg", "png"], accept_multiple_files=True,
        key="image_upload", on_change=_store_uploads)
    st.file_uploader(tr("folder_upload", language), type=["jpg", "jpeg", "png"], accept_multiple_files="directory",
        key="folder_upload", on_change=_store_uploads)
    st.file_uploader(tr("zip_upload", language), type=["zip"], accept_multiple_files=False,
        key="zip_upload", on_change=_store_uploads)
    for warning in st.session_state.get("zip_warnings", []):
        st.warning(f"{tr('zip_warning', language)}: {warning}")
    selected = st.session_state.selected_images
    st.caption(f"{tr('selected', language)} · {len(selected)}")
    if selected:
        if st.button(tr("clear_selected", language), key="clear_selected"):
            st.session_state.removed_upload_ids.update(selected)
            selected.clear()
            st.rerun()
        for uid, item in list(selected.items()):
            cols = st.columns([1, 5, 1])
            image, why = validate_image_bytes(item["data"], item["filename"])
            if image:
                cols[0].image(image, width=84)
            elif why:
                cols[0].warning(tr(why, language))
            cols[1].write(item["filename"])
            if cols[2].button(tr("remove", language), key=f"remove_{uid}"):
                selected.pop(uid, None)
                st.session_state.removed_upload_ids.add(uid)
                st.rerun()
        if st.button(tr("analyze_all", language), type="primary", use_container_width=True):
            try:
                predictor = load_predictor()
                progress = st.progress(0.0, text=tr("analyzing", language))
                items = list(selected.items())
                batch = process_batch(items, lambda uid, item: _run_one(predictor, uid, item),
                    lambda index, total: progress.progress(index / total, text=f"{tr('analyzing', language)} {index}/{total}"))
                progress.empty()
                prior = {item["uid"] for item in st.session_state.analysis_results}
                st.session_state.analysis_results.extend(item for item in batch if item["uid"] not in prior)
                good = sum(not item.get("error") for item in batch)
                uncertain = sum(not item.get("reliable") and not item.get("error") for item in batch)
                failed = sum(bool(item.get("error")) for item in batch)
                a, b, c = st.columns(3)
                a.metric(tr("processed", language), good)
                b.metric(tr("uncertain_results", language), uncertain)
                c.metric(tr("failed_results", language), failed)
                st.success(tr("status_ok", language))
            except Exception as exc:
                st.error(f"{tr('error', language)} {type(exc).__name__}: {exc}")
    else:
        st.info(tr("no_images", language))
    if st.session_state.analysis_results:
        st.divider()
        for result in reversed(st.session_state.analysis_results[-5:]):
            _render_result(result, language)


def _render_history(language: str) -> None:
    st.subheader(tr("history", language))
    results = st.session_state.analysis_results
    if not results:
        st.info(tr("clear", language))
        return
    rows = [make_result_row(result, language) for result in results]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    selected_uid = st.selectbox(tr("view_individual", language), options=[result["uid"] for result in results],
        format_func=lambda uid: next(result["filename"] for result in results if result["uid"] == uid), key=f"history_detail_{language}")
    selected_result = next(result for result in results if result["uid"] == selected_uid)
    _render_result(selected_result, language)
    st.download_button(tr("download_csv", language), results_csv(results, language),
        file_name="PaddyGuard_results.csv", mime="text/csv; charset=utf-8-sig", key=f"csv_{language}")
    st.caption(tr("history_session", language))
    if not st.session_state.get("confirm_clear_history"):
        if st.button(tr("clear_history", language)):
            st.session_state.confirm_clear_history = True
            st.rerun()
    else:
        st.warning(tr("confirm_clear_text", language))
        yes, no = st.columns(2)
        if yes.button(tr("confirm_clear", language)):
            st.session_state.analysis_results = []
            st.session_state.confirm_clear_history = False
            st.rerun()
        if no.button(tr("cancel", language)):
            st.session_state.confirm_clear_history = False
            st.rerun()


def _render_treatment(language: str) -> None:
    st.subheader(tr("treatment_page", language))
    results = [result for result in st.session_state.analysis_results if result.get("report")]
    if not results:
        st.info(tr("screen_first", language))
        return
    selected_uid = st.selectbox(tr("screening_result", language), options=[result["uid"] for result in results],
        format_func=lambda uid: next(result["filename"] for result in results if result["uid"] == uid))
    result = next(result for result in results if result["uid"] == selected_uid)
    if not result.get("reliable"):
        st.error(tr("uncertain", language))
        return
    disease = result["report"].get("predicted_class", "")
    info = disease_info(disease, language)
    if not info or not info.get("records"):
        st.info(tr("unavailable", language))
        return
    st.markdown(f"### {info['label']}")
    st.write(info["ipm"])
    st.info(tr("severity_guidance_unavailable", language))
    st.warning(tr("chemical_withheld", language))
    st.markdown(f"[TNAU source page]({info['url']})")
    record = info["records"][0]
    st.caption(f"{record['record_id']} · {record['source_page']} · TNAU Agritech Portal. {tr('treatment_note', language)}")


def _render_about(language: str) -> None:
    st.subheader(tr("about", language))
    for key in ("about_model", "about_data", "about_severity", "about_advisory", "about_privacy"):
        st.write(tr(key, language))


for key, default in (("selected_images", {}), ("removed_upload_ids", set()), ("analysis_results", []), ("navigation", "home")):
    if key not in st.session_state:
        st.session_state[key] = default
with st.sidebar:
    language = st.radio("Language / மொழி", LANGUAGES, key="language")
    st.markdown(f"<div class='pg-brand'>🌾 {tr('title', language)}</div>", unsafe_allow_html=True)
    st.caption(tr("subtitle", language))
    st.divider()
    st.markdown(f"<div class='pg-menu-title'>{tr('menu', language).upper()}</div>", unsafe_allow_html=True)
    page_keys = ("home", "diagnose", "history", "treatment_page", "about")
    page_labels = [tr(key, language) for key in page_keys]
    current = st.session_state.get("navigation", "home")
    page_key = st.radio("Navigation", page_keys, index=page_keys.index(current) if current in page_keys else 0,
        format_func=lambda key: tr(key, language), label_visibility="collapsed", key="navigation")
st.markdown(f"<div class='pg-header'><h1>🌾 {tr(page_key, language)}</h1><div>{tr('subtitle', language)}</div></div>", unsafe_allow_html=True)
if page_key == "home":
    _render_home(language)
elif page_key == "diagnose":
    _render_diagnose(language)
elif page_key == "history":
    _render_history(language)
elif page_key == "treatment_page":
    _render_treatment(language)
else:
    _render_about(language)
