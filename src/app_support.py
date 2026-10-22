"""Pure, testable support for the Streamlit farmer interface."""
from __future__ import annotations

import csv
import io
from pathlib import Path
import warnings
import hashlib
import zipfile
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError
from src.advisory.records import guidance_for_disease

LANGUAGES = ("English", "தமிழ்")

TEXT = {
    "title": ("PaddyGuard-TN", "PaddyGuard-TN"),
    "subtitle": ("Rice leaf screening for Tamil Nadu farmers", "தமிழ்நாட்டு விவசாயிகளுக்கான நெல் இலை பரிசோதனை"),
    "prototype": (
        "Research screening tool only. This is not a confirmed field diagnosis.",
        "இது ஆராய்ச்சி நோக்கிலான ஆரம்ப பரிசோதனை மட்டுமே. வயல்வெளி உறுதிப்படுத்தப்பட்ட நோயறிதல் அல்ல.",
    ),
    "language_review": (
        "Tamil agricultural wording is a draft and needs review by a Tamil-speaking agricultural expert.",
        "தமிழ் வேளாண் சொற்கள் வரைவு நிலையில் உள்ளன; தமிழ் பேசும் வேளாண் நிபுணரின் மதிப்பாய்வு தேவை.",
    ),
    "upload": ("Add rice-leaf photos (JPG, JPEG, PNG; up to 10 MB each)", "நெல் இலைப் படங்களைச் சேர்க்கவும் (JPG, JPEG, PNG; ஒவ்வொன்றும் 10 MB வரை)"),
    "photo_tip": (
        "For a clearer result, photograph one leaf in focus with even light and little background.",
        "தெளிவான முடிவுக்கு, ஒரே இலையை நன்றாக ஒளியில், பின்னணி குறைவாகவும் தெளிவாகவும் படமெடுக்கவும்.",
    ),
    "selected": ("Selected photos", "தேர்ந்தெடுத்த படங்கள்"),
    "remove": ("Remove", "நீக்கு"),
    "analyze_all": ("Analyse All", "அனைத்தையும் பரிசோதிக்கவும்"),
    "analyzing": ("Analysing each photo separately…", "ஒவ்வொரு படத்தையும் தனித்தனியாகப் பரிசோதிக்கிறது…"),
    "summary": ("Screening summary", "பரிசோதனைச் சுருக்கம்"),
    "total": ("Images selected", "தேர்ந்தெடுத்த படங்கள்"),
    "analyzed_ok": ("Analysed successfully", "வெற்றிகரமாகப் பரிசோதித்தவை"),
    "failed_count": ("Could not analyse", "பரிசோதிக்க முடியாதவை"),
    "uncertain_count": ("Uncertain results", "உறுதியற்ற முடிவுகள்"),
    "results": ("Individual results", "ஒவ்வொரு படத்தின் முடிவுகள்"),
    "possible_disease": ("Possible match", "பொருந்தக்கூடிய நோய்"),
    "next_action": ("Recommended next step", "அடுத்து செய்ய வேண்டியது"),
    "uncertain": (
        "The result is uncertain or the photo may be outside the model's scope. Do not use disease-specific advice. Take a clearer photo and ask a local TNAU KVK or agricultural officer to check the crop.",
        "முடிவு உறுதியற்றது அல்லது படம் இந்த மாதிரியின் வரம்பிற்கு வெளியே இருக்கலாம். நோய்க்குரிய ஆலோசனையைப் பின்பற்ற வேண்டாம். தெளிவான படத்தை எடுத்து, உள்ளூர் TNAU KVK அல்லது வேளாண் அலுவலரிடம் பயிரைச் சரிபார்க்கவும்.",
    ),
    "reliable_caveat": (
        "No uncertainty flag was triggered. This is still a model suggestion, not a confirmed diagnosis; verify symptoms in the field.",
        "உறுதியின்மை எச்சரிக்கை எழவில்லை. இருந்தாலும் இது மாதிரியின் கணிப்பு மட்டுமே; உறுதியான நோயறிதல் அல்ல. வயலில் அறிகுறிகளைச் சரிபார்க்கவும்.",
    ),
    "severity": ("Estimated leaf damage", "இலை சேதத்தின் மதிப்பீடு"),
    "lesion_pct": ("Lesion-like area", "பாதிப்பு போன்ற பகுதி"),
    "ses": ("Estimated SES score", "மதிப்பிடப்பட்ட SES மதிப்பெண்"),
    "severity_caveat": (
        "Heuristic estimate only; not validated against expert severity scores.",
        "நிற அடிப்படையிலான தோராய மதிப்பீடு மட்டுமே; நிபுணர் தீவிர மதிப்பீடுகளுடன் சரிபார்க்கப்படவில்லை.",
    ),
    "advisory": ("Verified guidance and sources", "சரிபார்க்கப்பட்ட வழிகாட்டலும் ஆதாரங்களும்"),
    "meaning": ("What this possible match means", "இந்த சாத்தியமான பொருத்தம் எதைக் குறிக்கிறது"),
    "symptoms": ("Symptoms to check", "சரிபார்க்க வேண்டிய அறிகுறிகள்"),
    "ipm": ("Cultural and IPM steps", "பயிர் பராமரிப்பு மற்றும் ஒருங்கிணைந்த மேலாண்மை"),
    "treatment": ("Further treatment", "மேலும் சிகிச்சை"),
    "unavailable": (
        "Disease-specific verified guidance is not available in this app. Please contact a local TNAU KVK or agricultural officer.",
        "இந்த செயலியில் இந்நோய்க்கான சரிபார்க்கப்பட்ட வழிகாட்டல் இல்லை. உள்ளூர் TNAU KVK அல்லது வேளாண் அலுவலரைத் தொடர்புகொள்ளவும்.",
    ),
    "chemical_withheld": (
        "No pesticide product or dose is provided. Check current official product labels and seek local expert guidance.",
        "பூச்சிக்கொல்லி தயாரிப்பு அல்லது அளவு இங்கு வழங்கப்படவில்லை. நடப்பு அதிகாரப்பூர்வ தயாரிப்பு லேபிளையும் உள்ளூர் நிபுணர் ஆலோசனையையும் சரிபார்க்கவும்.",
    ),
    "tech": ("Technical Details", "தொழில்நுட்ப விவரங்கள்"),
    "gradcam": ("Grad-CAM highlights influential image regions; it does not confirm a disease.", "Grad-CAM படத்தில் மாதிரி கவனித்த பகுதிகளை காட்டுகிறது; இது நோயை உறுதிப்படுத்தாது."),
    "download_csv": ("Download results CSV", "முடிவுகள் CSV-ஐப் பதிவிறக்கவும்"),
    "download_report": ("Download farmer-friendly report", "விவசாயிக்கான அறிக்கையைப் பதிவிறக்கவும்"),
    "advisory_status": ("Advisory status", "ஆலோசனை நிலை"),
    "score": ("Model score", "மாதிரி மதிப்பெண்"),
    "uncertain_status": ("Uncertain — advice withheld", "உறுதியற்றது — ஆலோசனை நிறுத்தப்பட்டது"),
    "available_status": ("Guidance available", "வழிகாட்டல் உள்ளது"),
    "error": ("This image could not be analysed. Other selected images will continue.", "இந்தப் படத்தைப் பரிசோதிக்க முடியவில்லை. தேர்ந்தெடுத்த மற்ற படங்கள் தொடர்ந்து பரிசோதிக்கப்படும்."),
    "oversize": ("File is larger than 10 MB.", "கோப்பு 10 MB-ஐ விடப் பெரியது."),
    "unsupported": ("Only JPG, JPEG, and PNG images are supported.", "JPG, JPEG, PNG படங்கள் மட்டுமே ஆதரிக்கப்படுகின்றன."),
    "invalid": ("This file is not a readable image or its dimensions are too large.", "இந்தக் கோப்பைப் படமாகத் திறக்க முடியவில்லை அல்லது அதன் அளவு மிகப் பெரியது."),
    "no_images": ("Add at least one image before analysis.", "பரிசோதனைக்கு முன் குறைந்தது ஒரு படத்தையாவது சேர்க்கவும்."),
    "source_note": ("Official TNAU source page", "அதிகாரப்பூர்வ TNAU ஆதாரப் பக்கம்"),
    "filename": ("Filename", "கோப்புப் பெயர்"),
    "analysis_date": ("Analysis date", "பரிசோதித்த தேதி"),
    "report_model_output": ("Unconfirmed model suggestion", "உறுதிப்படுத்தப்படாத மாதிரி கணிப்பு"),
    "report_ood": ("OOD / uncertainty status", "பட வரம்பு / உறுதியின்மை நிலை"),
    "table_class": ("Possible disease (model output)", "சாத்தியமான நோய் (மாதிரி முடிவு)"),
    "table_ood": ("OOD status", "பட வரம்பு நிலை"),
    "table_uncertain": ("Uncertain", "உறுதியற்றது"),
    "table_severity_pct": ("Lesion area % (heuristic, unvalidated)", "பாதிப்பு பகுதி % (தோராயம், சரிபார்க்கப்படாதது)"),
    "table_severity_score": ("SES score (heuristic, unvalidated)", "SES மதிப்பெண் (தோராயம், சரிபார்க்கப்படாதது)"),
    "table_source": ("Official source link", "அதிகாரப்பூர்வ ஆதார இணைப்பு"),
    "table_error": ("Processing issue", "செயலாக்கச் சிக்கல்"),
    "report_disclaimer": (
        "Research prototype. Model outputs are not confirmed diagnoses. Severity is a heuristic estimate, not validated against expert scores. Verify advice locally. No pesticide products or doses are included.",
        "ஆராய்ச்சி முன்மாதிரி. மாதிரி முடிவுகள் உறுதிப்படுத்தப்பட்ட நோயறிதல் அல்ல. தீவிரம் தோராய மதிப்பீடு மட்டுமே; நிபுணர் மதிப்பெண்களுடன் சரிபார்க்கப்படவில்லை. உள்ளூர் நிபுணரிடம் ஆலோசனையை உறுதிப்படுத்தவும். பூச்சிக்கொல்லி தயாரிப்பு அல்லது அளவு சேர்க்கப்படவில்லை.",
    ),
    "status_ok": ("Screening complete", "ஆரம்பப் பரிசோதனை முடிந்தது"),
    "clear": ("No image selected", "படம் தேர்ந்தெடுக்கப்படவில்லை"),
    "ood_suspected": ("OOD suspected", "படம் மாதிரியின் வரம்பிற்கு வெளியே இருக்கலாம்"),
    "in_domain": ("No OOD flag", "பட வரம்பு எச்சரிக்கை இல்லை"),
    "ood_unknown": ("Not available", "தகவல் இல்லை"),
    "yes": ("Yes", "ஆம்"),
    "no": ("No", "இல்லை"),
    "band_low": ("Low", "குறைவு"),
    "band_moderate": ("Moderate", "மிதமானது"),
    "band_severe": ("Severe", "தீவிரம்"),
    "home": ("Home", "முகப்பு"), "diagnose": ("Diagnose", "பரிசோதனை"),
    "history": ("My Results", "என் முடிவுகள்"), "treatment_page": ("Treatment Advice", "சிகிச்சை ஆலோசனை"),
    "about": ("About", "தகவல்"), "zip_upload": ("Or upload a ZIP folder of leaf photos", "அல்லது இலைப் படங்கள் உள்ள ZIP கோப்பைப் பதிவேற்றவும்"),
    "folder_upload": ("Or choose a folder of leaf photos", "அல்லது இலைப் படங்கள் உள்ள கோப்புறையைத் தேர்ந்தெடுக்கவும்"),
    "processed": ("Processed", "செயலாக்கப்பட்டவை"), "uncertain_results": ("Uncertain", "உறுதியற்றவை"),
    "failed_results": ("Failed", "தோல்வியடைந்தவை"), "view_individual": ("View an individual result", "தனிப்பட்ட முடிவைப் பார்க்கவும்"),
    "menu": ("Menu", "பட்டியல்"), "screening_result": ("Screening result", "பரிசோதனை முடிவு"),
    "severity_guidance_unavailable": ("Severity-specific guidance is not available from the checked sources. The damage estimate is not used to choose treatment.", "சரிபார்க்கப்பட்ட ஆதாரங்களில் தீவிரத்திற்கேற்ற தனி வழிகாட்டல் இல்லை. சேத மதிப்பீட்டின் அடிப்படையில் சிகிச்சை தேர்வு செய்யப்படுவதில்லை."),
    "clear_selected": ("Clear selected photos", "தேர்ந்தெடுத்த படங்களை அழிக்கவும்"),
    "home_intro": ("Screen a rice leaf photo, review the uncertainty and damage estimate, then check linked TNAU guidance.", "நெல் இலைப் படத்தைப் பரிசோதித்து, உறுதியின்மை மற்றும் சேத மதிப்பீட்டைப் பார்த்து, TNAU வழிகாட்டலைச் சரிபார்க்கவும்."),
    "clear_history": ("Clear history", "வரலாற்றை அழிக்கவும்"),
    "home_steps": ("- Add clear JPG/PNG leaf photos, or upload a ZIP of photos.\n- Review the possible match, uncertainty warning, and heuristic damage estimate.\n- Use source-linked TNAU cultural/IPM guidance only when the safety checks pass.", "- தெளிவான JPG/PNG இலைப் படங்களைச் சேர்க்கவும் அல்லது படங்கள் உள்ள ZIP கோப்பைப் பதிவேற்றவும்.\n- சாத்தியமான பொருத்தம், உறுதியின்மை எச்சரிக்கை, தோராய சேத மதிப்பீடு ஆகியவற்றைப் பாருங்கள்.\n- பாதுகாப்புச் சோதனைகள் அனுமதித்தால் மட்டுமே ஆதார இணைப்புள்ள TNAU பயிர் பராமரிப்பு வழிகாட்டலைப் பயன்படுத்தவும்."),
    "home_caveat": ("Research prototype only. An OOD check cannot identify every unfamiliar image.", "ஆராய்ச்சி முன்மாதிரி மட்டுமே. அறியாத ஒவ்வொரு படத்தையும் OOD சோதனை கண்டறியும் என்று உறுதி இல்லை."),
    "zip_warning": ("ZIP entry skipped or rejected", "ZIP கோப்பிலுள்ள உருப்படி தவிர்க்கப்பட்டது அல்லது நிராகரிக்கப்பட்டது"),
    "history_session": ("History stays in this browser session and is not saved to a server database.", "முடிவு வரலாறு இந்த உலாவி அமர்வில் மட்டுமே இருக்கும்; சேவையகத்தில் சேமிக்கப்படாது."),
    "confirm_clear_text": ("Clear all screening results from this session?", "இந்த அமர்விலுள்ள எல்லா பரிசோதனை முடிவுகளையும் அழிக்கவா?"),
    "confirm_clear": ("Confirm clear", "அழிப்பதை உறுதிப்படுத்தவும்"),
    "cancel": ("Cancel", "ரத்துசெய்"),
    "screen_first": ("Run a screening first to see matching source guidance.", "ஆதார வழிகாட்டலைப் பார்க்க முதலில் படத்தைப் பரிசோதிக்கவும்."),
    "about_model": ("PaddyGuard-TN uses the existing MobileNetV3-Small checkpoint, Grad-CAM, heuristic colour-based lesion segmentation, and OOD checks.", "PaddyGuard-TN ஏற்கனவே உள்ள MobileNetV3-Small மாதிரி, Grad-CAM, நிற அடிப்படையிலான தோராய பாதிப்பு பிரிப்பு, OOD சோதனைகள் ஆகியவற்றைப் பயன்படுத்துகிறது."),
    "about_data": ("The training collection contains 120 controlled-background images. External field-photo generalization has not been evaluated because an independent dataset is not available locally.", "பயிற்சித் தொகுப்பில் கட்டுப்படுத்தப்பட்ட பின்னணியுள்ள 120 படங்கள் உள்ளன. சுயாதீன தரவுத்தொகுப்பு உள்ளூரில் இல்லாததால் வயல்வெளிப் படங்களில் பொதுமைப்படுத்தல் மதிப்பிடப்படவில்லை."),
    "about_severity": ("Severity estimates have not been validated against expert-scored leaves. Tamil agricultural wording is provisional.", "தீவிர மதிப்பீடுகள் நிபுணர்கள் மதிப்பிட்ட இலைகளுடன் சரிபார்க்கப்படவில்லை. தமிழ் வேளாண் சொற்கள் இன்னும் வரைவு நிலையில் உள்ளன."),
    "about_advisory": ("Advice uses disease-matched cultural/IPM records with citations. Current applicable CIBRC label evidence is unavailable, so no chemical product or dose is shown.", "ஆலோசனை நோய்க்குப் பொருந்தும் ஆதார இணைப்புள்ள பயிர் பராமரிப்பு பதிவுகளைப் பயன்படுத்துகிறது. பொருந்தக்கூடிய நடப்பு CIBRC லேபிள் ஆதாரம் இல்லாததால் ரசாயனத் தயாரிப்பு அல்லது அளவு காட்டப்படாது."),
    "about_privacy": ("Uploaded images are processed in the current app session and are not added to the project dataset.", "பதிவேற்றிய படங்கள் தற்போதைய செயலி அமர்வில் மட்டுமே செயலாக்கப்படும்; திட்டத்தின் தரவுத்தொகுப்பில் சேர்க்கப்படாது."),
    "treatment_note": ("Prediction remains unconfirmed; severity is not used to select treatment.", "கணிப்பு உறுதிப்படுத்தப்படாதது; சிகிச்சையைத் தேர்ந்தெடுக்க தீவிர மதிப்பீடு பயன்படுத்தப்படுவதில்லை."),
}

# Each entry is a cautious summary of the linked TNAU page. Tamil terms remain
# provisional until reviewed by a Tamil-speaking crop-protection specialist.
GUIDANCE = {
    "Bacterial leaf blight": {
        "label": ("Bacterial leaf blight", "பாக்டீரியா இலைக்கருகல் (Bacterial leaf blight)"),
        "meaning": ("A possible bacterial leaf blight match. The model cannot confirm the cause.", "பாக்டீரியா இலைக்கருகல் நோயுடன் பொருந்தக்கூடும். காரணத்தை மாதிரி உறுதிப்படுத்த முடியாது."),
        "symptoms": ("TNAU describes yellow-to-straw lesions beginning at leaf margins, sometimes with wavy edges.", "TNAU விளக்கத்தில், இலை ஓரங்களில் மஞ்சள் முதல் வைக்கோல் நிறப் புண்கள் தோன்றி, சில நேரங்களில் அலைபோன்ற விளிம்புகளுடன் காணப்படும்."),
        "ipm": ("Use tolerant varieties, avoid excess nitrogen, maintain field sanitation, and avoid injuring seedlings during transplanting.", "தாங்கும் திறன் கொண்ட ரகங்களைப் பயன்படுத்தவும்; அதிக நைட்ரஜனைத் தவிர்க்கவும்; வயல் சுத்தத்தைப் பேணவும்; நாற்று நடும் போது காயப்படுத்த வேண்டாம்."),
        "url": "https://www.agritech.tnau.ac.in/expert_system/paddy/cpdisblb.html",
    },
    "Brown spot": {
        "label": ("Brown spot", "பழுப்பு புள்ளி நோய் (Brown spot)"),
        "meaning": ("A possible brown spot match. Similar-looking symptoms can have other causes.", "பழுப்பு புள்ளி நோயுடன் பொருந்தக்கூடும். இதேபோல் தோன்றும் அறிகுறிகளுக்கு வேறு காரணங்களும் இருக்கலாம்."),
        "symptoms": ("TNAU describes small brown dots that may become round or oval spots and merge into larger patches.", "TNAU விளக்கத்தில், சிறிய பழுப்பு புள்ளிகள் வட்டம் அல்லது நீள்வட்டப் புள்ளிகளாக வளர்ந்து, பெரிய திட்டுகளாக ஒன்றுசேரலாம்."),
        "ipm": ("Use disease-free seed, support balanced crop nutrition, and prevent water stress. Ask an agricultural officer about field-specific steps.", "நோயற்ற விதையைப் பயன்படுத்தவும்; பயிருக்குச் சமச்சீர் ஊட்டச்சத்தைப் பேணவும்; நீர் அழுத்தத்தைத் தவிர்க்கவும். வயலுக்கேற்ற நடைமுறைக்கு வேளாண் அலுவலரிடம் கேட்கவும்."),
        "url": "https://www.agritech.tnau.ac.in/expert_system/paddy/cpdisbrownspot.html",
    },
    "Leaf smut": {
        "label": ("Leaf smut", "இலை ஸ்மட் (Leaf smut)"),
        "meaning": ("The model suggests leaf smut, but verified TNAU leaf-smut guidance was not found for this interface. Leaf smut is distinct from false smut.", "மாதிரி இலை ஸ்மட் எனக் கணிக்கிறது; ஆனால் இந்த செயலிக்கான சரிபார்க்கப்பட்ட TNAU இலை ஸ்மட் வழிகாட்டல் கிடைக்கவில்லை. இலை ஸ்மட், பொய்க் கரிப்பூட்டிலிருந்து வேறுபட்டது."),
        "symptoms": ("Please have the leaf inspected by a qualified local agricultural expert; this app does not provide a verified symptom guide for leaf smut.", "தகுதியான உள்ளூர் வேளாண் நிபுணரிடம் இலையைச் சரிபார்க்கவும்; இலை ஸ்மட்டிற்கான சரிபார்க்கப்பட்ட அறிகுறி வழிகாட்டி இந்தச் செயலியில் இல்லை."),
        "ipm": ("Disease-specific cultural or IPM guidance is unavailable here. Contact a local TNAU KVK or agricultural officer.", "இந்நோய்க்கான பயிர் பராமரிப்பு அல்லது ஒருங்கிணைந்த மேலாண்மை வழிகாட்டல் இங்கு இல்லை. உள்ளூர் TNAU KVK அல்லது வேளாண் அலுவலரைத் தொடர்புகொள்ளவும்."),
        "url": "",
    },
}


def tr(key: str, language: str = "English") -> str:
    """Return a farmer-facing string from the central bilingual dictionary."""
    value = TEXT[key]
    return value[1] if language == "தமிழ்" else value[0]


def severity_band_text(band: str, language: str = "English") -> str:
    key = {"low": "band_low", "moderate": "band_moderate", "severe": "band_severe"}.get(band.lower())
    return tr(key, language) if key else band


def disease_info(disease: str, language: str = "English") -> Dict[str, Any] | None:
    info = GUIDANCE.get(disease)
    if not info:
        return None
    index = 1 if language == "தமிழ்" else 0
    rendered = {key: (value[index] if isinstance(value, tuple) else value) for key, value in info.items()}
    records = guidance_for_disease(disease)
    rendered["records"] = records
    rendered["url"] = records[0]["source_url"] if records else ""
    if records and language != "தமிழ்":
        rendered["ipm"] = records[0]["guidance_text"]
    return rendered


def read_zip_images(data: bytes, archive_name: str, *, max_entries: int = 100,
                    max_total_bytes: int = 100 * 1024 * 1024,
                    max_image_bytes: int = 10 * 1024 * 1024) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Read image members safely without extracting ZIP paths to disk."""
    if not archive_name.lower().endswith(".zip"):
        return [], ["not_zip"]
    images: list[tuple[str, bytes]] = []
    warnings_out: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = [entry for entry in archive.infolist() if not entry.is_dir()]
            if len(members) > max_entries:
                return [], ["too_many_entries"]
            declared_total = sum(entry.file_size for entry in members)
            if declared_total > max_total_bytes:
                return [], ["archive_too_large"]
            for entry in members:
                path = entry.filename.replace("\\", "/")
                parts = path.split("/")
                if path.startswith("/") or any(part in {"..", ""} for part in parts):
                    warnings_out.append("unsafe_path")
                    continue
                if Path(path).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                if entry.file_size > max_image_bytes:
                    warnings_out.append("oversize_member")
                    continue
                if entry.compress_size and entry.file_size / entry.compress_size > 200:
                    warnings_out.append("suspicious_compression")
                    continue
                try:
                    payload = archive.read(entry)
                except (OSError, RuntimeError, zipfile.BadZipFile):
                    warnings_out.append("unreadable_member")
                    continue
                if len(payload) > max_image_bytes:
                    warnings_out.append("oversize_member")
                    continue
                images.append((Path(path).name, payload))
    except (zipfile.BadZipFile, OSError, ValueError):
        return [], ["invalid_archive"]
    if not images and not warnings_out:
        warnings_out.append("no_supported_images")
    return images, warnings_out


def validate_image_bytes(data: bytes, filename: str, max_bytes: int = 10 * 1024 * 1024,
                         max_pixels: int = 25_000_000) -> Tuple[Image.Image | None, str | None]:
    """Validate format, payload size and decoded dimensions before inference."""
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in {"jpg", "jpeg", "png"}:
        return None, "unsupported"
    if len(data) > max_bytes:
        return None, "oversize"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                if probe.format not in {"JPEG", "PNG"} or probe.width * probe.height > max_pixels:
                    return None, "invalid"
                probe.verify()
            with Image.open(io.BytesIO(data)) as image:
                return ImageOps.exif_transpose(image).convert("RGB"), None
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        return None, "invalid"


def result_is_reliable(report: Dict[str, Any]) -> bool:
    top = report.get("top3_predictions") or []
    margin = top[0]["probability"] - top[1]["probability"] if len(top) > 1 else 0.0
    return (not report.get("ood_detection", {}).get("is_ood", True)
            and float(report.get("confidence", 0.0)) >= 0.70 and margin >= 0.15)


def process_batch(items: Iterable[Tuple[str, Dict[str, Any]]], analyzer,
                  on_progress=None) -> List[Dict[str, Any]]:
    """Analyze each (id, payload) independently; retain failures and continue."""
    item_list = list(items)
    results = []
    for index, (uid, item) in enumerate(item_list, start=1):
        try:
            results.append(analyzer(uid, item))
        except Exception as exc:
            results.append({"uid": uid, "filename": item.get("filename", ""),
                            "error": f"{type(exc).__name__}: {exc}", "reliable": False})
        if on_progress:
            on_progress(index, len(item_list))
    return results


def make_result_row(result: Dict[str, Any], language: str = "English") -> Dict[str, Any]:
    report = result.get("report") or {}
    reliable = result.get("reliable", False)
    severity = report.get("severity_assessment") or {}
    label = disease_info(report.get("predicted_class", ""), language)
    return {
        tr("filename", language): result.get("filename", ""),
        tr("table_class", language): label["label"] if label else report.get("predicted_class", ""),
        tr("score", language): report.get("confidence"),
        tr("table_ood", language): (tr("ood_suspected", language) if report.get("ood_detection", {}).get("is_ood") else tr("in_domain", language)) if report else tr("ood_unknown", language),
        tr("table_uncertain", language): tr("yes", language) if (not reliable or bool(result.get("error"))) else tr("no", language),
        tr("table_severity_pct", language): severity.get("lesion_percentage"),
        tr("table_severity_score", language): severity.get("irri_ses_score"),
        tr("advisory_status", language): tr("available_status", language) if reliable and label and label.get("url") else tr("unavailable", language),
        tr("table_source", language): label.get("url", "") if reliable and label and label.get("url") else "",
        tr("table_error", language): tr(result["error"], language) if result.get("error") in {"oversize", "unsupported", "invalid"} else (tr("error", language) if result.get("error") else ""),
    }


def results_csv(results: Iterable[Dict[str, Any]], language: str = "English",
                analysed_at: str | None = None) -> bytes:
    rows = [make_result_row(result, language) for result in results]
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=[tr("analysis_date", language), tr("report_disclaimer", language), *rows[0].keys()] if rows else [tr("analysis_date", language), tr("report_disclaimer", language)])
    writer.writeheader()
    date = analysed_at or datetime.now().astimezone().isoformat(timespec="seconds")
    disclaimer = tr("report_disclaimer", language)
    for row in rows:
        writer.writerow({tr("analysis_date", language): date, tr("report_disclaimer", language): disclaimer, **row})
    return out.getvalue().encode("utf-8-sig")


def farmer_report(result: Dict[str, Any], language: str = "English",
                  analysed_at: str | None = None) -> str:
    row = make_result_row(result, language)
    report = result.get("report") or {}
    if result.get("error"):
        finding = tr("error", language) + " " + str(result["error"])
    elif report:
        info = disease_info(report.get("predicted_class", ""), language)
        finding = f"{info['label'] if info else report.get('predicted_class')} ({float(report.get('confidence', 0)):.1%})"
    else:
        finding = tr("error", language)
    lines = [tr("title", language), f"{tr('filename', language)}: {result.get('filename', '')}"]
    if report:
        lines.extend([
            f"{tr('report_model_output', language)}: {finding}",
            f"{tr('report_ood', language)}: {tr('uncertain_status', language) if not result.get('reliable') else tr('available_status', language)}",
        ])
        if not result.get("reliable"):
            lines.append(tr("uncertain", language))
        severity = report.get("severity_assessment") or {}
        if severity:
            lines.append(f"{tr('lesion_pct', language)}: {severity.get('lesion_percentage', '—')}%")
            lines.append(f"{tr('ses', language)}: {severity.get('irri_ses_score', '—')} ({severity_band_text(severity.get('irri_ses_band', '—'), language)})")
            lines.append(tr("severity_caveat", language))
    else:
        lines.append(f"{tr('summary', language)}: {finding}")
    lines.extend([tr("report_disclaimer", language),
                  f"{tr('advisory', language)}: {row[tr('advisory_status', language)]}"])
    source_url = row[tr("table_source", language)]
    if source_url:
        lines.append(f"{tr('source_note', language)}: {source_url}")
    lines.append(f"{tr('analysis_date', language)}: {analysed_at or datetime.now().astimezone().isoformat(timespec='seconds')}")
    return "\n\n".join(lines)
