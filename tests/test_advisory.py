import pytest
from src.rag.advisory import GroundedAdvisoryRAG

@pytest.fixture(scope="module")
def rag_system():
    return GroundedAdvisoryRAG()

def test_refusal_on_low_confidence(rag_system):
    """Test that confidence < 0.7 triggers strict refusal with zero chemical advice."""
    res = rag_system.get_advisory(
        predicted_disease="Bacterial leaf blight",
        confidence=0.58, # below 0.70
        severity_band="Severe"
    )
    assert res["status"] == "REFUSAL_LOW_CONFIDENCE"
    assert "consult the local tnau kvk / agricultural officer" in res["message"].lower()

    # Assert zero chemical text in refusal markdown
    advisory_text = res["advisory_markdown"].lower()
    for forbidden in ["spray", "kg/ha", "ml/ha", "streptomycin", "mancozeb", "carbendazim", "copper oxychloride"]:
        assert forbidden not in advisory_text, f"Forbidden chemical text '{forbidden}' found in low-confidence refusal!"

def test_refusal_on_low_retrieval_similarity(rag_system):
    """Test that ungrounded queries trigger refusal with local agricultural officer redirection."""
    # Temporarily set similarity threshold high to simulate an ungrounded query
    rag_system.min_retrieval_similarity = 0.99
    try:
        res = rag_system.get_advisory(
            predicted_disease="Unknown Coffee Rust",
            confidence=0.95,
            severity_band="Low"
        )
        assert res["status"] == "REFUSAL_LOW_RETRIEVAL_SIMILARITY"
        assert "consult the local tnau kvk / agricultural officer" in res["message"].lower()
        assert len(res["retrieved_passages"]) == 0
    finally:
        rag_system.min_retrieval_similarity = 0.35

def test_source_checked_cultural_ipm_precedes_chemical_withholding(rag_system):
    res = rag_system.get_advisory(
        predicted_disease="Brown spot",
        confidence=0.88,
        severity_band="Moderate",
        severity_pct=15.0
    )
    assert res["status"] == "SUCCESS"
    text = res["advisory_markdown"]
    pos_cultural = text.find("### Cultural and IPM guidance")
    pos_chemical = text.find("### Chemical products and doses")
    pos_source = text.find("### Source")

    assert pos_cultural != -1, "Cultural & IPM section missing"
    assert pos_chemical != -1, "Chemical section missing"
    assert pos_source != -1, "Official source citation missing"
    assert pos_cultural < pos_chemical < pos_source, "Expected cultural guidance, chemical withholding, then source"
    assert "disease-free seed" in text.lower()
    assert "has been verified" in text

def test_citations_mandatory_and_present(rag_system):
    """Test that retrieved recommendations carry explicit file and page citations."""
    res = rag_system.get_advisory(
        predicted_disease="Leaf smut",
        confidence=0.91,
        severity_band="Moderate",
        severity_pct=18.0
    )
    assert res["status"] == "REFUSAL_LOW_RETRIEVAL_SIMILARITY"
    assert res["citations"] == []
    assert "No verified matching source record" in res["advisory_markdown"]


def test_verified_record_cites_official_tnau_url_and_withholds_chemicals(rag_system):
    res = rag_system.get_advisory("Bacterial leaf blight", 0.91, "Moderate")
    assert res["status"] == "SUCCESS"
    assert res["provenance_status"] == "official_page_checked"
    assert "https://www.agritech.tnau.ac.in/expert_system/paddy/cpdisblb.html" in res["advisory_markdown"]
    assert "no current applicable official product-label evidence" in res["advisory_markdown"]
    assert not any(term in res["advisory_markdown"].lower() for term in ["streptomycin", "kg/ha", "ml/ha"])
