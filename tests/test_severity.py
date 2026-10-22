import numpy as np
import pytest
from src.severity.analyzer import segment_leaf_and_lesions, create_severity_overlay

def test_severity_synthetic_blank_image():
    """Test that a pure white background produces zero leaf area and zero severity."""
    white_img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    leaf_mask, lesion_mask, pct, ses_score, ses_band = segment_leaf_and_lesions(white_img)

    assert np.sum(leaf_mask) == 0
    assert np.sum(lesion_mask) == 0
    assert pct == 0.0
    assert ses_score == 0
    assert ses_band == "Low"

    # Text is intentionally placed in the UI, never painted into image pixels.
    overlay = create_severity_overlay(white_img, leaf_mask, lesion_mask, pct, ses_score, ses_band)
    assert np.array_equal(overlay, white_img)

def test_severity_on_real_image():
    """Test heuristic segmentation on a real dataset image."""
    import cv2
    from pathlib import Path
    import pandas as pd

    df = pd.read_csv("data/processed/clean_manifest.csv")
    sample_path = Path(df.iloc[0]["rel_path"])
    img = cv2.imread(str(sample_path))

    leaf_mask, lesion_mask, pct, ses_score, ses_band = segment_leaf_and_lesions(img)

    # Assertions
    assert leaf_mask.shape == img.shape[:2]
    assert lesion_mask.shape == img.shape[:2]
    assert np.sum(leaf_mask > 0) > 1000, "Leaf mask should detect leaf area"
    assert 0.0 <= pct <= 100.0, "Severity percentage must be in [0, 100]"
    assert ses_score in [0, 1, 3, 5, 7, 9], f"Invalid IRRI SES score: {ses_score}"
    assert ses_band in ["Low", "Moderate", "Severe"], f"Invalid IRRI SES band: {ses_band}"

    # Test overlay creation
    overlay = create_severity_overlay(img, leaf_mask, lesion_mask, pct, ses_score, ses_band)
    assert overlay.shape == img.shape
