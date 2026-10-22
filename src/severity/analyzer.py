from pathlib import Path
from typing import Dict, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

def segment_leaf_and_lesions(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int, str]:
    """Segment leaf blade and disease lesions using Lab/HSV thresholding and morphology.
    Returns:
        leaf_mask: uint8 binary mask (255 for leaf, 0 for background)
        lesion_mask: uint8 binary mask (255 for lesion, 0 elsewhere)
        severity_pct: float percentage of leaf area damaged
        ses_score: IRRI SES 0-9 scale integer
        ses_band: Low, Moderate, or Severe
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    h, w, _ = image_bgr.shape

    # 1. Leaf Mask Extraction (separating leaf from white/light neutral background)
    # Background has high luminance and very low saturation
    val = hsv[:, :, 2]
    sat = hsv[:, :, 1]
    lum = lab[:, :, 0]

    # Leaf pixels have lower luminance than stark white background and noticeable color saturation
    leaf_mask = ((lum < 240) & (sat > 25)).astype(np.uint8) * 255

    # Refine with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep largest connected component as the leaf
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        leaf_mask = (labels == largest_label).astype(np.uint8) * 255

    leaf_area = int(np.sum(leaf_mask > 0))
    if leaf_area == 0:
        return leaf_mask, np.zeros_like(leaf_mask), 0.0, 0, "Low"

    # 2. Lesion Mask Extraction within the leaf blade
    # Lesions are necrotic, brownish/dark, or yellow halo
    # Compute relative darkness and discoloration within the leaf
    leaf_pixels_lum = lum[leaf_mask > 0]
    mean_lum = np.mean(leaf_pixels_lum)
    std_lum = np.std(leaf_pixels_lum)

    # Lesions in Lab: darker than healthy leaf (lum < mean - 0.5*std) OR brownish (high 'a' or lower 'b')
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    # Brown/necrosis signature: lower luminance OR elevated 'a' (reddish-brown hue)
    dark_lesions = (lum < (mean_lum - 0.4 * std_lum)) & (leaf_mask > 0)
    brown_lesions = (a_channel > 132) & (leaf_mask > 0)
    smut_dark = (val < 110) & (leaf_mask > 0)

    raw_lesion = (dark_lesions | brown_lesions | smut_dark).astype(np.uint8) * 255
    lesion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lesion_mask = cv2.morphologyEx(raw_lesion, cv2.MORPH_OPEN, lesion_kernel, iterations=1)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, lesion_kernel, iterations=1)
    lesion_mask = cv2.bitwise_and(lesion_mask, leaf_mask)

    lesion_area = int(np.sum(lesion_mask > 0))
    severity_pct = (lesion_area / leaf_area) * 100.0

    # 3. Map to IRRI Standard Evaluation System (SES) 0-9 bands
    if severity_pct == 0.0:
        ses_score = 0
        ses_band = "Low"
    elif severity_pct <= 1.0:
        ses_score = 1
        ses_band = "Low"
    elif severity_pct <= 5.0:
        ses_score = 3
        ses_band = "Low"
    elif severity_pct <= 25.0:
        ses_score = 5
        ses_band = "Moderate"
    elif severity_pct <= 50.0:
        ses_score = 7
        ses_band = "Severe"
    else:
        ses_score = 9
        ses_band = "Severe"

    return leaf_mask, lesion_mask, round(severity_pct, 2), ses_score, ses_band

def create_severity_overlay(
    image_bgr: np.ndarray,
    leaf_mask: np.ndarray,
    lesion_mask: np.ndarray,
    severity_pct: float,
    ses_score: int,
    ses_band: str,
    title: str = ""
) -> np.ndarray:
    """Return a clean image overlay; display severity text outside the image."""
    overlay = image_bgr.copy()

    # Draw leaf boundary (cyan)
    contours_leaf, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_leaf, -1, (255, 255, 0), 2)

    # Tint lesions in red with alpha blending
    red_tint = overlay.copy()
    red_tint[lesion_mask > 0] = [0, 0, 255] # Red in BGR
    cv2.addWeighted(red_tint, 0.45, overlay, 0.55, 0, overlay)

    # Highlight lesion contours in bright yellow
    contours_lesion, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_lesion, -1, (0, 255, 255), 1)

    return overlay

def run_phase6_severity_evaluation(
    manifest_path: Path = Path("data/processed/clean_manifest.csv"),
    output_dir: Path = Path("results/figures/severity_overlays"),
    num_samples: int = 20
):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest_path)
    print(f"[Severity Analyzer] Analyzing {num_samples} samples across classes...")

    # Pick balanced samples from each class
    samples_per_cls = num_samples // df["class_name"].nunique()
    sampled_dfs = [df[df["class_name"] == c].head(samples_per_cls + 1) for c in df["class_name"].unique()]
    sample_df = pd.concat(sampled_dfs).head(num_samples).reset_index(drop=True)

    results = []
    for idx, row in sample_df.iterrows():
        img_bgr = cv2.imread(row["rel_path"])
        leaf_mask, lesion_mask, pct, ses_score, ses_band = segment_leaf_and_lesions(img_bgr)
        overlay = create_severity_overlay(img_bgr, leaf_mask, lesion_mask, pct, ses_score, ses_band, title=row["class_name"])

        save_path = output_dir / f"sample_{idx+1:02d}_{row['class_name'].replace(' ', '_')}.png"
        cv2.imwrite(str(save_path), overlay)

        results.append({
            "sample_id": idx + 1,
            "filename": row["filename"],
            "class_name": row["class_name"],
            "leaf_pixels": int(np.sum(leaf_mask > 0)),
            "lesion_pixels": int(np.sum(lesion_mask > 0)),
            "severity_pct": pct,
            "irri_ses_score": ses_score,
            "irri_ses_band": ses_band,
            "validation_status": "heuristic, not validated against expert scoring",
            "overlay_path": str(save_path)
        })

    res_df = pd.DataFrame(results)
    table_path = Path("results/tables/severity_heuristic_samples.csv")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(table_path, index=False)
    print(f"[Severity Analyzer] Generated and saved {len(res_df)} overlays to {output_dir}")
    print(f"[Severity Analyzer] Saved severity metrics table to {table_path}")
    return res_df

if __name__ == "__main__":
    run_phase6_severity_evaluation()
