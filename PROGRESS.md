# PROGRESS — Paddy Disease Diagnosis Pipeline

**Project:** Paddy Disease Diagnosis, Severity Grading & Cross-Dataset
Generalization with a TNAU/CIBRC-Grounded Treatment Advisory for Tamil Nadu
Rice Farmers
**Root:** `C:\Users\gunal\Desktop\Paddy Disease Diagnosis`
**Environment:** Python 3.11, `.venv`, CPU-only, seed 42, 12 threads
**Last verified:** 2026-10-20 (Stage 0 orientation)

---

## Staged Execution Status

| Stage | Description | Status |
|-------|-------------|--------|
| **Stage 0** | Orient & Verify | DONE |
| **Stage 1** | Cross-Dataset Generalization | Pending user "continue" |
| **Stage 2** | Calibration Improvement | Pending |
| **Stage 3** | OOD Detection Hardening | Pending |
| **Stage 4** | Advisory Expansion | Pending |
| **Stage 5** | Final Report & Packaging | Pending |
| **Stage 6** | Stretch Goals | Pending |

---

## Phases Completed (Previous Session - Phases 0-9)

### Phase 0 - Setup
- Python 3.11 venv created
- All CPU-only packages installed (torch 2.14.1+cpu, timm 1.0.30, sklearn 1.9.1, etc.)
- Deterministic seed 42, 12 CPU threads
- requirements.txt written

### Phase 1 - Data Audit
- 120 images audited -- 0 corrupt, 0 SHA-256 duplicates, 0 pHash near-duplicates (<=4 threshold)
- Classes: Bacterial leaf blight (40), Brown spot (40), Leaf smut (40) -- perfectly balanced
- Manifest: data/processed/clean_manifest.csv (120 rows confirmed)
- Figures: dataset_sample_grid.png, image_dimension_distribution.png
- Report: results/reports/audit_report.md

### Phase 2 - Data Splits
- Locked test set: 24 images (8/class, stratified) -- evaluated ONCE, must NOT be reused for tuning
- Dev set: 96 images
- CV: 15 folds (5-fold x 3 repeats), repeated stratified
- Zero-leakage assertions passed
- data/processed/splits.json confirmed

### Phase 3 - Training

| Model | CV Macro-F1 | Train Time |
|-------|-------------|------------|
| Frozen EffNet-B0 + LogReg | 0.8545 +/- 0.0914 | -- |
| Frozen EffNet-B0 + SVM | 0.8550 +/- 0.0864 | -- |
| MobileNetV3-Small-100 | 0.8293 +/- 0.0729 | 25.4 min |
| MobileNetV3-Large-100 | 0.7900 +/- 0.0945 | 26.4 min |
| EfficientNet-B0 | 0.7482 +/- 0.0868 | 28.6 min |

- 45 checkpoints saved (15 per deep model x 3 models) -- confirmed

### Phase 4 - Evaluation & Model Selection
- Winner: mobilenetv3_small_100 -- overlapping CIs -> cheapest by pre-registered rule
  - 1.52M params, 9ms latency, 360.8 img/s
- Locked test (run ONCE, NOT to be reused):
  - Accuracy: 0.7083, Macro-F1: 0.7037, MCC: 0.5669, Kappa: 0.5625, ECE: 0.2492
- Tables: model_comparison.csv, locked_test_predictions.csv (24 rows confirmed)
- Figures: training_curves.png, confusion_matrices.png, calibration_curves.png, locked_test_confusion_matrix.png

### Phase 5 - Explainability (Grad-CAM)
- 10 test samples (3 correct, 7 incorrect)
- Mean lesion attention: 23.45%, background: 50.29%, density ratio: 1.65x
- Target layer: model.blocks[-1] (NOT conv_head -- yields all-zero CAM)
- 10 overlay PNGs in results/figures/gradcam/ -- confirmed
- Report: results/reports/explainability_report.md

### Phase 6 - Severity Grading
- HSV/Lab morphological segmentation on 20 samples
- IRRI SES bands mapped
- LABELED HEURISTIC, NOT VALIDATED AGAINST EXPERT SCORING
- 20 overlays in results/figures/severity_overlays/ -- confirmed

### Phase 7 - Cross-Dataset Generalization (Scaffolded/Skipped)
- No Dataset 2 available -> skipped
- Skip note logged in results/reports/generalization_report.md
- Code scaffold: src/data/generalization.py

### Phase 8 - RAG Advisory
- FAISS IndexFlatIP, 384-dim, 6 chunks from TNAU + CIBRC texts
- Model: all-MiniLM-L6-v2
- Advisory pipeline: Cultural/IPM -> Chemical -> Disclaimer
- Refusal below 70% confidence threshold
- data/processed/faiss_index.bin, data/processed/kb_chunks.json confirmed
- 4 tests passing

### Phase 9 - Predict CLI
- predict.py -- single-image and batch modes
- OOD detection: Max-Softmax (threshold 0.65) + Energy Score (threshold -3.0)
- Known limitation: Gaussian noise gets softmax = 1.0 (documented in final_report.md)

---

## Stage 0 Verification Results (2026-10-20)

### Test Suite
`
tests/test_advisory.py::test_refusal_on_low_confidence           PASSED
tests/test_advisory.py::test_refusal_on_low_retrieval_similarity PASSED
tests/test_advisory.py::test_cultural_ipm_comes_before_chemical  PASSED
tests/test_advisory.py::test_citations_mandatory_and_present     PASSED
tests/test_severity.py::test_severity_synthetic_blank_image      PASSED
tests/test_severity.py::test_severity_on_real_image              PASSED
tests/test_splits.py::test_splits_integrity                      PASSED
7 passed in 15.10s
`

### File Verification Against Walkthrough Claims

| Claim | Expected | Actual | Status |
|-------|----------|--------|--------|
| clean_manifest.csv rows | 120 | 120 | PASS |
| Per-class image count | 40 each | 40 each | PASS |
| splits.json test_count | 24 | 24 | PASS |
| splits.json dev_count | 96 | 96 | PASS |
| splits.json cv_folds | 15 | 15 | PASS |
| Checkpoint files | 45 | 45 | PASS |
| locked_test_predictions.csv rows | 24 | 24 | PASS |
| Grad-CAM figures | 10 | 10 | PASS |
| Severity overlays | 20 | 20 | PASS |
| Locked test Accuracy | 0.7083 | 0.7083 | PASS |
| Locked test Macro-F1 | 0.7037 | 0.7037 | PASS |
| Locked test MCC | 0.5669 | 0.5669 | PASS |
| Locked test ECE | 0.2492 | 0.2492 | PASS |
| MobileNetV3-Small CV F1 | 0.8293 +/- 0.0729 | 0.8293 +/- 0.0729 | PASS |

NO MISMATCHES FOUND. All 14 claims verified.

### Known Issues (Documented, Not Blocking)
1. OOD Gaussian noise failure: Max-softmax = 1.0 for pure noise. Energy score also fails (-3059).
   Documented in final_report.md. Not fixed.
2. SVC FutureWarning: sklearn 1.9.1 deprecated probability=True on SVC.
   Should migrate to CalibratedClassifierCV(SVC(), ensemble=False). Low priority.
3. Grad-CAM note: model.conv_head yields all-zero CAM. Correct target: model.blocks[-1]
   with requires_grad=True. Already fixed and working.
4. Locked test seen once: any further evaluation must be flagged "second look -- no longer pristine".

---

## Plan for Stages 1-6

### Stage 1 - Cross-Dataset Generalization
- No Dataset 2 available in Phase 7.
- Options: (a) download second dataset for zero-shot eval,
           (b) synthetic corruption robustness (CIFAR-C style),
           (c) document and skip with justification.
- Decision required from user.

### Stage 2 - Calibration Improvement
- ECE = 0.2492 is high. Options: Temperature Scaling, Platt Scaling.
- Must fit on dev/CV set only. Locked test must NOT be used for fitting.

### Stage 3 - OOD Detection Hardening
- Current failure: pure noise bypasses both thresholds.
- Options: ODIN, Mahalanobis distance, energy-based with dev-set calibrated threshold.

### Stage 4 - Advisory Expansion
- Currently 6 RAG chunks. Can expand TNAU/CIBRC knowledge base.
- Maintain retrieval-only (no generative LLM).

### Stage 5 - Final Report & Packaging
- Update final_report.md with all stage results.
- Package as reproducible ZIP. Add REPRODUCE.md.

### Stage 6 - Stretch Goals
- Severity validation against expert labels (if available).
- Grad-CAM on all 24 test images (currently 10).
- Per-disease advisory confidence histogram.

## Continuation update (2026-10-21)

- Added the bilingual five-view Streamlit interface, ZIP and multiple-image ingestion, session history, reports, and the evidence-gated Treatment Advice view. UI browser rendering could not be verified in this restricted environment.
- Added `data/advisory/source_manifest.csv` and `data/advisory/advisory_records.csv`. The three local PDFs were reviewed: the file named for BLB is a general index, the portal PDF concerns false smut, and the Brown spot PDF has relevant content. The expected CIBRC label directory is empty. Chemical advice remains disabled.
- Changed normal app/CLI advisory output to exact disease-matched checked records; the legacy FAISS index is only loaded on explicit request. Uncertain/OOD predictions and unsupported classes receive no disease-specific recommendation.
- Added opt-in `leaf_crop` preprocessing to the existing repeated-CV trainer. Command: `python -m src.models.train --models mobilenetv3_small_100 --background-mode leaf_crop`. It saves separate outputs and uses the same dev folds; the experiment has not been run, so the model comparison is still pending.
- Independent dataset files and expert severity labels remain unavailable. No external generalization or severity accuracy is claimed. The existing locked test metrics/checkpoints were not modified.
- Latest full test suite after the crop-ablation changes: 22 passed in 11.79s. Changed modules pass `py_compile`.

---

## Key Constraints (Never Violate)
- Never fabricate metrics -- all numbers must come from executed code.
- Locked test set evaluated ONCE. Any reuse must be flagged.
- Pre-registered decision rule: if bootstrap CIs overlap -> pick cheapest.
- Advisory: retrieval-only, no LLM, refusal below 70%, Cultural/IPM before Chemical.
- Severity grading labeled "heuristic, not validated against expert scoring" everywhere.
- All existing tests must pass after every stage change.
- Never overwrite without archiving to results/archive/<name>_v1 first.
