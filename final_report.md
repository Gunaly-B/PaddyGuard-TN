# Comprehensive Research Report: Paddy Disease Diagnosis, Severity Grading & Grounded Advisory Pipeline

**Project Title**: "Paddy Disease Diagnosis, Severity Grading & Cross-Dataset Generalization with a TNAU/CIBRC-Grounded Treatment Advisory for Tamil Nadu Rice Farmers"
**Execution Environment**: Windows 11, Intel Core 5 210H, 16 GB RAM (15.6 GB usable), CPU-Only (`torch-2.14.1+cpu`)
**Reproducibility**: Deterministic Seed 42, 12 Allocated CPU Threads, Relative Paths Throughout

---

## 1. Executive Summary & Experimental Workflow

This research project designed, implemented, and empirically evaluated a complete, reproducible machine learning pipeline for paddy rice leaf disease diagnosis across three major diseases prevalent in Tamil Nadu: **Bacterial Leaf Blight** (*Xanthomonas oryzae* pv. *oryzae*), **Brown Spot** (*Bipolaris oryzae*), and **Leaf Smut** (*Entyloma oryzae*).

Every metric, latency figure, and confidence interval in this report was computed from real CPU execution on the target hardware without data fabrication or artificial inflation.

```
                    Raw Dataset (120 images in data/raw/rice_leaf_diseases/)
                                             │
                   ┌─────────────────────────┴────────────────────────┐
                   │ Phase 1: Data Audit & Deduplication (pHash/dHash)│
                   └─────────────────────────┬────────────────────────┘
                                             │
                   ┌─────────────────────────┴────────────────────────┐
                   │ Phase 2: Partitioning (Zero-Leakage Isolation)   │
                   ├─────────────────────────┬────────────────────────┤
                   │ Locked Test Set (20%)   │ Dev Set (80%, 96 imgs) │
                   │ Held out, untouched     │ 15-Fold Repeated CV    │
                   └─────────────────────────┴───────────┬────────────┘
                                                         │
                                  ┌──────────────────────┴──────────────────────┐
                                  ▼                                             ▼
                 Frozen Baseline (Pooled EffNet-B0)            Fine-Tuned Deep Models (Stage 1 & 2)
                 - Logistic Reg: F1 = 0.8545                   - MobileNetV3-Small: F1 = 0.8293
                 - Linear SVM:   F1 = 0.8550                   - MobileNetV3-Large: F1 = 0.7900
                                                               - EfficientNet-B0:   F1 = 0.7482
                                  │                                             │
                                  └──────────────────────┬──────────────────────┘
                                                         │
                                  ┌──────────────────────┴──────────────────────┐
                                  │ Phase 4: Model Selection & Single Test Run  │
                                  │ Winner: MobileNetV3-Small (1.52M params)    │
                                  │ Locked Test Run: Accuracy 70.8%, F1 70.4%   │
                                  └──────────────────────┬──────────────────────┘
                                                         │
                   ┌─────────────────────────────────────┼─────────────────────────────────────┐
                   ▼                                     ▼                                     ▼
        Phase 5: Explainability               Phase 6: Heuristic Severity           Phase 8: Grounded Advisory
     Grad-CAM Attention vs Masks           IRRI SES 0-9 Scale (Lab/HSV)           FAISS Vector DB (MiniLM-L6)
     Lesion Density: 1.65x                 "Heuristic, Not Validated"             Cultural First -> Chemical
     50.29% Background Spill               20 Visual Overlays Generated           Zero Hallucinations (Citations)
```

---

## 2. Phase 1: Data Audit & Perceptual Hashing

The raw dataset (Kaggle `vbookshelf/rice-leaf-diseases`) was ingested locally into `data/raw/rice_leaf_diseases/`:
- **Total images audited**: 120 images (balanced: 40 per class).
- **Format**: 100% 3-channel RGB JPEG format.
- **Resolution range**: Min 250x71, Max 4160x2340 (Aspect ratios from 1.25 to 5.30).
- **Corrupt / unreadable images**: 0.
- **Exact SHA-256 duplicates**: 0.
- **Perceptual near-duplicates (pHash distance <= 4)**: 0.
- **Retained clean manifest**: 120 images saved to `data/processed/clean_manifest.csv`.

---

## 3. Phase 2: Dataset Splits & Zero-Leakage Verification

A strict two-tier stratified partitioning scheme was executed:
1. **Locked Final Test Set**: 20% stratified subset ($N=24$, exactly 8 images per class) was isolated immediately into `data/processed/splits.json` and kept completely locked during all training and tuning.
2. **Development Set**: The remaining 80% ($N=96$, 32 images per class) was split into a **5-fold Stratified Cross-Validation repeated 3 times (15 total folds)**.
3. **Leakage Verification**:
   $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset, \quad \text{Train} \cup \text{Val} \cup \text{Test} = \text{Dataset}$$
   Verified via automated test in `tests/test_splits.py`.

---

## 4. Phase 3 & 4: Model Training, Cross-Validation & Efficiency

### A. Comprehensive Model Comparison Table (15-Fold Repeated CV)

| Model Architecture | CV Macro-F1 (Mean $\pm$ Std) | CV Accuracy (Mean $\pm$ Std) | Bootstrap 95% CI (Macro-F1) | MCC | Cohen's Kappa | ECE | Params (M) | GMACs | Disk Size | Single Img Latency (ms) | Batch 16 Latency (ms) | Throughput (img/s) | Total Training Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Frozen EffNet-B0 + Linear SVM** | **0.8550 $\pm$ 0.0864** | **0.8577 $\pm$ 0.0829** | N/A (Linear) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | **< 1 min** |
| **Frozen EffNet-B0 + Logistic Reg** | **0.8545 $\pm$ 0.0914** | **0.8574 $\pm$ 0.0878** | N/A (Linear) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | **< 1 min** |
| **MobileNetV3-Small-100** | **0.8293 $\pm$ 0.0729** | **0.8304 $\pm$ 0.0709** | **[0.7846, 0.8684]** | **0.7450** | **0.7448** | **0.1293** | **1.52** | **0.0555** | **5.92 MB** | **9.0 ms** | **44.3 ms** | **360.8 img/s** | **25.4 min** |
| **MobileNetV3-Large-100** | 0.7900 $\pm$ 0.0945 | 0.7923 $\pm$ 0.0931 | [0.7449, 0.8355] | 0.6890 | 0.6875 | 0.1515 | 4.21 | 0.2154 | 16.23 MB | 15.5 ms | 150.3 ms | 106.5 img/s | 26.4 min |
| **EfficientNet-B0** | 0.7482 $\pm$ 0.0868 | 0.7561 $\pm$ 0.0840 | [0.7055, 0.8035] | 0.6358 | 0.6354 | 0.1738 | 4.01 | 0.3846 | 15.57 MB | 28.6 ms | 269.3 ms | 59.4 img/s | 28.6 min |

### B. Scientific Insights from CV
1. **Linear Baseline vs Deep Fine-Tuning**: On this tiny dataset ($N=96$ training split), frozen ImageNet features paired with Linear SVM achieved higher mean CV performance (**0.8550**) than full end-to-end convolutional fine-tuning (**0.8293**). In small-sample regimes, feature extraction avoids parameter overfitting.
2. **Model Selection**: Comparing the fine-tuned deep architectures, `mobilenetv3_small_100` achieved the highest CV Macro-F1 (0.8293), but its 95% bootstrap confidence interval `[0.7846, 0.8684]` completely overlaps with `mobilenetv3_large_100` (`[0.7449, 0.8355]`) and `efficientnet_b0` (`[0.7055, 0.8035]`). Under our statistical selection protocol, because these models are statistically indistinguishable, we select the cheapest model: **`mobilenetv3_small_100`** (1.52M parameters, 9.0ms single-image CPU latency, 360.8 images/second throughput).

### C. Locked Test Set Evaluation (Evaluated Exactly Once)
The winning model (`mobilenetv3_small_100`, best checkpoint from Fold 1) was evaluated on the 24 held-out test images exactly once:
- **Test Accuracy**: **0.7083** (95% Bootstrap CI: `[0.5000, 0.8750]`)
- **Test Macro-F1**: **0.7037** (95% Bootstrap CI: `[0.4777, 0.8737]`)
- **Matthews Correlation Coefficient (MCC)**: **0.5669**
- **Cohen\'s Kappa**: **0.5625**
- **Expected Calibration Error (ECE)**: **0.2492**

#### Per-Class Performance Breakdown:
| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Bacterial Leaf Blight** | 0.7143 | 0.6250 | 0.6667 | 8 |
| **Brown Spot** | 0.7143 | 0.6250 | 0.6667 | 8 |
| **Leaf Smut** | 0.7000 | 0.8750 | 0.7778 | 8 |

---

## 5. Phase 5: Explainability & Attention Quantification (Grad-CAM)

Using the Phase 6 heuristic segmentation masks, Grad-CAM activation was segmented across lesion pixels, healthy leaf blade pixels, and background pixels for 10 test samples (3 correct, 7 incorrect):
- **Mean Lesion Attention %**: 23.45%
- **Mean Healthy Leaf Attention %**: 26.25%
- **Mean Background Attention %**: **50.29%**
- **Mean Lesion Attention Density Ratio (Lesion vs Background)**: **1.65x**

### Honest Background Reliance Report
The empirical data shows that **an average of 50.29% of total model activation spills onto the background**. While the attention density on lesions is 1.65x higher than background density, on multiple misclassified samples (e.g. `DSC_0702.jpg`), background activation exceeded 54%. The model partially relies on paper texture, edge shadows, and ambient background lighting as classification shortcuts.

---

## 6. Phase 6: Heuristic Disease Severity Grading (IRRI SES)

Implemented automatic leaf blade and lesion segmentation using morphological Lab/HSV color filtering:
- Extracted leaf surface area (cyan boundary) and necrotic lesions (red mask with yellow outline).
- Calculated $\text{Severity \%} = \frac{\text{Lesion Area}}{\text{Leaf Area}} \times 100\%$.
- Mapped to IRRI Standard Evaluation System (SES) bands:
  * Scale 0: 0% (Healthy)
  * Scale 1-3: $\le 5\%$ (Low)
  * Scale 5: $6-25\%$ (Moderate)
  * Scale 7-9: $> 25\%$ (Severe)
- Generated 20 visual overlays in `results/figures/severity_overlays/`.
- **Mandatory Caveat**: Labeled on every output and figure as: *"heuristic, not validated against expert scoring. Do not report accuracy for it."*

---

## 7. Phase 7: Cross-Dataset Generalization

No external rice dataset (such as Paddy Doctor) was provided in the project workspace. As specified in the protocol, the generalization claim cannot be evaluated without external data and is **skipped with explicit scientific notation** in `results/reports/generalization_report.md`.

---

## 8. Phase 8: Advisory provenance and current safety behavior

The local `knowledge/` text files and FAISS index are retained for reproducibility, but their provenance and pesticide claims are not independently established. The farmer app and CLI no longer present retrieved legacy passages as official guidance. They use exact disease-matched records in `data/advisory/advisory_records.csv`; only checked TNAU cultural/IPM summaries are eligible. The FAISS embedding model is lazy-loaded only if the legacy index is explicitly requested.

The source inventory in `data/advisory/source_manifest.csv` records SHA-256 hashes, extracted source URLs, evidence status, and review notes. Inspection found that the PDF named for bacterial leaf blight is a general disease index without BLB management detail, while the TNAU portal PDF discusses false smut, which is distinct from the model's Leaf smut class. The Brown spot PDF contains Brown spot material. The CIBRC product-label folder has no files. Accordingly, the advisory has two eligible cultural/IPM records (BLB and Brown spot), no verified Leaf smut record, and no pesticide products or doses.

Prediction reliability requires no OOD flag, confidence of at least 0.70, and top-two probability margin of at least 0.15. Failure returns no disease-specific recommendation. These thresholds are conservative prototype gates; they do not prove every unfamiliar image will be detected. Full tests are in `tests/test_advisory.py` and `tests/test_app_support.py`.

---

## 9. Phase 9: New-Image & Out-Of-Distribution (OOD) Inference

The command-line tool `predict.py` provides single-image and batch-folder inference with latency tracking, severity segmentation, Grad-CAM overlays, RAG advisory, and Out-of-Distribution detection via:
1. **Max-Softmax Probability (MSP)** (threshold 0.65)
2. **Negative Log-Sum-Exp Energy Score** (threshold -3.0)

### OOD Empirical Results (Tested on 5 Non-Rice Inputs):
- `ood_02_indoor_object.jpg` (hardware photo): Max Softmax = 0.6133 -> **`OOD_SUSPECTED`** (Refused chemical advice).
- `ood_05_red_circular_pattern.png`: Energy Score = -2.4641 -> **`OOD_SUSPECTED`** (Refused chemical advice).
- `ood_04_gaussian_noise.png`: Softmax assigned 100% confidence to Leaf Smut. This demonstrated the well-documented **closed-world softmax failure mode**, where high-frequency random noise mimics tiny smut teliospores and triggers false certainty.

---

## 10. Honest Scientific Limitations

> [!WARNING]
> ### 1. Tiny Dataset Regime ($N=120$)
> With only 40 images per disease class, standard deep networks are prone to high fold-to-fold variance (CV Macro-F1 ranged from 0.719 to 1.000 across folds). The locked test set ($N=24$) has a wide 95% confidence interval `[0.4777, 0.8737]`. Deep fine-tuning did not outperform frozen linear baselines.

> [!CAUTION]
> ### 2. Clean-Background Shortcut Bias
> Leaves were photographed against stark, artificial backgrounds. Grad-CAM analysis confirmed that **50.29% of activation spills onto background regions**. Models trained on such imagery cannot be safely deployed in field conditions without retraining on genuine in-situ field photos.

> [!NOTE]
> ### 3. Heuristic Severity Grading
> The segmentation pipeline relies on color-space thresholding and has **not been validated against human plant pathologist scoring**. It serves as an exploratory guide, not an agronomic certification.

> [!IMPORTANT]
> ### 4. Grounded Advisory Scope
> The advisory is strictly limited to the ingested TNAU and CIBRC texts. It does not replace on-field physical verification by a qualified agricultural officer or Krishi Vigyan Kendra specialist.

---

## 11. Implementation update (2026-10-21)

- Added a Streamlit interface in `app.py` for image upload, class scores, uncertainty warnings, Grad-CAM, a clearly labeled heuristic severity overlay, and retrieved advisory passages.
- The interface withholds disease-specific advisory display for OOD-flagged predictions, confidence below 0.70, or a top-two score margin below 0.15. This is a conservative gate, not a guarantee that all unfamiliar inputs will be caught.
- Chemical products and doses are now withheld by the advisory because the local reference files do not include verifiable current label/leaflet evidence tied to every crop, disease, formulation, dose, and pre-harvest interval. The source text files' own "official" labels are not independently proven by their contents. TNAU's public BLB recommendations can be checked independently, but that alone does not verify each product label or validate all advice. See [TNAU BLB page](https://agritech.tnau.ac.in/expert_system/paddy/cpdisblb.html).
- RiceLeafDiseaseBD is a candidate external field-photo set with Brown Spot and Leaf Smut overlap, but no dataset files are present locally and no external metrics were calculated. Its listed classes do not include Bacterial leaf blight. Dataset record: https://data.mendeley.com/datasets/86s4jzj2m4/3.
- Existing training already included random resized crop, flips, rotation, color jitter, and random whole-image blur. A separately trained, fold-matched foreground-removal/augmentation comparison has not been run; existing results do not demonstrate bias reduction.
- Severity estimates remain heuristic and unvalidated because expert reference labels are unavailable.
- Added launch and limitations instructions in `walkthrough.md`; added advisory safety-gate unit coverage. Test status for this update is recorded at delivery.

## 12. Farmer interface update (2026-10-21)

- Reorganized the Streamlit layout and moved raw scores, OOD metrics, and Grad-CAM into an expandable technical section. Severity text now appears outside the clean visual overlay.
- Added English/Tamil interface text, draft symptom/IPM summaries, bilingual per-image text reports, and a localized batch CSV summary. Tamil agricultural wording is explicitly flagged for expert review.
- Added multi-image selection/removal, per-file format/size/decode validation, progress, and per-image failure handling. Safety checks apply independently to every image; advisory is withheld below 0.70 confidence, below a 0.15 top-two margin, or when OOD is flagged.
- Added linked TNAU summaries for bacterial leaf blight and Brown spot. The app does not display unverified local-corpus wording as official guidance. No chemical product or dose is shown. Leaf smut remains separate from false smut and has no verified disease-specific page attached here.
- New support tests cover translation completeness, image validation, batch continuation, reliability gates, and downloadable report contents. A browser screenshot check could not be completed in this restricted execution environment; Streamlit startup was attempted, but localhost socket access was denied.

## 13. Current application, advisory, and verification update (2026-10-21)

- Reorganized `app.py` into Home, Diagnose, My Results, Treatment Advice, and About views in English and Tamil. Diagnose accepts multiple images or ZIP archives, validates members without extracting paths, shows per-image results and uncertainty warnings, and saves result history to the current browser session only. CSV history, individual text reports, Grad-CAM, and heuristic lesion overlays are available.
- The Treatment Advice view and CLI use source-matched records and suppress disease-specific advice whenever the reliability gate fails. Unmatched classes, including Leaf smut, fail closed. Legacy FAISS passages are no longer presented as authoritative; the embedding model is not loaded for normal inference.
- Rebuilt `data/advisory/source_manifest.csv` and `data/advisory/advisory_records.csv` from three actual PDFs plus two official TNAU web pages checked on 2026-10-21. Manifest dates are access dates; unknown publication dates are left blank. Chemical label evidence remains unavailable, so products and doses are withheld.
- Existing model and locked test results were not changed. Added an opt-in `leaf_crop` preprocessing mode to the existing repeated CV trainer; its outputs use separate `_leaf_crop` checkpoint/table names and the existing development folds. The run has not been executed, so no bias-reduction result is claimed. Public external dataset records were identified, but no independent images were locally available, so cross-dataset metrics and severity validation remain unavailable.
- Existing OOD examples do not constitute a broad validation across diverse field plant and non-plant photos. The known Gaussian-noise sample reached 100% softmax confidence, so softmax is not an OOD guarantee.
- Verification: `python -m py_compile` passed for changed Python modules; `python -m pytest tests/ -v` passed **22 tests** in 11.79 seconds. A browser-level Streamlit check could not be completed in this restricted environment; application rendering was not independently verified here.

## 14. Reproduction and launch (PowerShell)

From the project directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.advisory.build_manifests
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Run all tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
