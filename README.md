# Paddy Disease Diagnosis, Severity Grading & Cross-Dataset Generalization
### Grounded Treatment Advisory for Tamil Nadu Rice Farmers (TNAU & CIBRC Aligned)

A reproducible, CPU-only research pipeline evaluating transfer learning, calibration, explainability, heuristic disease severity grading, and grounded agronomic treatment retrieval on rice leaf disease imagery.

---

## 1. System & Hardware Specifications
- **Operating System**: Windows 11
- **Processor**: Intel Core 5 210H (12 threads allocated via `torch.set_num_threads(os.cpu_count())`)
- **Memory**: 16 GB RAM (15.6 GB usable)
- **Acceleration**: CPU-Only (`torch-2.14.1+cpu`, `torchvision-0.29.1+cpu`, NO CUDA)
- **Image Resolution**: 224 x 224
- **Deterministic Seed**: Fixed across Python, NumPy, PyTorch (`seed=42`)

---

## 2. Project Directory Structure

```
├── data/
│   ├── raw/rice_leaf_diseases/     # Local read-only Kaggle vbookshelf dataset
│   └── processed/                  # Manifests, splits.json, FAISS index, features
├── knowledge/                      # Verbatim TNAU Agritech & CIBRC statutory texts
├── results/
│   ├── checkpoints/                # Best model checkpoints per fold (.pt)
│   ├── figures/                    # Sample grids, distributions, curves, Grad-CAM, overlays
│   ├── tables/                     # CSV logs, comparison matrices, metric summaries
│   └── reports/                    # audit_report.md, locked_test_report.md, etc.
├── src/
│   ├── data/                       # audit.py, splits.py, dataset.py, generalization.py
│   ├── models/                     # baseline.py, model_factory.py, train.py, evaluate.py, profiler.py
│   ├── explainability/             # gradcam.py
│   ├── severity/                   # analyzer.py (HSV/Lab heuristic segmentation)
│   ├── rag/                        # advisory.py (FAISS vector retrieval)
│   └── utils/                      # seed.py
├── tests/                          # pytest test suite
├── predict.py                      # Production inference CLI with OOD detection
├── app.py                          # Streamlit farmer-facing screening interface
├── requirements.txt                # Pinned CPU dependencies
└── README.md
```

`walkthrough.md` documents the current status and safe operating limits of the
web application and extended evaluation steps.

## Quick start: farmer application (PowerShell)

Open PowerShell in the project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Choose **English** or **தமிழ்** in the persistent left sidebar, then use Home,
Diagnose, My Results, Treatment Advice, and About. Diagnose accepts one or more
JPG/JPEG/PNG photos, an entire folder (Streamlit directory upload), or a ZIP
containing nested folders. Remove unwanted images and choose **Analyse All / அனைத்தையும்
பரிசோதிக்கவும்**. Results are evaluated independently and retained in the current
browser session. Download a CSV history or per-image text report. Disease-specific
guidance is only shown after the uncertainty gate and cites checked TNAU pages.
The treatment page shows matched general cultural/IPM guidance and states when
severity-specific advice is unavailable. The CIBRC label folder
is empty, so pesticide products and doses remain suppressed. Tamil agricultural
wording remains provisional and needs expert review.

Run the test suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Rebuild the evidence inventory and advisory records after adding or removing
documents under `advisory/source_documents/`:

```powershell
.\.venv\Scripts\python.exe scripts\build_advisory_data.py
```

The builder records file hashes and content mismatches. PDF creation/download
timestamps are not treated as publication dates. It does not infer CIBRC label
approval from general advisory text.

The builder also refreshes `data/advisory/advisory_diagnostic_report.md` with
the source inventory, structured record coverage, legacy index status, advisory
lookup outcome by model class, and remaining source gaps.

---

## 3. Step-by-Step Reproduction Guide (PowerShell)

### Step 0: Setup Virtual Environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 1: Run Phase 1 Data Audit & Perceptual Hashing
```powershell
.\.venv\Scripts\python.exe -m src.data.audit
```

### Step 2: Generate Locked Test and 15-Fold Repeated CV Splits
```powershell
.\.venv\Scripts\python.exe -m src.data.splits
```

### Step 3: Train Frozen Feature Baselines & Deep CNNs
```powershell
# 1. Baseline: Frozen EfficientNet-B0 + Logistic Regression & Linear SVM
.\.venv\Scripts\python.exe -m src.models.baseline

# 2. 15-Fold CV (3 repeats x 5 folds) for MobileNetV3-Small, MobileNetV3-Large, EfficientNet-B0
.\.venv\Scripts\python.exe -m src.models.train
```

Optional matched background-crop ablation (15 CV folds for the selected model;
uses the same saved development folds and writes separate checkpoints/tables):

```powershell
.\.venv\Scripts\python.exe -m src.models.train --models mobilenetv3_small_100 --background-mode leaf_crop
```

This compares a largest-saturated-component crop preprocessing variant with the
existing standard run. It does not touch the locked test set. No ablation metrics
are claimed until this command completes and its fold results are reviewed.

### Step 4: Full Evaluation, Calibration, Efficiency Profiling & Locked Test Run
```powershell
.\.venv\Scripts\python.exe -m src.models.evaluate
```

### Step 5 & 6: Severity Grading & Explainability Overlays
```powershell
# Heuristic IRRI SES Severity Grading
.\.venv\Scripts\python.exe -m src.severity.analyzer

# Grad-CAM Attention Heatmaps & Lesion vs Background Density
.\.venv\Scripts\python.exe -m src.explainability.gradcam
```

### Step 7: Cross-Dataset Generalization
```powershell
.\.venv\Scripts\python.exe -m src.data.generalization
```

### Step 8: Verify RAG Advisory with Pytest
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### Step 9: Run Inference on New Images
```powershell
# Single image inference
.\.venv\Scripts\python.exe predict.py "data/raw/rice_leaf_diseases/Brown spot/DSC_0100.jpg"

# Batch folder inference
.\.venv\Scripts\python.exe predict.py --folder "data/ood_samples" --output_dir "results/predictions"
```

### Step 10: Launch the web application
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app uses the existing fold checkpoint selected by `predict.py`. Confidence,
margin, and OOD flags gate disease-specific advisory display. Grad-CAM and
severity outputs are visual aids; severity is heuristic and unvalidated.
Chemical products and doses are withheld until current, applicable official
product-label evidence is added and verified. The advisory's citations refer
to local source files and do not themselves establish label validity.

### Cross-dataset evaluation
Obtain an independently collected dataset, inspect its license and labels, and
organize image files under one class folder per label. Then call
`evaluate_cross_dataset` from `src.data.generalization` with that root and a
chosen existing checkpoint. This is zero-shot evaluation; do not tune on it.
Current candidate/compatibility status is in `results/reports/generalization_report.md`.

### Tests
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
