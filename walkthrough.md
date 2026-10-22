# PaddyGuard-TN farmer application walkthrough

## Launch on Windows

```powershell
cd "C:\Users\gunal\Desktop\Paddy Disease Diagnosis"
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Choose English or Tamil in the persistent left sidebar and navigate between Home, Diagnose, My Results, Treatment Advice, and About. Diagnose accepts one/multiple JPG/JPEG/PNG files, a folder, or a ZIP containing nested folders (10 MB each; ZIP limits apply as above). ZIP members are read in memory; paths are not extracted. Each image is processed independently, and a corrupt image does not stop the remaining batch. Folder upload requires the supported Streamlit version in `requirements.txt`.

## Read results

- The possible disease and next step are prominent. Scores are model outputs, not confirmed diagnoses.
- Low confidence (<70%), a top-two score gap below 15 percentage points, or an OOD flag withholds disease-specific advice.
- Severity percentage and SES estimate are displayed separately from the clean mask overlay. Severity remains a heuristic and has not been validated against expert scores.
- Official TNAU cultural/IPM summaries are shown for bacterial leaf blight and Brown spot only when the prediction passes the reliability gate. The “BLB” PDF currently in `advisory/source_documents/` is a generic disease index; the false-smut PDF is not leaf-smut evidence. Neither is used to support matching treatment guidance.
- Pesticide products and doses are withheld. The CIBRC label/leaflet directory currently contains no files, and no current crop/disease/formulation label verification has been completed.
- Tamil agriculture terms are draft translations and need review by a Tamil-speaking crop-protection expert.
- Technical model scores, OOD details, and Grad-CAM appear in the expandable **Technical Details / தொழில்நுட்ப விவரங்கள்** section.

Each image has an individual farmer-friendly report download. The session history can be downloaded as CSV and cleared after a confirmation step. Reports include dates, uncertainty, heuristic severity caveats, and official links only where available. History remains in the current app session and is not stored in a database.

The source inventory and advisory records are generated from the local documents and checked official web pages:

```powershell
.\.venv\Scripts\python.exe scripts\build_advisory_data.py
```

This writes `data/advisory/source_manifest.csv`, `data/advisory/advisory_records.csv`, and `data/advisory/advisory_diagnostic_report.md`. Review extraction status and notes before adding records to the app. Chemical entries fail closed unless exact current official label evidence is recorded.

The same command updates `data/advisory/advisory_diagnostic_report.md`, including the actual local PDF findings, legacy index status, class coverage, and source access gaps.

An optional background-crop ablation uses the existing development CV folds and writes separate outputs without touching the locked test set:

```powershell
.\.venv\Scripts\python.exe -m src.models.train --models mobilenetv3_small_100 --background-mode leaf_crop
```

This experiment has not yet been run; do not interpret the preprocessing hook as evidence of improved generalization.

## Tests and existing CLI

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe predict.py "data/raw/rice_leaf_diseases/Brown spot/DSC_0100.jpg"
.\.venv\Scripts\python.exe predict.py --folder "data/ood_samples" --output_dir "results/predictions"
```

This remains a research screening prototype trained with only 120 controlled-background images. It has not been validated for field deployment. Independent-dataset evaluation, a matched background-removal experiment, and expert severity validation remain incomplete.
