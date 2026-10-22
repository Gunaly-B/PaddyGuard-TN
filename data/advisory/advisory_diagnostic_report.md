# PaddyGuard-TN advisory diagnostic report

Generated: 2026-10-21

## Source folder check

- Requested location exists: `False` â€” `C:\Users\gunal\Desktop\Paddy Disease Diagnosis\data\advisory\source_documents`.
- Legacy project source folder exists: `True` â€” `C:\Users\gunal\Desktop\Paddy Disease Diagnosis\advisory\source_documents`.
- CIBRC labels subfolder exists: `True`; files found: 0.
- Local source files discovered recursively: 3; extracted successfully: 3; empty: 0; failed/unsupported: 0.

## Per-file extraction results

- `advisory/source_documents/TNAU Agritech Portal __ Crop Protection.pdf` (pdf): `extracted`, 2507 extracted characters; title `TNAU Agritech Portal :: Crop Protection`, URL `https://agritech.tnau.ac.in/crop_protection/rice_diseases/rice_7.html`, publisher `Tamil Nadu Agricultural University`.
- `advisory/source_documents/tnau_bacterial_leaf_blight.pdf` (pdf): `extracted`, 1011 extracted characters; title `Crop Protection`, URL `https://agritech.tnau.ac.in/expert_system/paddy/cpdiseases.html`, publisher `Tamil Nadu Agricultural University`.
- `advisory/source_documents/tnau_brown_spot.pdf` (pdf): `extracted`, 5995 extracted characters; title `Crop Protection`, URL `https://agritech.tnau.ac.in/expert_system/paddy/cpdisbrownspot.html`, publisher `Tamil Nadu Agricultural University`.

## CSV and advisory records

- Source manifest rows: 5 (includes local files and previously reviewed official web entries).
- Advisory records: 2; by disease: {'Bacterial leaf blight': 1, 'Brown spot': 1}; by category: {'general_cultural_ipm': 2}.
- Records with source IDs: 2. Severity-specific records: 0. Chemical label verified records: 0.
- All records have matching source IDs; their exact official source URLs and sections are included. No chemical or severity-specific records were created.

## FAISS index status

- FAISS file exists: True; metadata exists: True.
- FAISS and metadata exist, but metadata has 6 legacy chunks from 2 sources (CIBRC_Registered_Paddy_Pesticides.txt, TNAU_Agritech_Portal_Paddy_Diseases.txt). These legacy materials are not verified records, so the index was not rebuilt or used for farmer-facing advice. The app resolves checked CSV records directly.

## Coverage limitations

- Brown spot and Bacterial leaf blight have reviewed general cultural/IPM records from the official TNAU URLs listed in the CSV.
- Leaf smut has no disease-matched record. The local false-smut document is about false smut, not model class Leaf smut.
- No current CIBRC/PPQS product label was available locally; chemical products and doses remain withheld.
- The requested `data/advisory/source_documents/` folder is absent. Local PDFs were found in `advisory/source_documents/` and were processed there. Move/copy future source files into the requested folder or leave them in the legacy folder; both are scanned.
- Empty local PDF extractions are flagged for OCR/manual review. Text-extractable documents do not by themselves establish current label verification.
