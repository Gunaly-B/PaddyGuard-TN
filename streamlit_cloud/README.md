# Deploy PaddyGuard-TN on Streamlit Community Cloud

This folder is the Community Cloud entrypoint. It launches the project-root `app.py`; its local `requirements.txt` lists only dependencies needed by the app. The root project remains available for research commands and tests.

## Files that must be in the GitHub repository

The root `.gitignore` keeps local datasets, most experiment outputs, and other fold checkpoints out of the repository. The deployment still requires these files:

- `streamlit_cloud/app.py`
- `streamlit_cloud/requirements.txt`
- root `app.py`, `predict.py`, and `src/`
- `data/processed/splits.json`
- `data/advisory/advisory_records.csv`
- `results/checkpoints/mobilenetv3_small_100/repeat_0_fold_1_best.pt`

The selected checkpoint is the one named in the existing locked-test report. No secrets are required by the app. Do not publish `.venv`, `data/raw`, advisory source-document PDFs, `.streamlit/secrets.toml`, or other local/private files.

## Publish and deploy

1. Review the files staged for the public GitHub repository. Confirm you have the right to publish the source code, selected checkpoint, and advisory records.
2. Push the project to a GitHub repository that your Streamlit account can access. Community Cloud requires a GitHub repository and repository access.
3. Sign in at [share.streamlit.io](https://share.streamlit.io/) and choose **Create app**.
4. Select the repository and branch. Set **Main file path** to `streamlit_cloud/app.py`.
5. In **Advanced settings**, select Python 3.11, then choose **Deploy**.
6. Wait for the build to complete and open the app URL. Test one supported image and confirm that prediction, visualizations, and advisory gating load correctly.

The app is intended as a research screening prototype. Its current model was trained on 120 controlled-background images; external field performance, severity grading, and robust OOD rejection are not established.
