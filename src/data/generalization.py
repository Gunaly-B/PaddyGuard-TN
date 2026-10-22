import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.data.dataset import PaddyDataset, get_transforms
from src.models.evaluate import evaluate_predictions
from src.models.model_factory import create_model

def normalize_label(name: str) -> str:
    """Normalize directory name to match standard disease names."""
    clean = name.lower().replace("_", " ").replace("-", " ").strip()
    compact = " ".join(clean.split())
    if compact in {"bacterial leaf blight", "bacterial blight", "leaf blight", "blb"}:
        return "Bacterial leaf blight"
    elif compact in {"brown spot", "brownspot"}:
        return "Brown spot"
    elif compact in {"leaf smut", "smut"}:
        return "Leaf smut"
    return name

def evaluate_cross_dataset(
    dataset2_path: Optional[str],
    winner_name: str,
    checkpoint_path: Path,
    splits_path: Path = Path("data/processed/splits.json"),
    results_dir: Path = Path("results")
):
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "reports" / "generalization_report.md"

    if dataset2_path is None or not Path(dataset2_path).exists():
        note = (
            "# Phase 7: Cross-Dataset Generalization Report\n\n"
            "> [!NOTE]\n"
            "> **Dataset 2 Not Provided / Not Available.**\n"
            "> The cross-dataset generalization claim cannot be evaluated without external field imagery (e.g., Paddy Doctor).\n"
            "> **Status**: Skipped with clear note as required by research protocol.\n\n"
            "### Implications for Real-World Deployment:\n"
            "- Laboratory/studio datasets with clean backgrounds do not guarantee field generalization.\n"
            "- Future work must validate the model on in-field farmer phone photographs with complex agro-ecological backgrounds.\n"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(note)
        print(f"[Phase 7 Generalization] Saved skip note to {report_path}")
        return None

    ds2_dir = Path(dataset2_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    domain1_classes = splits["metadata"]["class_names"]

    # Scan dataset 2 subdirectories
    subdirs = [d for d in ds2_dir.iterdir() if d.is_dir()]
    mapping = {}
    for d in subdirs:
        norm = normalize_label(d.name)
        if norm in domain1_classes:
            mapping[d.name] = norm

    print(f"[Phase 7] Found {len(mapping)} overlapping classes in {ds2_dir.name}: {mapping}")
    if not mapping:
        print("[Phase 7 Warning] No overlapping disease classes found between Dataset 1 and Dataset 2.")
        return None

    # Collect images
    records = []
    for folder_name, norm_label in mapping.items():
        folder_p = ds2_dir / folder_name
        for f in folder_p.glob("*.*"):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                records.append({
                    "rel_path": str(f),
                    "filename": f.name,
                    "raw_class": folder_name,
                    "class_name": norm_label
                })

    df2 = pd.DataFrame(records)
    print(f"[Phase 7] Dataset 2 contains {len(df2)} images across {len(mapping)} overlapping classes.")

    # Load Winner Model
    model = create_model(winner_name, num_classes=len(domain1_classes), pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # Zero-shot evaluation
    if df2.empty:
        raise ValueError(f"No readable image paths found under compatible class folders in {ds2_dir}")

    loader2 = DataLoader(
        PaddyDataset(df2, transform=get_transforms(224, is_train=False),
                     class_to_idx={name: idx for idx, name in enumerate(domain1_classes)}),
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels, _ in loader2:
            outs = model(imgs)
            probs = torch.softmax(outs, dim=1).numpy()
            all_preds.extend(outs.argmax(dim=1).numpy())
            all_targets.extend(labels.numpy())
            all_probs.append(probs)

    all_targets = np.array(all_targets)
    all_probs = np.concatenate(all_probs, axis=0)
    zero_shot_metrics = evaluate_predictions(all_targets, all_probs, domain1_classes)

    # Report external metrics only. The historical locked-test metric is not
    # recomputed or used as a tuning signal here.
    zero_shot_f1 = zero_shot_metrics["macro_f1"]
    print(f"[Phase 7] Zero-Shot Cross-Dataset Macro-F1: {zero_shot_f1:.4f}")

    reports_dir = results_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "generalization_report.md").write_text(
        f"# Cross-dataset zero-shot evaluation\n\n"
        f"Dataset: `{ds2_dir}`. No training or tuning was performed on this dataset.\n\n"
        f"Compatible class folders: {', '.join(mapping)}. Samples: {len(df2)}.\n\n"
        f"Macro-F1: {zero_shot_metrics['macro_f1']:.4f}.\n\n"
        f"This evaluation uses the existing checkpoint `{checkpoint_path}`. Interpret results by compatible-class coverage;\n"
        f"unmatched classes are excluded and are not evidence of generalization to those diseases.\n",
        encoding="utf-8"
    )
    return zero_shot_metrics

if __name__ == "__main__":
    evaluate_cross_dataset(None, "mobilenetv3_small_100", Path("results/checkpoints/sample.pt"))
