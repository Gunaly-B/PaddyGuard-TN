import json
from pathlib import Path
import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from src.utils.seed import set_seed
from src.data.dataset import PaddyDataset, get_transforms

def extract_features(
    manifest_path: Path = Path("data/processed/clean_manifest.csv"),
    cache_path: Path = Path("data/processed/features_efficientnet_b0.npz"),
    model_name: str = "efficientnet_b0",
    batch_size: int = 16
):
    set_seed(42)
    df = pd.read_csv(manifest_path)

    if cache_path.exists():
        print(f"[Baseline] Loading cached features from {cache_path}...")
        data = np.load(cache_path)
        return data["features"], data["labels"], df

    print(f"[Baseline] Extracting pooled features using pretrained {model_name}...")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()

    dataset = PaddyDataset(df, transform=get_transforms(224, is_train=False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    feature_list = []
    label_list = []

    with torch.no_grad():
        for imgs, labels, _ in loader:
            feats = model(imgs) # shape: (B, 1280)
            feature_list.append(feats.cpu().numpy())
            label_list.append(labels.numpy())

    features = np.concatenate(feature_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, labels=labels)
    print(f"[Baseline] Saved features shape={features.shape} to {cache_path}")
    return features, labels, df

def run_baseline_evaluation(
    splits_path: Path = Path("data/processed/splits.json"),
    cache_path: Path = Path("data/processed/features_efficientnet_b0.npz"),
    results_dir: Path = Path("results/tables")
):
    features, labels, df = extract_features(cache_path=cache_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    cv_folds = splits["cv_folds"]
    print(f"[Baseline] Running 15-fold CV evaluation on {len(cv_folds)} folds...")

    lr_results = []
    svm_results = []

    for fold_info in cv_folds:
        split_id = fold_info["split_id"]
        rep = fold_info["repeat"]
        fold = fold_info["fold"]
        train_idx = fold_info["train_indices"]
        val_idx = fold_info["val_indices"]

        X_train, y_train = features[train_idx], labels[train_idx]
        X_val, y_val = features[val_idx], labels[val_idx]

        # 1. Logistic Regression
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_val)
        lr_acc = accuracy_score(y_val, lr_pred)
        lr_f1 = f1_score(y_val, lr_pred, average="macro")
        lr_results.append({
            "model": "Frozen_EffNetB0_LogisticRegression",
            "split_id": split_id,
            "repeat": rep,
            "fold": fold,
            "val_acc": lr_acc,
            "val_macro_f1": lr_f1
        })

        # 2. Linear SVM
        svm = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
        svm.fit(X_train, y_train)
        svm_pred = svm.predict(X_val)
        svm_acc = accuracy_score(y_val, svm_pred)
        svm_f1 = f1_score(y_val, svm_pred, average="macro")
        svm_results.append({
            "model": "Frozen_EffNetB0_LinearSVM",
            "split_id": split_id,
            "repeat": rep,
            "fold": fold,
            "val_acc": svm_acc,
            "val_macro_f1": svm_f1
        })

    all_res = lr_results + svm_results
    res_df = pd.DataFrame(all_res)
    results_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(results_dir / "baseline_cv_folds.csv", index=False)

    summary = []
    for model_name in ["Frozen_EffNetB0_LogisticRegression", "Frozen_EffNetB0_LinearSVM"]:
        sub = res_df[res_df["model"] == model_name]
        acc_mean, acc_std = sub["val_acc"].mean(), sub["val_acc"].std()
        f1_mean, f1_std = sub["val_macro_f1"].mean(), sub["val_macro_f1"].std()
        summary.append({
            "Model": model_name,
            "CV Acc (Mean +/- Std)": f"{acc_mean:.4f} +/- {acc_std:.4f}",
            "CV Macro-F1 (Mean +/- Std)": f"{f1_mean:.4f} +/- {f1_std:.4f}",
            "Mean_Acc": acc_mean,
            "Std_Acc": acc_std,
            "Mean_F1": f1_mean,
            "Std_F1": f1_std
        })
        print(f"\n[Baseline Result] {model_name}:")
        print(f"  CV Accuracy: {acc_mean:.4f} +/- {acc_std:.4f}")
        print(f"  CV Macro-F1: {f1_mean:.4f} +/- {f1_std:.4f}")

    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(results_dir / "baseline_summary.csv", index=False)
    return res_df, sum_df

if __name__ == "__main__":
    run_baseline_evaluation()
