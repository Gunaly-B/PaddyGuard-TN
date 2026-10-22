import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
import torch
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.data.dataset import PaddyDataset, get_transforms
from src.models.model_factory import create_model
from src.models.profiler import profile_model_efficiency

def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error (ECE) and bin accuracies/confidences."""
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
            bin_counts.append(np.sum(in_bin))
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_counts.append(0)

    return float(ece), np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)

def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """Compute non-parametric bootstrap 95% confidence intervals."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    point_est = metric_fn(y_true, y_pred)
    boot_vals = []

    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_vals.append(metric_fn(y_true[idx], y_pred[idx]))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_vals, 100 * alpha))
    upper = float(np.percentile(boot_vals, 100 * (1.0 - alpha)))
    return float(point_est), lower, upper

def evaluate_predictions(y_true: np.ndarray, y_probs: np.ndarray, class_names: List[str]) -> Dict:
    y_pred = np.argmax(y_probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    # One-vs-rest ROC-AUC and PR-AUC
    y_true_onehot = np.eye(len(class_names))[y_true]
    roc_aucs, pr_aucs = {}, {}
    for i, cls in enumerate(class_names):
        try:
            roc_aucs[cls] = float(roc_auc_score(y_true_onehot[:, i], y_probs[:, i]))
        except Exception:
            roc_aucs[cls] = float("nan")
        try:
            pr_aucs[cls] = float(average_precision_score(y_true_onehot[:, i], y_probs[:, i]))
        except Exception:
            pr_aucs[cls] = float("nan")

    # Bootstrap CIs
    _, acc_low, acc_high = bootstrap_metric_ci(y_true, y_pred, accuracy_score)
    _, f1_low, f1_high = bootstrap_metric_ci(y_true, y_pred, lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0))

    # ECE
    ece, bin_accs, bin_confs, bin_counts = compute_ece(y_true, y_probs)

    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    return {
        "accuracy": acc,
        "acc_ci_95": (acc_low, acc_high),
        "macro_f1": macro_f1,
        "f1_ci_95": (f1_low, f1_high),
        "weighted_f1": weighted_f1,
        "mcc": mcc,
        "kappa": kappa,
        "ece": ece,
        "per_class": {
            cls: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "roc_auc": roc_aucs[cls],
                "pr_auc": pr_aucs[cls],
                "support": int(supp[i])
            } for i, cls in enumerate(class_names)
        },
        "cm_raw": cm_raw,
        "cm_norm": cm_norm,
        "classification_rep": classification_report(y_true, y_pred, target_names=class_names, digits=4)
    }

def run_phase4_evaluation(
    results_dir: Path = Path("results"),
    data_dir: Path = Path("data/processed")
):
    set_seed(42)
    with open(data_dir / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    class_names = splits["metadata"]["class_names"]
    clean_df = pd.read_csv(data_dir / "clean_manifest.csv")
    models = ["mobilenetv3_small_100", "mobilenetv3_large_100", "efficientnet_b0"]

    # 1. Evaluate CV for all 3 fine-tuned models
    cv_eval_results = {}
    model_profiles = {}

    for m in models:
        oof_path = results_dir / "tables" / f"{m}_oof_predictions.json"
        with open(oof_path, "r", encoding="utf-8") as f:
            oof_data = json.load(f)

        # Aggregate all out-of-fold predictions across the 15 folds
        all_targets = []
        all_probs = []
        for fold_res in oof_data:
            all_targets.extend(fold_res["val_targets"])
            all_probs.extend(fold_res["val_probs"])

        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        # Compute comprehensive metrics
        metrics = evaluate_predictions(all_targets, all_probs, class_names)
        cv_eval_results[m] = metrics

        # Profile efficiency
        sample_model = create_model(m, num_classes=3, pretrained=False)
        prof = profile_model_efficiency(sample_model, m)

        # Calculate model checkpoint file size
        sample_ckpt = results_dir / "checkpoints" / m / "repeat_0_fold_0_best.pt"
        if sample_ckpt.exists():
            prof["ckpt_size_mb"] = round(sample_ckpt.stat().st_size / (1024 * 1024), 2)
        else:
            prof["ckpt_size_mb"] = 0.0

        # Get average training time per epoch
        train_logs = pd.read_csv(results_dir / "tables" / f"{m}_training_logs.csv")
        prof["avg_epoch_sec"] = round(train_logs["epoch_sec"].mean(), 2)
        prof["total_train_min"] = round(train_logs["epoch_sec"].sum() / 60.0, 2)

        model_profiles[m] = prof

    # 2. Add Baseline results for comparison
    baseline_df = pd.read_csv(results_dir / "tables" / "baseline_summary.csv")

    # 3. Build Model Comparison Table
    comparison_rows = []

    # Add Baselines
    for _, brow in baseline_df.iterrows():
        comparison_rows.append({
            "Architecture": brow["Model"],
            "CV Macro-F1 (Mean +/- Std)": brow["CV Macro-F1 (Mean +/- Std)"],
            "CV Accuracy (Mean +/- Std)": brow["CV Acc (Mean +/- Std)"],
            "Bootstrap 95% CI (F1)": "N/A (Linear Baseline)",
            "MCC": "N/A",
            "Cohen's Kappa": "N/A",
            "ECE": "N/A",
            "Params (M)": "N/A",
            "GMACs": "N/A",
            "Disk (MB)": "N/A",
            "BS=1 Latency (ms)": "N/A",
            "BS=16 Latency (ms)": "N/A",
            "Throughput (img/s)": "N/A",
            "Train Time": "N/A"
        })

    # Add Deep Models
    for m in models:
        met = cv_eval_results[m]
        prof = model_profiles[m]

        # Get fold-level mean and std
        fold_csv = pd.read_csv(results_dir / "tables" / f"{m}_cv_folds.csv")
        f1_m, f1_s = fold_csv["val_f1"].mean(), fold_csv["val_f1"].std()
        acc_m, acc_s = fold_csv["val_acc"].mean(), fold_csv["val_acc"].std()

        comparison_rows.append({
            "Architecture": m,
            "CV Macro-F1 (Mean +/- Std)": f"{f1_m:.4f} +/- {f1_s:.4f}",
            "CV Accuracy (Mean +/- Std)": f"{acc_m:.4f} +/- {acc_s:.4f}",
            "Bootstrap 95% CI (F1)": f"[{met['f1_ci_95'][0]:.4f}, {met['f1_ci_95'][1]:.4f}]",
            "MCC": f"{met['mcc']:.4f}",
            "Cohen's Kappa": f"{met['kappa']:.4f}",
            "ECE": f"{met['ece']:.4f}",
            "Params (M)": f"{prof['params_million']:.2f}",
            "GMACs": f"{prof['gmacs']:.4f}",
            "Disk (MB)": f"{prof['ckpt_size_mb']:.2f}",
            "BS=1 Latency (ms)": f"{prof['bs1_latency_mean_ms']:.1f}",
            "BS=16 Latency (ms)": f"{prof['bs16_latency_mean_ms']:.1f}",
            "Throughput (img/s)": f"{prof['throughput_img_per_s']:.1f}",
            "Train Time": f"{prof['total_train_min']:.1f} min"
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(results_dir / "tables" / "model_comparison.csv", index=False)

    with open(results_dir / "tables" / "model_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Model Comparison Table (Cross-Validation)\n\n")
        f.write(comp_df.to_markdown(index=False))

    print("\n[Phase 4] Model Comparison Table saved.")
    print(comp_df.to_string())

    # 4. Generate Plots
    # A. Training Curves
    plot_training_curves(models, results_dir)
    # B. Confusion Matrices
    plot_confusion_matrices(models, cv_eval_results, class_names, results_dir)
    # C. Calibration Curves
    plot_calibration_curves(models, data_dir, results_dir)

    # 5. Select Winning Model
    # Rule: Pick best by CV macro-F1. If CIs overlap, pick cheaper model.
    winner_name = select_winning_model(models, cv_eval_results, model_profiles)
    print(f"\n=======================================================")
    print(f"  SELECTED WINNER MODEL: {winner_name}")
    print(f"=======================================================")

    # 6. Evaluate Winner on Locked Test Set EXACTLY ONCE
    test_results = evaluate_locked_test_set(winner_name, splits, clean_df, class_names, results_dir)

    return comp_df, winner_name, test_results

def select_winning_model(models: List[str], cv_results: Dict, profiles: Dict) -> str:
    # Sort by Macro-F1 point estimate
    sorted_models = sorted(models, key=lambda m: cv_results[m]["macro_f1"], reverse=True)
    best_m = sorted_models[0]
    best_ci = cv_results[best_m]["f1_ci_95"]

    print(f"\n[Model Selection] Best candidate: {best_m} (Macro-F1: {cv_results[best_m]['macro_f1']:.4f}, 95% CI: [{best_ci[0]:.4f}, {best_ci[1]:.4f}])")

    # Check if competitor CIs overlap with best
    statistically_tied = []
    for m in sorted_models:
        ci = cv_results[m]["f1_ci_95"]
        # Overlap if max(low1, low2) <= min(high1, high2)
        overlap = max(best_ci[0], ci[0]) <= min(best_ci[1], ci[1])
        if overlap:
            statistically_tied.append(m)
            print(f"  - {m}: 95% CI [{ci[0]:.4f}, {ci[1]:.4f}] overlaps with {best_m}. Statistically indistinguishable.")
        else:
            print(f"  - {m}: 95% CI [{ci[0]:.4f}, {ci[1]:.4f}] does NOT overlap with {best_m}.")

    if len(statistically_tied) > 1:
        # Pick the cheaper model (fewest parameters / lowest latency)
        winner = min(statistically_tied, key=lambda m: profiles[m]["total_params"])
        print(f"[Model Selection Decision] Confidence intervals overlap among {statistically_tied}. Selecting the cheaper architecture: {winner} ({profiles[winner]['params_million']}M params, {profiles[winner]['bs1_latency_mean_ms']}ms latency)")
        return winner
    else:
        return best_m

def evaluate_locked_test_set(
    winner_name: str,
    splits: dict,
    clean_df: pd.DataFrame,
    class_names: List[str],
    results_dir: Path
) -> Dict:
    print(f"\n[Locked Test Set] Evaluating winner '{winner_name}' on locked test set EXACTLY ONCE...")
    test_idx = splits["locked_test_indices"]
    test_df = clean_df.iloc[test_idx].reset_index(drop=True)
    print(f"[Locked Test Set] Test sample count: {len(test_df)}")

    test_loader = DataLoader(
        PaddyDataset(test_df, transform=get_transforms(224, is_train=False)),
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    # Find the best checkpoint among CV folds (highest val F1)
    fold_csv = pd.read_csv(results_dir / "tables" / f"{winner_name}_cv_folds.csv")
    best_fold_row = fold_csv.loc[fold_csv["val_f1"].idxmax()]
    best_rep = int(best_fold_row["repeat"])
    best_fold = int(best_fold_row["fold"])
    best_ckpt = results_dir / "checkpoints" / winner_name / f"repeat_{best_rep}_fold_{best_fold}_best.pt"
    print(f"[Locked Test Set] Loading best checkpoint from Repeat {best_rep} Fold {best_fold}: {best_ckpt}")

    model = create_model(winner_name, num_classes=3, pretrained=False)
    model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
    model.eval()

    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).numpy()
            preds = outputs.argmax(dim=1).numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            all_probs.append(probs)

    all_targets = np.array(all_targets)
    all_probs = np.concatenate(all_probs, axis=0)

    test_metrics = evaluate_predictions(all_targets, all_probs, class_names)

    # Save locked test predictions and report
    test_records = []
    for i, idx in enumerate(test_idx):
        row = clean_df.iloc[idx]
        test_records.append({
            "test_idx": idx,
            "filename": row["filename"],
            "true_class": row["class_name"],
            "pred_class": class_names[all_preds[i]],
            "confidence": float(np.max(all_probs[i])),
            "correct": bool(row["class_name"] == class_names[all_preds[i]]),
            "prob_bacterial_leaf_blight": float(all_probs[i, 0]),
            "prob_brown_spot": float(all_probs[i, 1]),
            "prob_leaf_smut": float(all_probs[i, 2])
        })
    test_pred_df = pd.DataFrame(test_records)
    test_pred_df.to_csv(results_dir / "tables" / "locked_test_predictions.csv", index=False)

    # Plot Test Confusion Matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(test_metrics["cm_raw"], annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title(f"Locked Test Set: Raw CM ({winner_name})")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")

    sns.heatmap(test_metrics["cm_norm"], annot=True, fmt=".2f", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title(f"Locked Test Set: Normalized CM ({winner_name})")
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")

    plt.tight_layout()
    fig.savefig(results_dir / "figures" / "locked_test_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Locked Test Set Evaluation Complete]")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f} (95% CI: [{test_metrics['acc_ci_95'][0]:.4f}, {test_metrics['acc_ci_95'][1]:.4f}])")
    print(f"  Macro-F1: {test_metrics['macro_f1']:.4f} (95% CI: [{test_metrics['f1_ci_95'][0]:.4f}, {test_metrics['f1_ci_95'][1]:.4f}])")
    print(f"  MCC: {test_metrics['mcc']:.4f} | Kappa: {test_metrics['kappa']:.4f} | ECE: {test_metrics['ece']:.4f}")
    print("\nClassification Report:\n", test_metrics["classification_rep"])

    # Save locked test report markdown
    report_md = f"""# Locked Final Test Set Evaluation Report

**Model Evaluated:** `{winner_name}` (Single Locked Run - Zero Prior Exposure)
**Checkpoint:** `{best_ckpt.name}` (from CV Repeat {best_rep}, Fold {best_fold})
**Test Set Size:** {len(test_df)} samples ({len(test_df)//3} per class, strictly stratified)

---

## 1. Summary Metrics

| Metric | Point Estimate | Non-Parametric Bootstrap 95% CI |
|:---|:---:|:---:|
| **Accuracy** | **{test_metrics['accuracy']:.4f}** | [{test_metrics['acc_ci_95'][0]:.4f}, {test_metrics['acc_ci_95'][1]:.4f}] |
| **Macro-F1** | **{test_metrics['macro_f1']:.4f}** | [{test_metrics['f1_ci_95'][0]:.4f}, {test_metrics['f1_ci_95'][1]:.4f}] |
| **Weighted-F1** | **{test_metrics['weighted_f1']:.4f}** | - |
| **Matthews Corr Coef (MCC)** | **{test_metrics['mcc']:.4f}** | - |
| **Cohen's Kappa** | **{test_metrics['kappa']:.4f}** | - |
| **Expected Calibration Error (ECE)** | **{test_metrics['ece']:.4f}** | - |

---

## 2. Per-Class Performance

| Class Name | Precision | Recall | F1-Score | ROC-AUC (OvR) | PR-AUC (OvR) | Test Support |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for cls in class_names:
        c = test_metrics["per_class"][cls]
        report_md += f"| {cls} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} | {c['roc_auc']:.4f} | {c['pr_auc']:.4f} | {c['support']} |\n"

    report_md += f"""
---

## 3. Classification Report & Confusion Matrix

```
{test_metrics['classification_rep']}
```

- Figure: [`results/figures/locked_test_confusion_matrix.png`](file:///c:/Users/gunal/Desktop/Paddy%20Disease%20Diagnosis/results/figures/locked_test_confusion_matrix.png)
- Per-sample predictions: [`results/tables/locked_test_predictions.csv`](file:///c:/Users/gunal/Desktop/Paddy%20Disease%20Diagnosis/results/tables/locked_test_predictions.csv)
"""
    with open(results_dir / "reports" / "locked_test_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    return test_metrics

def plot_training_curves(models: List[str], results_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"mobilenetv3_small_100": "#1f77b4", "mobilenetv3_large_100": "#ff7f0e", "efficientnet_b0": "#2ca02c"}

    for m in models:
        df = pd.read_csv(results_dir / "tables" / f"{m}_training_logs.csv")
        # Average across all 15 folds per epoch
        epoch_grp = df.groupby("epoch").agg({"train_loss": "mean", "val_loss": "mean", "val_f1": "mean"}).reset_index()
        ax1.plot(epoch_grp["epoch"], epoch_grp["train_loss"], label=f"{m} (Train Loss)", color=colors[m], linestyle="--")
        ax1.plot(epoch_grp["epoch"], epoch_grp["val_loss"], label=f"{m} (Val Loss)", color=colors[m])
        ax2.plot(epoch_grp["epoch"], epoch_grp["val_f1"], label=m, color=colors[m], linewidth=2)

    ax1.axvline(10, color="gray", linestyle=":", label="Stage 2 Unfreeze")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss (Label Smoothing=0.1)")
    ax1.set_title("Training & Validation Loss Across Stages")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.axvline(10, color="gray", linestyle=":", label="Stage 2 Unfreeze")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Macro-F1")
    ax2.set_title("Validation Macro-F1 Progression")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(results_dir / "figures" / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Phase 4] Saved training curves to results/figures/training_curves.png")

def plot_confusion_matrices(models: List[str], cv_results: Dict, class_names: List[str], results_dir: Path):
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 4 * len(models)))
    for i, m in enumerate(models):
        cm_raw = cv_results[m]["cm_raw"]
        cm_norm = cv_results[m]["cm_norm"]

        sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=axes[i, 0])
        axes[i, 0].set_title(f"{m} - Raw Confusion Matrix (CV Aggregated)")
        axes[i, 0].set_ylabel("True")
        axes[i, 0].set_xlabel("Predicted")

        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                    xticklabels=class_names, yticklabels=class_names, ax=axes[i, 1])
        axes[i, 1].set_title(f"{m} - Normalized Confusion Matrix (CV Aggregated)")
        axes[i, 1].set_ylabel("True")
        axes[i, 1].set_xlabel("Predicted")

    plt.tight_layout()
    fig.savefig(results_dir / "figures" / "confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Phase 4] Saved confusion matrices to results/figures/confusion_matrices.png")

def plot_calibration_curves(models: List[str], data_dir: Path, results_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    colors = {"mobilenetv3_small_100": "#1f77b4", "mobilenetv3_large_100": "#ff7f0e", "efficientnet_b0": "#2ca02c"}

    for m in models:
        oof_path = results_dir / "tables" / f"{m}_oof_predictions.json"
        with open(oof_path, "r", encoding="utf-8") as f:
            oof_data = json.load(f)
        all_targets, all_probs = [], []
        for f_res in oof_data:
            all_targets.extend(f_res["val_targets"])
            all_probs.extend(f_res["val_probs"])

        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        ece, bin_accs, bin_confs, _ = compute_ece(all_targets, all_probs, n_bins=10)
        ax.plot(bin_confs, bin_accs, marker="o", color=colors[m], label=f"{m} (ECE={ece:.4f})")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagrams & Expected Calibration Error (ECE)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(results_dir / "figures" / "calibration_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Phase 4] Saved calibration curves to results/figures/calibration_curves.png")

if __name__ == "__main__":
    run_phase4_evaluation()
