import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.utils.seed import set_seed
from src.data.dataset import PaddyDataset, get_transforms
from src.models.model_factory import create_model, freeze_backbone, unfreeze_last_blocks

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return epoch_loss, acc, f1

@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())
        all_probs.append(probs)

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    all_probs = np.concatenate(all_probs, axis=0)
    return epoch_loss, acc, f1, np.array(all_targets), all_probs

def train_single_fold(
    model_name: str,
    fold_info: dict,
    clean_df: pd.DataFrame,
    device: torch.device,
    checkpoint_dir: Path,
    stage1_epochs: int = 10,
    stage2_epochs: int = 15,
    patience: int = 5,
    batch_size: int = 16,
    background_mode: str = "standard",
) -> Dict:
    split_id = fold_info["split_id"]
    rep = fold_info["repeat"]
    fold = fold_info["fold"]
    train_idx = fold_info["train_indices"]
    val_idx = fold_info["val_indices"]

    train_df = clean_df.iloc[train_idx].reset_index(drop=True)
    val_df = clean_df.iloc[val_idx].reset_index(drop=True)

    train_loader = DataLoader(
        PaddyDataset(train_df, transform=get_transforms(224, is_train=True, background_mode=background_mode)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        PaddyDataset(val_df, transform=get_transforms(224, is_train=False, background_mode=background_mode)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = create_model(model_name, num_classes=3, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    epoch_logs = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_weights = None
    best_val_probs = None
    best_val_targets = None
    no_improve = 0

    t_start = time.perf_counter()

    # ---- STAGE 1: Train Head Only ----
    freeze_backbone(model, model_name)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)

    for epoch in range(1, stage1_epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_targets, val_probs = evaluate_loader(model, val_loader, criterion, device)
        dt = time.perf_counter() - t0

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_probs = val_probs
            best_val_targets = val_targets

        epoch_logs.append({
            "model_name": experiment_name,
            "split_id": split_id,
            "repeat": rep,
            "fold": fold,
            "stage": 1,
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_f1": tr_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "epoch_sec": dt
        })

    # ---- STAGE 2: Unfreeze Last Blocks and Fine-Tune ----
    unfreeze_last_blocks(model, model_name)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_epochs, eta_min=1e-6)

    for epoch_sub in range(1, stage2_epochs + 1):
        epoch = stage1_epochs + epoch_sub
        t0 = time.perf_counter()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_loss, val_acc, val_f1, val_targets, val_probs = evaluate_loader(model, val_loader, criterion, device)
        dt = time.perf_counter() - t0

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_probs = val_probs
            best_val_targets = val_targets
            no_improve = 0
        else:
            no_improve += 1

        epoch_logs.append({
            "model_name": model_name,
            "split_id": split_id,
            "repeat": rep,
            "fold": fold,
            "stage": 2,
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_f1": tr_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "epoch_sec": dt
        })

        if no_improve >= patience:
            break

    total_training_sec = time.perf_counter() - t_start

    # Save best checkpoint for this fold
    ckpt_path = checkpoint_dir / f"repeat_{rep}_fold_{fold}_best.pt"
    torch.save(best_weights, ckpt_path)

    # Compute best validation accuracy
    best_val_acc = accuracy_score(best_val_targets, best_val_probs.argmax(axis=1))

    return {
        "split_id": split_id,
        "repeat": rep,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_acc": best_val_acc,
        "val_indices": val_idx,
        "val_targets": best_val_targets.tolist(),
        "val_probs": best_val_probs.tolist(),
        "total_training_sec": total_training_sec,
        "checkpoint_path": str(ckpt_path),
        "epoch_logs": epoch_logs
    }

def run_cross_validation_experiment(
    models: List[str] = ["mobilenetv3_small_100", "mobilenetv3_large_100", "efficientnet_b0"],
    manifest_path: Path = Path("data/processed/clean_manifest.csv"),
    splits_path: Path = Path("data/processed/splits.json"),
    results_dir: Path = Path("results"),
    background_mode: str = "standard",
):
    if background_mode not in {"standard", "leaf_crop"}:
        raise ValueError("background_mode must be 'standard' or 'leaf_crop'")
    set_seed(42)
    device = torch.device("cpu")
    print(f"[Train CV] Device: {device} | Threads: {torch.get_num_threads()}")

    clean_df = pd.read_csv(manifest_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    cv_folds = splits["cv_folds"]
    all_models_summary = []

    for model_name in models:
        experiment_name = model_name if background_mode == "standard" else f"{model_name}_{background_mode}"
        print(f"\n=======================================================")
        print(f"  STARTING 15-FOLD CV: {experiment_name}")
        print(f"=======================================================")
        ckpt_dir = results_dir / "checkpoints" / experiment_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        fold_results = []
        all_epoch_logs = []
        model_t0 = time.perf_counter()

        for fold_info in cv_folds:
            split_id = fold_info["split_id"]
            rep = fold_info["repeat"]
            fold = fold_info["fold"]

            res = train_single_fold(
                model_name=model_name,
                fold_info=fold_info,
                clean_df=clean_df,
                device=device,
                checkpoint_dir=ckpt_dir,
                stage1_epochs=10,
                stage2_epochs=15,
                patience=5,
                batch_size=16,
                background_mode=background_mode
            )
            fold_results.append(res)
            all_epoch_logs.extend(res["epoch_logs"])
            print(f"[{model_name}] Rep {rep} Fold {fold} -> Val F1: {res['best_val_f1']:.4f} | Val Acc: {res['best_val_acc']:.4f} | Time: {res['total_training_sec']:.1f}s")

        model_total_time = time.perf_counter() - model_t0

        # Save per-epoch training logs
        logs_df = pd.DataFrame(all_epoch_logs)
        logs_path = results_dir / "tables" / f"{experiment_name}_training_logs.csv"
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        logs_df.to_csv(logs_path, index=False)

        # Save fold-level summary
        fold_summary = [{
            "model_name": model_name,
            "split_id": r["split_id"],
            "repeat": r["repeat"],
            "fold": r["fold"],
            "best_epoch": r["best_epoch"],
            "val_f1": r["best_val_f1"],
            "val_acc": r["best_val_acc"],
            "time_sec": r["total_training_sec"]
        } for r in fold_results]
        fold_df = pd.DataFrame(fold_summary)
        fold_df.to_csv(results_dir / "tables" / f"{experiment_name}_cv_folds.csv", index=False)

        # Save full prediction records for Phase 4 metric calculations
        preds_save_path = results_dir / "tables" / f"{experiment_name}_oof_predictions.json"
        with open(preds_save_path, "w", encoding="utf-8") as f:
            json.dump(fold_results, f)

        mean_f1 = fold_df["val_f1"].mean()
        std_f1 = fold_df["val_f1"].std()
        mean_acc = fold_df["val_acc"].mean()
        std_acc = fold_df["val_acc"].std()

        print(f"\n[{experiment_name} 15-Fold Summary]")
        print(f"  CV Macro-F1: {mean_f1:.4f} +/- {std_f1:.4f}")
        print(f"  CV Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
        print(f"  Total CV Time: {model_total_time:.1f}s ({model_total_time/60:.2f} min)")

        all_models_summary.append({
            "model_name": experiment_name,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "total_time_sec": model_total_time
        })

    print("\n=======================================================")
    print("  ALL 3 MODELS CROSS-VALIDATION COMPLETE")
    print("=======================================================")
    return all_models_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train repeated stratified CV models without touching the locked test set.")
    parser.add_argument("--models", default="mobilenetv3_small_100,mobilenetv3_large_100,efficientnet_b0",
                        help="Comma-separated model names")
    parser.add_argument("--background-mode", choices=["standard", "leaf_crop"], default="standard",
                        help="Optional largest-component crop ablation")
    args = parser.parse_args()
    run_cross_validation_experiment(
        models=[name.strip() for name in args.models.split(",") if name.strip()],
        background_mode=args.background_mode,
    )
