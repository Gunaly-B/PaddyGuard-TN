import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold

def create_splits(
    manifest_path: Path = Path("data/processed/clean_manifest.csv"),
    output_path: Path = Path("data/processed/splits.json"),
    test_size: float = 0.20,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42
):
    print(f"[Splits] Loading manifest from {manifest_path}...")
    df = pd.read_csv(manifest_path)
    total_samples = len(df)
    labels = df["class_name"].values
    indices = np.arange(total_samples)

    print(f"[Splits] Total samples: {total_samples}")
    print(f"[Splits] Class distribution:\n{df['class_name'].value_counts().to_string()}")

    # 1. Stratified split to lock final test set (~20%)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    dev_idx, test_idx = next(sss.split(indices, labels))

    test_idx = sorted(test_idx.tolist())
    dev_idx = sorted(dev_idx.tolist())

    dev_labels = labels[dev_idx]
    test_labels = labels[test_idx]

    print(f"\n[Splits] Locked Test Set size: {len(test_idx)} ({len(test_idx)/total_samples*100:.1f}%)")
    test_classes, test_counts = np.unique(test_labels, return_counts=True)
    for c, cnt in zip(test_classes, test_counts):
        print(f"  - Test [{c}]: {cnt}")

    print(f"[Splits] Remainder (Dev Set) size: {len(dev_idx)} ({len(dev_idx)/total_samples*100:.1f}%)")
    dev_classes, dev_counts = np.unique(dev_labels, return_counts=True)
    for c, cnt in zip(dev_classes, dev_counts):
        print(f"  - Dev [{c}]: {cnt}")

    # 2. Repeated Stratified 5-Fold CV on the Dev Set
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    cv_folds = []
    # Note: rskf returns indices relative to the dev set (0..len(dev_idx)-1)
    # We map them back to global manifest indices!
    dev_idx_arr = np.array(dev_idx)

    fold_global_counter = 0
    for repeat_idx in range(n_repeats):
        for fold_idx in range(n_splits):
            pass # We will iterate with enumerate(rskf.split(dev_idx_arr, dev_labels))

    for i, (train_sub_idx, val_sub_idx) in enumerate(rskf.split(dev_idx_arr, dev_labels)):
        repeat_num = i // n_splits
        fold_num = i % n_splits

        # Map to global indices
        train_global = sorted(dev_idx_arr[train_sub_idx].tolist())
        val_global = sorted(dev_idx_arr[val_sub_idx].tolist())

        # Strict Assertions for Zero Overlap and Exhaustive Coverage
        set_train = set(train_global)
        set_val = set(val_global)
        set_test = set(test_idx)

        assert len(set_train & set_val) == 0, f"Overlap between train and val in repeat {repeat_num} fold {fold_num}!"
        assert len(set_train & set_test) == 0, f"Data leakage: train set contains locked test samples!"
        assert len(set_val & set_test) == 0, f"Data leakage: val set contains locked test samples!"
        assert (set_train | set_val | set_test) == set(indices), "Union of train, val, and test does not match total indices!"
        assert len(train_global) + len(val_global) + len(test_idx) == total_samples, "Index counts do not sum to total samples!"

        cv_folds.append({
            "split_id": i,
            "repeat": repeat_num,
            "fold": fold_num,
            "train_indices": train_global,
            "val_indices": val_global,
            "train_count": len(train_global),
            "val_count": len(val_global)
        })

    print(f"\n[Splits] Generated {len(cv_folds)} CV folds ({n_repeats} repeats x {n_splits} folds).")
    print(f"[Splits] Sample fold 0 train count: {cv_folds[0]['train_count']}, val count: {cv_folds[0]['val_count']}")

    splits_data = {
        "metadata": {
            "total_samples": total_samples,
            "test_size_fraction": test_size,
            "test_count": len(test_idx),
            "dev_count": len(dev_idx),
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "random_state": random_state,
            "class_names": sorted(df["class_name"].unique().tolist())
        },
        "locked_test_indices": test_idx,
        "dev_indices": dev_idx,
        "cv_folds": cv_folds
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)

    print(f"[Splits] Saved all split indices and validation metadata to {output_path}")
    print("[Splits] All assertions PASSED: ZERO leakage, ZERO overlap, strict stratification.")
    return splits_data

if __name__ == "__main__":
    create_splits()
