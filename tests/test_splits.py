import json
from pathlib import Path
import pandas as pd
import pytest

def test_splits_integrity():
    splits_path = Path("data/processed/splits.json")
    manifest_path = Path("data/processed/clean_manifest.csv")

    assert splits_path.exists(), "splits.json does not exist"
    assert manifest_path.exists(), "clean_manifest.csv does not exist"

    df = pd.read_csv(manifest_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    total_samples = len(df)
    test_idx = set(splits["locked_test_indices"])
    dev_idx = set(splits["dev_indices"])

    # 1. Test set and Dev set must partition the entire dataset
    assert len(test_idx & dev_idx) == 0, "Leakage: Overlap between locked test and dev sets"
    assert len(test_idx | dev_idx) == total_samples, "Test + Dev does not cover all samples"
    assert len(test_idx) == int(total_samples * splits["metadata"]["test_size_fraction"]), "Test set size mismatch"

    # 2. Test stratification in locked test set
    test_classes = df.iloc[list(test_idx)]["class_name"].value_counts()
    assert len(test_classes.unique()) == 1, "Locked test set is not balanced across classes"

    # 3. Check every CV fold in every repeat
    for fold in splits["cv_folds"]:
        tr_idx = set(fold["train_indices"])
        val_idx = set(fold["val_indices"])

        # Strict isolation
        assert len(tr_idx & val_idx) == 0, f"Leakage: Train-Val overlap in fold {fold['split_id']}"
        assert len(tr_idx & test_idx) == 0, f"Leakage: Train set contains locked test sample in fold {fold['split_id']}"
        assert len(val_idx & test_idx) == 0, f"Leakage: Val set contains locked test sample in fold {fold['split_id']}"
        assert (tr_idx | val_idx | test_idx) == set(range(total_samples)), f"Missing samples in fold {fold['split_id']}"
