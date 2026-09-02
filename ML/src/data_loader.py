"""Data loading and dataset-version helpers for NASA C-MAPSS files."""

import hashlib
import os
import pandas as pd
import numpy as np
from src.utils import get_project_root

COLUMN_NAMES = [
    "unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"
] + [f"sensor_{i}" for i in range(1, 22)]

def get_data_dir() -> str:
    """Locate dataset directory, checking workspace fallback paths."""
    project_root = get_project_root()
    local_data = os.path.join(project_root, "data")
    if os.path.exists(os.path.join(local_data, "train_FD001.txt")):
        return local_data
    
    # Fallback to parent MechSage raw data dir
    mech_sage_data = os.path.abspath(os.path.join(project_root, "..", "data", "NASA C-MAPSS-1 Turbofan Engine Degradation Dataset"))
    if os.path.exists(mech_sage_data):
        return mech_sage_data
    
    return local_data


def get_dataset_file_paths(dataset_id: str = "FD001") -> dict:
    """Return the raw train, test, and RUL file paths for a C-MAPSS subset."""
    data_dir = get_data_dir()
    return {
        "train": os.path.join(data_dir, f"train_{dataset_id}.txt"),
        "test": os.path.join(data_dir, f"test_{dataset_id}.txt"),
        "rul": os.path.join(data_dir, f"RUL_{dataset_id}.txt"),
    }


def compute_dataset_version(dataset_id: str = "FD001") -> str:
    """
    Build a deterministic short hash from the raw files.

    The hash is logged to MLflow as the dataset version so future runs can
    tell whether the exact local data files changed.
    """
    digest = hashlib.sha256()
    for label, path in sorted(get_dataset_file_paths(dataset_id).items()):
        if not os.path.exists(path):
            continue
        digest.update(label.encode("utf-8"))
        digest.update(os.path.basename(path).encode("utf-8"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:12]

def load_cmapss_data(dataset_id: str = "FD001", max_rul_clip: int = 125, anomaly_threshold: int = 30):
    """
    Load NASA C-MAPSS train and test datasets, computing target RUL and anomaly labels.
    
    Args:
        dataset_id: Choice of 'FD001', 'FD002', 'FD003', or 'FD004'
        max_rul_clip: Piecewise linear clip cap for RUL target
        anomaly_threshold: RUL threshold cycles below which sample is labeled anomalous
        
    Returns:
        train_df, test_df
    """
    paths = get_dataset_file_paths(dataset_id)
    train_file = paths["train"]
    test_file = paths["test"]
    rul_file = paths["rul"]
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Dataset file not found at {train_file}")
        
    train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
    truth_rul = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["final_rul"])
    
    # Sort immediately by unit_number and time_in_cycles
    train_df = train_df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)
    test_df = test_df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)
    
    
    # Calculate RUL for train set
    max_cycles_train = train_df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycles_train.columns = ["unit_number", "max_cycle"]
    train_df = train_df.merge(max_cycles_train, on="unit_number", how="left")
    train_df["RUL_raw"] = train_df["max_cycle"] - train_df["time_in_cycles"]
    train_df["RUL"] = train_df["RUL_raw"].clip(upper=max_rul_clip)
    train_df["anomaly"] = (train_df["RUL_raw"] <= anomaly_threshold).astype(int)
    train_df.drop(columns=["max_cycle"], inplace=True)
    
    # Calculate RUL for test set using true ground truth at test end
    max_cycles_test = test_df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycles_test.columns = ["unit_number", "max_cycle"]
    truth_rul["unit_number"] = np.arange(1, len(truth_rul) + 1)
    
    test_df = test_df.merge(max_cycles_test, on="unit_number", how="left")
    test_df = test_df.merge(truth_rul, on="unit_number", how="left")
    test_df["RUL_raw"] = (test_df["max_cycle"] - test_df["time_in_cycles"]) + test_df["final_rul"]
    test_df["RUL"] = test_df["RUL_raw"].clip(upper=max_rul_clip)
    test_df["anomaly"] = (test_df["RUL_raw"] <= anomaly_threshold).astype(int)
    test_df.drop(columns=["max_cycle", "final_rul"], inplace=True)
    
    return train_df, test_df
