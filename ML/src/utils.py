"""Shared utilities for reproducibility, paths, directories, and plotting."""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed: int = 42):
    """Set random seed for reproducibility across numpy and python."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def get_project_root() -> str:
    """Get absolute path to project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def ensure_directories():
    """Ensure required directories exist within the project."""
    root = get_project_root()
    dirs = [
        os.path.join(root, "data"),
        os.path.join(root, "models"),
        os.path.join(root, "artifacts"),
        os.path.join(root, "reports"),
        os.path.join(root, "notebooks")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def plot_rul_predictions(y_true, y_pred, title="RUL Prediction vs Actual", save_path=None):
    """Plot true vs predicted RUL values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual RUL", color="blue", alpha=0.7)
    plt.plot(y_pred, label="Predicted RUL", color="red", linestyle="--", alpha=0.8)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("RUL (Cycles)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(cm, classes=["Nominal", "Anomaly"], title="Confusion Matrix", save_path=None):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
