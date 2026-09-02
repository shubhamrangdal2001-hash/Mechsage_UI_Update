"""Reusable visualization helpers for model evaluation artifacts."""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def plot_residuals(y_true, y_pred, title: str, save_path: str) -> str:
    """Save a residual plot for regression models.

    Args:
        y_true: Ground-truth regression values.
        y_pred: Predicted regression values.
        title: Plot title.
        save_path: Output PNG path.

    Returns:
        Output path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    residuals = np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1)
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals, alpha=0.45, s=14, color="#2f6f73")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Predicted value")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    return save_path


def plot_roc_curve(y_true, y_score, title: str, save_path: str) -> Optional[str]:
    """Save an ROC curve when both classes are present."""
    if y_score is None or len(set(np.asarray(y_true).reshape(-1))) < 2:
        return None
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    return save_path


def plot_precision_recall_curve(y_true, y_score, title: str, save_path: str) -> Optional[str]:
    """Save a precision-recall curve when both classes are present."""
    if y_score is None or len(set(np.asarray(y_true).reshape(-1))) < 2:
        return None
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    return save_path
