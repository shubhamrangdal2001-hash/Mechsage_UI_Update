"""Evaluation metrics for regression, classification, and anomaly detection."""

import math
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


EPSILON = 1e-8


def _as_array(values) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def _safe_auc(metric_fn, y_true, y_score) -> Optional[float]:
    if y_score is None or len(np.unique(y_true)) < 2:
        return None
    try:
        return float(metric_fn(y_true, y_score))
    except ValueError:
        return None


def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def nasa_scoring_function(y_true, y_pred) -> float:
    """NASA PHM08 asymmetric RUL score. Lower is better."""
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    delta = y_pred - y_true
    score = np.where(delta < 0, np.exp(-delta / 13.0) - 1, np.exp(delta / 10.0) - 1)
    return float(np.sum(score))


def adjusted_r2(y_true, y_pred, n_features: Optional[int] = None) -> float:
    base_r2 = float(r2_score(y_true, y_pred))
    if not n_features:
        return base_r2
    n_samples = len(_as_array(y_true))
    if n_samples <= n_features + 1:
        return base_r2
    return float(1 - (1 - base_r2) * (n_samples - 1) / (n_samples - n_features - 1))


def mape(y_true, y_pred) -> float:
    y_true = _as_array(y_true).astype(float)
    y_pred = _as_array(y_pred).astype(float)
    denominator = np.maximum(np.abs(y_true), EPSILON)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def compute_regression_metrics(y_true, y_pred, n_features: Optional[int] = None) -> dict:
    """Compute the full regression metric set logged to MLflow."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "Adjusted_R2": adjusted_r2(y_true, y_pred, n_features=n_features),
        "MAPE": mape(y_true, y_pred),
        "NASA_Score": nasa_scoring_function(y_true, y_pred),
    }


def compute_rul_metrics(y_true, y_pred, n_features: Optional[int] = None) -> dict:
    return compute_regression_metrics(y_true, y_pred, n_features=n_features)


def compute_classification_metrics(y_true, y_pred, y_score=None) -> dict:
    """Compute binary classification metrics with stable zero-division behavior."""
    y_true = _as_array(y_true).astype(int)
    y_pred = _as_array(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1_Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": _safe_auc(roc_auc_score, y_true, y_score),
        "PR_AUC": _safe_auc(average_precision_score, y_true, y_score),
        "False_Positive_Rate": float(fpr),
        "False_Negative_Rate": float(fnr),
        "Confusion_Matrix": cm.tolist(),
    }


def compute_anomaly_metrics(y_true, y_pred, y_prob=None) -> dict:
    return compute_classification_metrics(y_true, y_pred, y_score=y_prob)
