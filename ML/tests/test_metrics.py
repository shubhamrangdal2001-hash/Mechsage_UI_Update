"""Tests for regression and anomaly/classification metric coverage."""

import numpy as np

from src.evaluate import compute_anomaly_metrics, compute_regression_metrics


def test_regression_metrics_include_required_values():
    metrics = compute_regression_metrics(
        y_true=np.array([100, 80, 60, 40]),
        y_pred=np.array([95, 82, 58, 45]),
        n_features=2,
    )

    for key in ["MAE", "MSE", "RMSE", "R2", "Adjusted_R2", "MAPE", "NASA_Score"]:
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_anomaly_metrics_include_rates_and_auc_values():
    metrics = compute_anomaly_metrics(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 1, 1, 0]),
        y_prob=np.array([0.1, 0.7, 0.8, 0.4]),
    )

    for key in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC",
        "PR_AUC",
        "False_Positive_Rate",
        "False_Negative_Rate",
        "Confusion_Matrix",
    ]:
        assert key in metrics
    assert metrics["Confusion_Matrix"] == [[1, 1], [1, 1]]
