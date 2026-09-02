"""Tests for hyperparameter tuning and MLflow run creation."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from src.hyperparameter_tuning import run_all_tuning
from src.mlflow_tracking import setup_mlflow


def test_all_tuning_methods_log_and_return_best_rows(workspace_tmp_path):
    X, y = make_regression(n_samples=60, n_features=5, noise=0.2, random_state=42)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, _, y_val, _ = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    tracking_dir = workspace_tmp_path / "mlruns"
    setup_mlflow("pytest_tuning", tracking_uri=tracking_dir.resolve().as_uri())

    context = {
        "dataset_id": "SYNTH",
        "dataset_version": "test",
        "seed": 42,
        "main_metric": "RMSE",
        "split_details": {},
        "preprocessing": {},
        "feature_engineering": {},
    }
    train_groups = pd.Series([1] * 18 + [2] * 18)

    rows = run_all_tuning(
        X_train=X_train,
        y_train=y_train,
        train_groups=train_groups,
        X_val=X_val,
        y_val=y_val,
        dataset_context=context,
        artifacts_dir=str(workspace_tmp_path / "artifacts"),
        feature_names=[f"x{i}" for i in range(X.shape[1])],
        model_names=["RandomForest"],
        methods=("grid", "random", "optuna"),
        n_trials=1,
        n_splits=2,
        seed=42,
        profile="quick",
    )

    assert len(rows) == 3
    assert all(row["MLflow run ID"] for row in rows)
    assert all(row["Model artifact path"] for row in rows)
    assert all(row["Test score"] is None for row in rows)
    assert all(np.isfinite(row["Validation score"]) for row in rows)
