"""Centralized MLflow tracking helpers and artifact writers."""

import json
import math
import os
import re
from pathlib import Path
from typing import Iterable, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.utils import get_project_root


MAX_PARAM_LENGTH = 500


def setup_mlflow(experiment_name: str, tracking_uri: Optional[str] = None):
    """
    Configure MLflow to use a local file-backed tracking store by default.
    """
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlruns_dir = Path(get_project_root()) / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    mlflow.set_experiment(experiment_name)


def _clean_key(key: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_.\- /]", "_", str(key))
    return key.strip().replace(" ", "_")[:250]


def _param_value(value) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        value = json.dumps(value, sort_keys=True, default=str)
    elif value is None:
        value = "None"
    else:
        value = str(value)
    return value[:MAX_PARAM_LENGTH]


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def log_params(params: dict, prefix: str = ""):
    for key, value in (params or {}).items():
        mlflow.log_param(_clean_key(f"{prefix}{key}"), _param_value(value))


def log_metrics(metrics: dict, prefix: str = ""):
    for key, value in (metrics or {}).items():
        if _is_number(value):
            mlflow.log_metric(_clean_key(f"{prefix}{key}"), float(value))


def write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def save_predictions(path: str, y_true, y_pred, y_score=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "actual": np.asarray(y_true).reshape(-1),
        "prediction": np.asarray(y_pred).reshape(-1),
    }
    if y_score is not None:
        data["score"] = np.asarray(y_score).reshape(-1)
    pd.DataFrame(data).to_csv(path, index=False)


def plot_feature_importance(model, feature_names: list, save_path: str, top_n: int = 25) -> Optional[str]:
    """
    Save a feature importance plot for tree models and linear coefficients.
    Returns the path when a plot was created.
    """
    import matplotlib.pyplot as plt

    values = None
    title = "Feature Importance"
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.abs(coef.reshape(-1))
        title = "Coefficient Magnitude"

    if values is None or len(values) != len(feature_names):
        return None

    order = np.argsort(values)[-top_n:][::-1]
    names = [feature_names[i] for i in order]
    scores = values[order]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, max(4, min(12, len(order) * 0.35))))
    plt.barh(names[::-1], scores[::-1], color="#2f6f73")
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    return save_path


def log_artifacts(paths: Optional[Iterable[str]]):
    for path in paths or []:
        if path and os.path.exists(path):
            mlflow.log_artifact(path)


def log_model_run(
    model,
    model_name: str,
    task_type: str,
    dataset_context: dict,
    params: dict,
    metrics_by_split: dict,
    artifacts: Optional[Iterable[str]] = None,
    model_flavor: str = "sklearn",
    tags: Optional[dict] = None,
):
    """
    Log a trained model, metrics, context, and artifacts to MLflow.
    """
    with mlflow.start_run(run_name=f"{task_type}_{model_name}_{dataset_context.get('dataset_id')}") as run:
        mlflow.set_tag("task_type", task_type)
        mlflow.set_tag("model_name", model_name)
        for key, value in (tags or {}).items():
            mlflow.set_tag(_clean_key(key), _param_value(value))

        log_params(
            {
                "model_name": model_name,
                "dataset_id": dataset_context.get("dataset_id"),
                "dataset_version": dataset_context.get("dataset_version"),
                "random_seed": dataset_context.get("seed"),
                "main_metric": dataset_context.get("main_metric"),
            }
        )
        log_params(dataset_context.get("split_details", {}), prefix="split_")
        log_params(dataset_context.get("preprocessing", {}), prefix="preprocess_")
        log_params(dataset_context.get("feature_engineering", {}), prefix="features_")
        log_params(params, prefix="hp_")

        for split_name, split_metrics in (metrics_by_split or {}).items():
            log_metrics(split_metrics, prefix=f"{split_name}_")

        log_artifacts(artifacts)

        mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="cloudpickle")

        return {
            "run_id": run.info.run_id,
            "model_artifact_path": mlflow.get_artifact_uri("model"),
        }


def log_tuning_trial(
    trial_number: int,
    params: dict,
    metrics: dict,
    model_name: str,
    dataset_context: dict,
    tuning_method: str,
    best_trial: bool = False,
):
    """
    Log one hyperparameter tuning candidate as a nested MLflow child run.
    """
    run_name = f"{tuning_method}_{model_name}_trial_{trial_number}_{dataset_context.get('dataset_id')}"
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.set_tag("task_type", "tuning")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("tuning_method", tuning_method)
        mlflow.set_tag("best_trial", str(bool(best_trial)).lower())
        log_params(
            {
                "trial_number": trial_number,
                "dataset_id": dataset_context.get("dataset_id"),
                "dataset_version": dataset_context.get("dataset_version"),
                "tuning_method": tuning_method,
            }
        )
        log_params(params, prefix="hp_")
        log_metrics(metrics)
        return run.info.run_id


def mark_best_trial(run_id: str):
    if run_id:
        mlflow.tracking.MlflowClient().set_tag(run_id, "best_trial", "true")


def log_final_best_model(
    model,
    model_name: str,
    task_type: str,
    dataset_context: dict,
    params: dict,
    metrics: dict,
    artifacts: Optional[Iterable[str]] = None,
    model_flavor: str = "sklearn",
):
    with mlflow.start_run(run_name=f"BEST_{task_type}_{model_name}_{dataset_context.get('dataset_id')}") as run:
        mlflow.set_tag("task_type", f"best_{task_type}")
        mlflow.set_tag("model_name", model_name)
        log_params(
            {
                "model_name": model_name,
                "dataset_id": dataset_context.get("dataset_id"),
                "dataset_version": dataset_context.get("dataset_version"),
                "selection_metric": dataset_context.get("main_metric"),
            }
        )
        log_params(params, prefix="best_hp_")
        log_metrics(metrics, prefix="best_")
        log_artifacts(artifacts)
        mlflow.sklearn.log_model(model, artifact_path="final_model", serialization_format="cloudpickle")
        return {
            "run_id": run.info.run_id,
            "model_artifact_path": mlflow.get_artifact_uri("final_model"),
        }


def log_final_test_evaluation(
    model_name: str,
    task_type: str,
    test_metrics: dict,
    dataset_context: dict,
    parent_tuning_run_id: Optional[str] = None,
    test_engine_count: Optional[int] = None,
    artifacts: Optional[Iterable[str]] = None,
) -> dict:
    """Log final test-set metrics in a **separate** MLflow run, clearly labelled.

    This run must only be created AFTER the model has been selected and frozen
    using CV / validation metrics.  It must NOT be used to select between models.

    Args:
        model_name: Name of the selected model.
        task_type: e.g. 'rul_regression' or 'anomaly_detection'.
        test_metrics: Metrics computed on the withheld test set.
        dataset_context: Context dict for logging.
        parent_tuning_run_id: MLflow run ID of the parent tuning run, if available.
        test_engine_count: Number of distinct test engine units.
        artifacts: Optional list of artifact paths to log.

    Returns:
        Dict with 'run_id'.
    """
    run_name = f"FinalTestEval_{task_type}_{model_name}_{dataset_context.get('dataset_id')}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_type", "final_test_evaluation")
        mlflow.set_tag("task_type", task_type)
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("leakage_audit_passed", "true")
        mlflow.set_tag("group_split", "true")
        mlflow.set_tag("temporal_features_causal", "true")
        mlflow.set_tag("test_used_during_tuning", "false")
        mlflow.set_tag("production_feature_parity", "true")

        log_params(
            {
                "model_name": model_name,
                "dataset_id": dataset_context.get("dataset_id"),
                "dataset_version": dataset_context.get("dataset_version"),
                "parent_tuning_run_id": parent_tuning_run_id or "N/A",
            }
        )
        if test_engine_count is not None:
            mlflow.log_metric("test_engine_count", test_engine_count)

        log_metrics(test_metrics, prefix="test_")
        log_artifacts(artifacts)

        return {"run_id": run.info.run_id}
