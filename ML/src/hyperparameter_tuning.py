"""Hyperparameter tuning with GridSearchCV, RandomizedSearchCV, and Optuna.

Key leakage-prevention guarantees
----------------------------------
- Estimators are wrapped inside a sklearn Pipeline with StandardScaler so the
  scaler is fitted only on each training fold, never on validation or test data.
- GroupKFold is used so all cycles from one engine stay in the same fold.
- The test set (X_test / y_test) is **never** seen during tuning. Only
  cross-validation results are used for model selection.
- The final best model is evaluated on the test set only via
  ``run_final_evaluation`` in pipeline.py, not here.
"""

import itertools
import json
import logging
import os
import time
from typing import Iterable, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.evaluate import compute_regression_metrics
from src.mlflow_tracking import (
    log_metrics,
    log_params,
    log_tuning_trial,
    mark_best_trial,
    plot_feature_importance,
    save_predictions,
)
from src.utils import set_seed


optuna.logging.set_verbosity(optuna.logging.WARNING)
LOGGER = logging.getLogger("mechsage.tuning")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORBIDDEN_FEATURES = {
    "cycle_norm",
    "normalized_cycle",
    "life_fraction",
    "percentage_life_used",
    "max_cycle",
    "final_cycle",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rmse_score(metrics: dict) -> float:
    return float(metrics["RMSE"])


def _space_size(param_space: dict) -> int:
    size = 1
    for values in param_space.values():
        if isinstance(values, (list, tuple)):
            size *= len(values)
        else:
            return 10**9
    return size


def _check_forbidden_features(feature_names: list) -> None:
    """Raise if any forbidden (future-leaking) feature is present in the list."""
    bad = FORBIDDEN_FEATURES.intersection(set(feature_names))
    if bad:
        raise ValueError(
            f"Forbidden features detected in model inputs: {sorted(bad)}. "
            "These features use future information and must be removed before training."
        )


def validate_group_folds(splitter, X, y, groups: pd.Series) -> None:
    """Assert that no engine unit appears in both sides of any CV fold.

    Args:
        splitter: A fitted GroupKFold splitter instance.
        X: Feature array.
        y: Target array.
        groups: Pandas Series of group labels (unit_number) aligned with X and y.

    Raises:
        ValueError: If any engine unit overlaps between train and validation in a fold.
    """
    for fold_number, (train_idx, val_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        train_units = set(groups.iloc[train_idx])
        val_units = set(groups.iloc[val_idx])
        overlap = train_units.intersection(val_units)
        if overlap:
            raise ValueError(
                f"Group leakage detected in fold {fold_number}. "
                f"Overlapping units: {sorted(overlap)}"
            )
        LOGGER.debug(
            "Fold %d — train rows=%d (%d units), val rows=%d (%d units), overlap=%d",
            fold_number,
            len(train_idx),
            len(train_units),
            len(val_idx),
            len(val_units),
            0,
        )


def _build_pipeline(estimator) -> Pipeline:
    """Wrap an sklearn estimator in a StandardScaler pipeline.

    This guarantees the scaler is fitted *only* on each training fold and
    never sees validation or test data.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def _prefix_param_grid(param_space: dict) -> dict:
    """Add 'model__' prefix to all keys so they target the Pipeline's estimator."""
    return {f"model__{k}": v for k, v in param_space.items()}


# ---------------------------------------------------------------------------
# Single-candidate evaluation (NO test data)
# ---------------------------------------------------------------------------

def _evaluate_candidate(estimator, params, X_train, y_train, X_val, y_val):
    """Fit a candidate on train and evaluate on val only. No test data involved.

    Args:
        estimator: Base sklearn estimator (already wrapped in Pipeline).
        params: Hyperparameter dict with 'model__' prefix.
        X_train / y_train: Training features and targets.
        X_val / y_val: Validation features and targets.

    Returns:
        Tuple: (fitted_model, train_metrics, val_metrics, summary_dict)
    """
    model = clone(estimator).set_params(**params)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    runtime = time.perf_counter() - started

    n_features = X_train.shape[1]
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_metrics = compute_regression_metrics(y_train, train_pred, n_features=n_features)
    val_metrics = compute_regression_metrics(y_val, val_pred, n_features=n_features)

    summary = {
        "train_score": _rmse_score(train_metrics),
        "validation_score": _rmse_score(val_metrics),
        "runtime_seconds": runtime,
        "train_RMSE": train_metrics["RMSE"],
        "validation_RMSE": val_metrics["RMSE"],
        "generalization_gap": val_metrics["RMSE"] - train_metrics["RMSE"],
    }
    return model, train_metrics, val_metrics, summary


# ---------------------------------------------------------------------------
# Base model specs
# ---------------------------------------------------------------------------

def _base_specs(seed: int, profile: str) -> dict:
    quick = profile == "quick"
    return {
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=seed, n_jobs=1 if quick else -1),
            "grid": _prefix_param_grid({
                "n_estimators": [20] if quick else [100, 200],
                "max_depth": [4] if quick else [6, 10, None],
                "min_samples_leaf": [1] if quick else [1, 2],
            }),
            "random": _prefix_param_grid({
                "n_estimators": [20, 40, 60] if quick else [100, 150, 200, 300],
                "max_depth": [3, 5, 8, None] if quick else [4, 6, 10, 14, None],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", None],
            }),
        },
        "XGBoost": {
            "estimator": XGBRegressor(random_state=seed, n_jobs=1 if quick else -1, verbosity=0),
            "grid": _prefix_param_grid({
                "n_estimators": [50] if quick else [150, 250],
                "learning_rate": [0.1] if quick else [0.05, 0.1],
                "max_depth": [3] if quick else [3, 6],
            }),
            "random": _prefix_param_grid({
                "n_estimators": [50, 80] if quick else [100, 200, 350],
                "learning_rate": [0.02, 0.05, 0.1, 0.2],
                "max_depth": [2, 3, 5] if quick else [3, 5, 7, 9],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
            }),
        },
        "LightGBM": {
            "estimator": LGBMRegressor(random_state=seed, n_jobs=1 if quick else -1, verbose=-1),
            "grid": _prefix_param_grid({
                "n_estimators": [50] if quick else [150, 250],
                "learning_rate": [0.1] if quick else [0.05, 0.1],
                "max_depth": [3] if quick else [4, 8, -1],
            }),
            "random": _prefix_param_grid({
                "n_estimators": [50, 80] if quick else [100, 200, 350],
                "learning_rate": [0.02, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, -1] if quick else [4, 6, 10, -1],
                "num_leaves": [15, 31, 63],
                "subsample": [0.7, 0.85, 1.0],
            }),
        },
        "CatBoost": {
            "estimator": CatBoostRegressor(random_seed=seed, verbose=0, allow_writing_files=False),
            "grid": _prefix_param_grid({
                "iterations": [50] if quick else [150, 250],
                "learning_rate": [0.1] if quick else [0.05, 0.1],
                "depth": [3] if quick else [4, 6],
            }),
            "random": _prefix_param_grid({
                "iterations": [50, 80] if quick else [100, 200, 350],
                "learning_rate": [0.02, 0.05, 0.1, 0.2],
                "depth": [3, 4, 6] if quick else [4, 6, 8],
                "l2_leaf_reg": [0.1, 1.0, 3.0, 10.0],
            }),
        },
    }


def _suggest_params(trial, model_name: str, quick: bool) -> dict:
    if model_name == "RandomForest":
        return _prefix_param_grid({
            "n_estimators": trial.suggest_int("n_estimators", 20 if quick else 80, 70 if quick else 350),
            "max_depth": trial.suggest_int("max_depth", 2, 8 if quick else 18),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4 if quick else 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", None]),
        })
    if model_name == "XGBoost":
        return _prefix_param_grid({
            "n_estimators": trial.suggest_int("n_estimators", 40 if quick else 100, 90 if quick else 450),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 5 if quick else 10),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        })
    if model_name == "LightGBM":
        return _prefix_param_grid({
            "n_estimators": trial.suggest_int("n_estimators", 40 if quick else 100, 90 if quick else 450),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6 if quick else 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63 if quick else 150),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        })
    if model_name == "CatBoost":
        return _prefix_param_grid({
            "iterations": trial.suggest_int("iterations", 40 if quick else 100, 90 if quick else 450),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 5 if quick else 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        })
    raise ValueError(f"No Optuna space configured for {model_name}")


# ---------------------------------------------------------------------------
# CV search (Grid / Random) using GroupKFold
# ---------------------------------------------------------------------------

def _run_cv_search(
    method: str,
    model_name: str,
    estimator,
    param_space: dict,
    X_train,
    y_train,
    train_groups: pd.Series,
    X_val,
    y_val,
    dataset_context: dict,
    artifacts_dir: str,
    feature_names: list,
    seed: int,
    n_trials: int,
    n_splits: int,
):
    """Run GridSearchCV or RandomizedSearchCV with GroupKFold.

    The test set is not passed and is never used here.
    """
    _check_forbidden_features(feature_names)
    pipeline = _build_pipeline(estimator)
    group_cv = GroupKFold(n_splits=n_splits)

    # Validate fold integrity before fitting
    validate_group_folds(group_cv, X_train, y_train, train_groups)
    LOGGER.info(
        "[%s/%s] Group-fold validation passed — 0 engine overlap across %d folds",
        method, model_name, n_splits,
    )

    search_kwargs = {
        "estimator": pipeline,
        "scoring": "neg_root_mean_squared_error",
        "cv": group_cv,
        "return_train_score": True,
        "n_jobs": 1,
        "refit": True,
    }
    if method == "grid":
        search_kwargs["param_grid"] = param_space
        search_cls = GridSearchCV
    else:
        search_kwargs["param_distributions"] = param_space
        search_kwargs["n_iter"] = min(n_trials, _space_size(param_space))
        search_kwargs["random_state"] = seed
        search_cls = RandomizedSearchCV

    parent_started = time.perf_counter()
    with mlflow.start_run(
        run_name=f"Tuning_{method}_{model_name}_{dataset_context.get('dataset_id')}"
    ):
        mlflow.set_tag("task_type", "hyperparameter_tuning")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("tuning_method", method)
        mlflow.set_tag("cv_strategy", "GroupKFold")
        mlflow.set_tag("test_used_during_tuning", "false")
        mlflow.set_tag("leakage_audit_passed", "true")
        mlflow.set_tag("group_split", "true")
        log_params(dataset_context, prefix="context_")
        log_params({"n_splits": n_splits, "group_column": "unit_number"}, prefix="cv_")

        search = search_cls(**search_kwargs)
        search.fit(X_train, y_train, groups=train_groups)

        train_unit_count = int(train_groups.nunique())
        mlflow.log_metric("train_engine_count", train_unit_count)
        mlflow.log_metric("group_overlap_count", 0)
        mlflow.log_metric("best_cv_score", -float(search.best_score_))

        trial_run_ids = []
        best_index = int(search.best_index_)
        for trial_number, params in enumerate(search.cv_results_["params"]):
            # Extract model-level params (without 'model__' prefix) for logging
            raw_params = {k.replace("model__", ""): v for k, v in params.items()}
            # Only cv metrics — no test metrics
            metrics = {
                "cv_train_score": -float(search.cv_results_["mean_train_score"][trial_number]),
                "cv_validation_score": -float(search.cv_results_["mean_test_score"][trial_number]),
                "cv_validation_std": float(search.cv_results_["std_test_score"][trial_number]),
                "best_trial_flag": 1.0 if trial_number == best_index else 0.0,
            }
            run_id = log_tuning_trial(
                trial_number,
                raw_params,
                metrics,
                model_name,
                dataset_context,
                method,
                best_trial=trial_number == best_index,
            )
            trial_run_ids.append(run_id)

        mark_best_trial(trial_run_ids[best_index])

        best_params = search.best_params_
        raw_best_params = {k.replace("model__", ""): v for k, v in best_params.items()}
        best_model = search.best_estimator_

        # Evaluate on VALIDATION only — no test set
        val_pred = best_model.predict(X_val)
        n_features = X_train.shape[1]
        val_metrics = compute_regression_metrics(y_val, val_pred, n_features=n_features)

        # Final train metrics for gap reporting
        train_pred = best_model.predict(X_train)
        train_metrics = compute_regression_metrics(y_train, train_pred, n_features=n_features)

        log_params(raw_best_params, prefix="best_hp_")
        log_metrics(train_metrics, prefix="best_train_")
        log_metrics(val_metrics, prefix="best_validation_")
        mlflow.log_metric("best_train_score", train_metrics["RMSE"])
        mlflow.log_metric("best_validation_score", val_metrics["RMSE"])
        mlflow.log_metric("generalization_gap", val_metrics["RMSE"] - train_metrics["RMSE"])

        # Save validation predictions artifact
        safe_name = f"{method}_{model_name}_{dataset_context.get('dataset_id')}"
        val_pred_path = os.path.join(artifacts_dir, f"{safe_name}_val_predictions.csv")
        save_predictions(val_pred_path, y_val, val_pred)

        # Feature importance on the underlying model inside the Pipeline
        inner_model = best_model.named_steps.get("model")
        importance_path = plot_feature_importance(
            inner_model,
            feature_names,
            os.path.join(artifacts_dir, f"{safe_name}_feature_importance.png"),
        )

        for path in [val_pred_path, importance_path]:
            if path and os.path.exists(path):
                mlflow.log_artifact(path)

        mlflow.sklearn.log_model(best_model, artifact_path="model", serialization_format="cloudpickle")
        runtime = time.perf_counter() - parent_started

        return {
            "Model name": f"{model_name}_{method}",
            "Best hyperparameters": json.dumps(raw_best_params, sort_keys=True),
            "Train score": train_metrics["RMSE"],
            "Validation score": val_metrics["RMSE"],
            "Test score": None,  # Not evaluated during tuning
            "Main metric": dataset_context.get("main_metric", "RMSE"),
            "Runtime": runtime,
            "MLflow run ID": mlflow.active_run().info.run_id,
            "Model artifact path": mlflow.get_artifact_uri("model"),
            "Task": "RUL Regression",
            "Estimator": best_model,
            "Best params dict": raw_best_params,
            "Val metrics dict": val_metrics,
        }


# ---------------------------------------------------------------------------
# Optuna search using GroupKFold folds for validation
# ---------------------------------------------------------------------------

def _run_optuna(
    model_name: str,
    estimator,
    X_train,
    y_train,
    train_groups: pd.Series,
    X_val,
    y_val,
    dataset_context: dict,
    artifacts_dir: str,
    feature_names: list,
    seed: int,
    n_trials: int,
    profile: str,
    n_splits: int,
):
    """Run Optuna tuning, evaluating candidates on GroupKFold cross-validation.

    No test data is ever passed or used here.
    """
    _check_forbidden_features(feature_names)
    quick = profile == "quick"
    pipeline = _build_pipeline(estimator)
    group_cv = GroupKFold(n_splits=n_splits)

    parent_started = time.perf_counter()
    with mlflow.start_run(
        run_name=f"Tuning_optuna_{model_name}_{dataset_context.get('dataset_id')}"
    ):
        mlflow.set_tag("task_type", "hyperparameter_tuning")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("tuning_method", "optuna")
        mlflow.set_tag("cv_strategy", "GroupKFold")
        mlflow.set_tag("test_used_during_tuning", "false")
        mlflow.set_tag("leakage_audit_passed", "true")
        log_params(dataset_context, prefix="context_")
        log_params({"n_splits": n_splits, "group_column": "unit_number"}, prefix="cv_")

        trial_run_ids = {}

        def objective(trial):
            params = _suggest_params(trial, model_name, quick)
            trial_pipeline = clone(pipeline).set_params(**params)

            # Evaluate via GroupKFold CV — no test set involved
            fold_val_rmses = []
            for train_idx, val_idx in group_cv.split(X_train, y_train, train_groups):
                X_fold_tr = X_train[train_idx]
                y_fold_tr = y_train[train_idx]
                X_fold_val = X_train[val_idx]
                y_fold_val = y_train[val_idx]

                trial_pipeline.fit(X_fold_tr, y_fold_tr)
                fold_pred = trial_pipeline.predict(X_fold_val)
                fold_metrics = compute_regression_metrics(y_fold_val, fold_pred, n_features=X_fold_tr.shape[1])
                fold_val_rmses.append(fold_metrics["RMSE"])

            mean_val_rmse = float(np.mean(fold_val_rmses))
            std_val_rmse = float(np.std(fold_val_rmses))

            raw_params = {k.replace("model__", ""): v for k, v in params.items()}
            metrics = {
                "cv_validation_score": mean_val_rmse,
                "cv_validation_std": std_val_rmse,
                "validation_score": mean_val_rmse,
            }
            run_id = log_tuning_trial(
                trial.number,
                raw_params,
                metrics,
                model_name,
                dataset_context,
                "optuna",
                best_trial=False,
            )
            trial_run_ids[trial.number] = run_id
            return mean_val_rmse

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_trial_number = study.best_trial.number
        mark_best_trial(trial_run_ids.get(best_trial_number))

        best_params = study.best_params
        best_pipeline_params = _prefix_param_grid(best_params)

        # Refit best model on full training data
        best_model = clone(pipeline).set_params(**best_pipeline_params)
        best_model.fit(X_train, y_train)

        # Evaluate on validation only
        val_pred = best_model.predict(X_val)
        n_features = X_train.shape[1]
        val_metrics = compute_regression_metrics(y_val, val_pred, n_features=n_features)
        train_pred = best_model.predict(X_train)
        train_metrics = compute_regression_metrics(y_train, train_pred, n_features=n_features)

        log_params(best_params, prefix="best_hp_")
        log_metrics(train_metrics, prefix="best_train_")
        log_metrics(val_metrics, prefix="best_validation_")
        mlflow.log_metric("best_train_score", train_metrics["RMSE"])
        mlflow.log_metric("best_validation_score", val_metrics["RMSE"])
        mlflow.log_metric("best_cv_score", study.best_value)
        mlflow.log_metric("group_overlap_count", 0)
        mlflow.log_metric("train_engine_count", int(train_groups.nunique()))
        mlflow.log_metric("generalization_gap", val_metrics["RMSE"] - train_metrics["RMSE"])

        safe_name = f"optuna_{model_name}_{dataset_context.get('dataset_id')}"
        val_pred_path = os.path.join(artifacts_dir, f"{safe_name}_val_predictions.csv")
        save_predictions(val_pred_path, y_val, val_pred)
        inner_model = best_model.named_steps.get("model")
        importance_path = plot_feature_importance(
            inner_model,
            feature_names,
            os.path.join(artifacts_dir, f"{safe_name}_feature_importance.png"),
        )
        for path in [val_pred_path, importance_path]:
            if path and os.path.exists(path):
                mlflow.log_artifact(path)

        mlflow.sklearn.log_model(best_model, artifact_path="model", serialization_format="cloudpickle")
        runtime = time.perf_counter() - parent_started

        return {
            "Model name": f"{model_name}_optuna",
            "Best hyperparameters": json.dumps(best_params, sort_keys=True),
            "Train score": train_metrics["RMSE"],
            "Validation score": val_metrics["RMSE"],
            "Test score": None,
            "Main metric": dataset_context.get("main_metric", "RMSE"),
            "Runtime": runtime,
            "MLflow run ID": mlflow.active_run().info.run_id,
            "Model artifact path": mlflow.get_artifact_uri("model"),
            "Task": "RUL Regression",
            "Estimator": best_model,
            "Best params dict": best_params,
            "Val metrics dict": val_metrics,
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_all_tuning(
    X_train,
    y_train,
    train_groups: pd.Series,
    X_val,
    y_val,
    dataset_context: dict,
    artifacts_dir: str,
    feature_names: list,
    model_names: Optional[Iterable[str]] = None,
    methods: Iterable[str] = ("grid", "random", "optuna"),
    n_trials: int = 10,
    n_splits: int = 3,
    seed: int = 42,
    profile: str = "full",
) -> list:
    """Run GridSearchCV, RandomizedSearchCV, and Optuna for suitable sklearn RUL models.

    IMPORTANT: The test set is NOT accepted as an argument here. All model selection
    is done using GroupKFold cross-validation on training data only.

    Args:
        X_train: Training features (unscaled; scaling happens inside each Pipeline fold).
        y_train: Training RUL targets.
        train_groups: Series of unit_number values aligned with X_train (used for GroupKFold).
        X_val: Validation features (held-out engine units).
        y_val: Validation RUL targets.
        dataset_context: Context dict logged to MLflow.
        artifacts_dir: Directory to save prediction CSVs and feature importance plots.
        feature_names: List of feature column names.
        model_names: Subset of models to tune. Defaults to all available.
        methods: Tuning methods to run ('grid', 'random', 'optuna').
        n_trials: Number of trials for Optuna / RandomizedSearchCV.
        n_splits: Number of GroupKFold folds.
        seed: Random seed for reproducibility.
        profile: 'quick', 'standard', or 'full' — controls search space size.

    Returns:
        List of result dicts with val scores and MLflow run IDs (no test scores).
    """
    _check_forbidden_features(feature_names)
    set_seed(seed)
    specs = _base_specs(seed=seed, profile=profile)
    selected = list(model_names or specs.keys())
    selected = [name for name in selected if name in specs]
    selected_methods = [method.lower() for method in methods]

    # Coerce groups to pandas Series
    if not isinstance(train_groups, pd.Series):
        train_groups = pd.Series(train_groups, name="unit_number")

    results = []
    for model_name in selected:
        spec = specs[model_name]
        estimator = spec["estimator"]
        if "grid" in selected_methods:
            results.append(
                _run_cv_search(
                    "grid", model_name, estimator, spec["grid"],
                    X_train, y_train, train_groups,
                    X_val, y_val,
                    dataset_context, artifacts_dir, feature_names,
                    seed, n_trials, n_splits,
                )
            )
        if "random" in selected_methods:
            results.append(
                _run_cv_search(
                    "random", model_name, estimator, spec["random"],
                    X_train, y_train, train_groups,
                    X_val, y_val,
                    dataset_context, artifacts_dir, feature_names,
                    seed, n_trials, n_splits,
                )
            )
        if "optuna" in selected_methods:
            results.append(
                _run_optuna(
                    model_name, estimator,
                    X_train, y_train, train_groups,
                    X_val, y_val,
                    dataset_context, artifacts_dir, feature_names,
                    seed, n_trials, profile, n_splits,
                )
            )

    return results
