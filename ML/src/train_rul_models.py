"""Model definitions and training helpers for RUL regression."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from src.evaluate import compute_rul_metrics
from src.utils import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISED ML MODELS FOR RUL REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def train_sklearn_rul_models(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    seed=42,
    models=None,
):
    """
    Train scikit-learn / tree-based RUL regression models.

    All models are supervised — they use ground-truth RUL labels derived
    from the C-MAPSS dataset. Model selection is done on the VALIDATION set
    only; test metrics are reported for final evaluation only.

    Args:
        X_train / y_train: Training features and targets.
        X_val / y_val:     Validation features and targets (used for model selection).
        X_test / y_test:   Test features and targets (reported only, NOT used for selection).
        seed:              Random seed for reproducibility.
        models:            Optional list of model names to train. If None, trains all.
                           Valid names: LinearRegression, RandomForest, XGBoost, LightGBM, CatBoost

    Returns:
        dict: { model_name: (trained_model, params_dict, val_metrics_dict, test_metrics_dict, y_pred_test) }
    """
    set_seed(seed)
    all_models = {
        "LinearRegression": LinearRegression(),
        "RandomForest":     RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1),
        "XGBoost":          XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                                         subsample=0.8, colsample_bytree=0.8,
                                         random_state=seed, n_jobs=-1, verbosity=0),
        "LightGBM":         LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                                          subsample=0.8, colsample_bytree=0.8,
                                          random_state=seed, n_jobs=-1, verbose=-1),
        "CatBoost":         CatBoostRegressor(iterations=200, learning_rate=0.05, depth=6,
                                              random_seed=seed, verbose=0),
    }
    # Filter by requested model list
    selected = {k: v for k, v in all_models.items() if models is None or k in models}

    results = {}
    for name, model in selected.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred_val  = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        val_metrics  = compute_rul_metrics(y_val,  y_pred_val)   # ← used for selection
        test_metrics = compute_rul_metrics(y_test, y_pred_test)  # ← reported only
        params = model.get_params() if hasattr(model, "get_params") else {}
        results[name] = (model, params, val_metrics, test_metrics, y_pred_test)
        print(f"    Val RMSE={val_metrics['RMSE']:.2f} | Test RMSE={test_metrics['RMSE']:.2f} | R²={test_metrics['R2']:.4f}")
    return results
