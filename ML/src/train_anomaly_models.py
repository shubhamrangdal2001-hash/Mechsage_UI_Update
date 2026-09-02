"""Supervised anomaly detection models for MechSage pipeline.

Three classifiers compete; the pipeline selects the one with the highest
validation F1 score.  All use labelled data (anomaly = RUL <= threshold).

Improvements over baseline:
  1. Regularised LightGBM params  -> fixes Train F1=1.0 / Test F1=0.59 gap
  2. RandomForest & XGBoost added -> diversity improves ensemble selection
  3. SMOTE oversampling            -> handles ~85/15 class imbalance
  4. Finer threshold sweep (0.01)  -> better decision boundary
"""

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.evaluate import compute_anomaly_metrics


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_smote(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    """
    Apply SMOTE oversampling on training data ONLY.
    Falls back to original data if imbalanced-learn is not installed or
    if the minority class has too few samples for SMOTE.
    """
    try:
        from imblearn.over_sampling import SMOTE
        n_minority = int(np.sum(y_train == 1))
        if n_minority < 6:
            return X_train, y_train   # too few samples for k_neighbors=5
        smote = SMOTE(random_state=seed, k_neighbors=min(5, n_minority - 1))
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res
    except ImportError:
        return X_train, y_train


def _best_threshold_val(y_val: np.ndarray, y_prob_val: np.ndarray) -> float:
    """
    Sweep thresholds at 0.01 step and pick the one maximising F1 on the
    VALIDATION set only — never on the test set.
    """
    from sklearn.metrics import f1_score as _f1
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.05, 0.96, 0.01):        # finer: 91 points vs 17
        y_tmp = (y_prob_val >= thresh).astype(int)
        f1 = _f1(y_val, y_tmp, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return float(best_thresh)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — LIGHTGBM (regularised to prevent overfitting)
# ─────────────────────────────────────────────────────────────────────────────

def train_lightgbm_anomaly(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
):
    """
    Regularised LightGBM binary classifier for supervised anomaly detection.

    Key changes from baseline to fix overfitting (Train=1.0, Test=0.59):
    - max_depth reduced 8 -> 5
    - num_leaves reduced 63 -> 31
    - min_child_samples=30  (min leaf size — prevents tiny pure leaves)
    - reg_alpha=0.1, reg_lambda=1.0  (L1 + L2 regularisation)
    - SMOTE applied on train only

    Threshold swept at 0.01 step on VALIDATION set only.
    """
    X_tr, y_tr = _apply_smote(X_train, y_train, seed)

    params = {
        "n_estimators":      300,
        "learning_rate":     0.05,
        "max_depth":         5,        # down from 8  — limits tree depth
        "num_leaves":        31,       # down from 63 — limits model complexity
        "min_child_samples": 30,       # NEW — prevents tiny leaf nodes
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,      # NEW — L1 regularisation
        "reg_lambda":        1.0,      # NEW — L2 regularisation
        "class_weight":      "balanced",
        "random_state":      seed,
        "n_jobs":            -1,
        "verbose":           -1,
    }
    model = LGBMClassifier(**params)
    model.fit(X_tr, y_tr)

    y_prob_val  = model.predict_proba(X_val)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    threshold = _best_threshold_val(y_val, y_prob_val)
    y_pred = (y_prob_test >= threshold).astype(int)

    params["best_threshold"]   = round(threshold, 3)
    params["threshold_source"] = "validation_f1_sweep_0.01"
    params["smote_applied"]    = True

    metrics = compute_anomaly_metrics(y_test, y_pred, y_prob=y_prob_test)
    return model, params, metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — RANDOM FOREST (diversity + natural feature bagging)
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest_anomaly(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
):
    """
    Random Forest binary classifier for supervised anomaly detection.

    Advantages over LightGBM for small/imbalanced datasets:
    - Natural bagging reduces variance / overfitting
    - class_weight='balanced_subsample' rebalances per bootstrap sample
    - max_features='sqrt' further decorrelates trees
    - SMOTE applied on train only
    """
    X_tr, y_tr = _apply_smote(X_train, y_train, seed)

    params = {
        "n_estimators":   200,
        "max_depth":      8,
        "min_samples_leaf": 10,
        "max_features":   "sqrt",
        "class_weight":   "balanced_subsample",
        "random_state":   seed,
        "n_jobs":         -1,
    }
    model = RandomForestClassifier(**params)
    model.fit(X_tr, y_tr)

    y_prob_val  = model.predict_proba(X_val)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    threshold = _best_threshold_val(y_val, y_prob_val)
    y_pred = (y_prob_test >= threshold).astype(int)

    params["best_threshold"]   = round(threshold, 3)
    params["threshold_source"] = "validation_f1_sweep_0.01"
    params["smote_applied"]    = True

    metrics = compute_anomaly_metrics(y_test, y_pred, y_prob=y_prob_test)
    return model, params, metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — XGBOOST (scale_pos_weight for imbalance)
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost_anomaly(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
):
    """
    XGBoost binary classifier for supervised anomaly detection.

    Uses scale_pos_weight = neg_count/pos_count to handle class imbalance
    natively instead of SMOTE (XGBoost handles this better internally).
    """
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    spw   = float(n_neg / max(n_pos, 1))

    params = {
        "n_estimators":     300,
        "learning_rate":    0.05,
        "max_depth":        5,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "scale_pos_weight": spw,
        "random_state":     seed,
        "n_jobs":           -1,
        "verbosity":        0,
        "eval_metric":      "logloss",
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_prob_val  = model.predict_proba(X_val)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    threshold = _best_threshold_val(y_val, y_prob_val)
    y_pred = (y_prob_test >= threshold).astype(int)

    params["best_threshold"]   = round(threshold, 3)
    params["threshold_source"] = "validation_f1_sweep_0.01"
    params["scale_pos_weight"] = round(spw, 3)

    metrics = compute_anomaly_metrics(y_test, y_pred, y_prob=y_prob_test)
    return model, params, metrics
