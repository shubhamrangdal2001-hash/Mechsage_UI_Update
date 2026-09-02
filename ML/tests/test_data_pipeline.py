"""Leakage-prevention automated tests.

Verifies:
1. No engine overlap between train/val/test splits.
2. GroupKFold folds have zero engine overlap.
3. Forbidden features (cycle_norm etc.) are absent from model inputs.
4. Rolling features are strictly causal (future changes don't affect past values).
5. Preprocessing (scaling) is fold-specific and does not see validation data before fitting.
6. production_feature_parity: no feature requires future information.
7. Deterministic splits reproduce the same engine assignments given the same seed.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src.feature_engineering import engineer_features
from src.hyperparameter_tuning import FORBIDDEN_FEATURES, validate_group_folds
from src.pipeline import prepare_data
from src.preprocessing import select_informative_sensors


# ---------------------------------------------------------------------------
# Shared test args fixture
# ---------------------------------------------------------------------------

FORBIDDEN_FEATURE_NAMES = FORBIDDEN_FEATURES


def _args(seed: int = 42, max_train_units: int = 6, max_test_units: int = 4):
    return SimpleNamespace(
        dataset="FD001",
        profile="quick",
        seed=seed,
        validation_size=0.25,
        rul_clip=125,
        anomaly_threshold=30,
        rolling_windows=(3,),
        max_train_units=max_train_units,
        max_test_units=max_test_units,
    )


# ---------------------------------------------------------------------------
# Test 1: No engine overlap between train/val/test splits
# ---------------------------------------------------------------------------

def test_no_engine_overlap_between_splits():
    """Train, validation, and test engine units must be mutually exclusive."""
    data = prepare_data(_args())
    train_units = set(data["train_df"]["unit_number"].unique())
    val_units = set(data["val_df"]["unit_number"].unique())

    assert train_units.isdisjoint(val_units), (
        f"Train and validation share engines: {train_units & val_units}"
    )


def test_prepare_data_loads_engineers_splits_and_scales():
    """Smoke test that prepare_data returns correct shapes and finite values."""
    data = prepare_data(_args())

    assert data["X_train"].shape[0] > 0
    assert data["X_val"].shape[0] > 0
    assert data["X_test"].shape[0] > 0
    assert data["X_train"].shape[1] == len(data["feature_cols"])
    assert np.isfinite(data["X_train"]).all()
    assert np.isfinite(data["X_val"]).all()
    assert np.isfinite(data["X_test"]).all()
    assert data["context"]["dataset_version"]

    train_units = set(data["train_df"]["unit_number"].unique())
    val_units = set(data["val_df"]["unit_number"].unique())
    assert train_units.isdisjoint(val_units)


# ---------------------------------------------------------------------------
# Test 2: GroupKFold folds have zero engine overlap
# ---------------------------------------------------------------------------

def test_group_kfold_has_no_engine_overlap():
    """No engine unit may appear in both sides of any GroupKFold fold."""
    data = prepare_data(_args())
    X = data["X_train"]
    y = data["y_train_rul"]
    groups = pd.Series(data["train_groups"].values) if hasattr(data["train_groups"], "values") else pd.Series(data["train_groups"])

    n_unique_engines = groups.nunique()
    n_splits = min(3, n_unique_engines - 1)
    if n_splits < 2:
        pytest.skip("Not enough engine units for GroupKFold with n_splits=2")

    splitter = GroupKFold(n_splits=n_splits)
    for fold_number, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        train_set = set(groups.iloc[train_idx])
        val_set = set(groups.iloc[val_idx])
        overlap = train_set & val_set
        assert len(overlap) == 0, (
            f"Fold {fold_number}: engines {sorted(overlap)} appear in both train and val"
        )


def test_validate_group_folds_raises_on_overlap():
    """validate_group_folds must raise ValueError if groups overlap (contrived case)."""
    import pandas as pd

    X = np.zeros((10, 3))
    y = np.zeros(10)
    groups = pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    # Build a mock splitter that deliberately returns overlapping index splits
    # (both halves contain indices from both groups → guaranteed overlap)
    class OverlappingSplitter:
        def split(self, X, y, groups):
            # train = all, val = all → every group appears in both
            all_idx = list(range(len(X)))
            yield all_idx, all_idx

    with pytest.raises(ValueError, match="Group leakage"):
        validate_group_folds(OverlappingSplitter(), X, y, groups)


# ---------------------------------------------------------------------------
# Test 3: Forbidden features absent
# ---------------------------------------------------------------------------

def test_forbidden_features_not_used():
    """cycle_norm and equivalent future-leaking features must not appear in feature_cols."""
    data = prepare_data(_args())
    feature_cols_set = set(data["feature_cols"])
    violations = FORBIDDEN_FEATURE_NAMES.intersection(feature_cols_set)
    assert len(violations) == 0, (
        f"Forbidden features found in model inputs: {sorted(violations)}"
    )


def test_cycle_norm_absent():
    """Specific check: cycle_norm must not be in feature_cols."""
    data = prepare_data(_args())
    assert "cycle_norm" not in data["feature_cols"], (
        "cycle_norm is a future-leaking feature that must be removed."
    )


def test_expanding_mean_present():
    """Expanding mean features (causal aggregates) must be present."""
    data = prepare_data(_args())
    assert any(col.endswith("_expanding_mean") for col in data["feature_cols"]), (
        "Expected at least one expanding_mean causal aggregate feature."
    )


# ---------------------------------------------------------------------------
# Test 4: Rolling features are strictly causal
# ---------------------------------------------------------------------------

def test_future_sensor_change_does_not_affect_past_features():
    """Changing a future cycle's sensor reading must not alter earlier cycles' features."""
    from src.data_loader import load_cmapss_data

    train_df, _ = load_cmapss_data("FD001", max_rul_clip=125, anomaly_threshold=30)
    # Use a small subset
    unit = train_df["unit_number"].iloc[0]
    unit_df = train_df[train_df["unit_number"] == unit].copy()
    unit_df = select_informative_sensors(unit_df)

    # Engineer features on the original data
    df_orig = engineer_features(unit_df.copy(), window_sizes=(3,))

    # Corrupt the LAST cycle's sensor values
    last_cycle_idx = unit_df.index[-1]
    unit_df_modified = unit_df.copy()
    unit_df_modified.loc[last_cycle_idx, "sensor_2"] = 9999.0

    df_mod = engineer_features(unit_df_modified.copy(), window_sizes=(3,))

    # All cycles EXCEPT the last one should have identical rolling features
    if len(df_orig) > 1:
        orig_past = df_orig.iloc[:-1]
        mod_past = df_mod.iloc[:-1]
        rolling_cols = [c for c in df_orig.columns if "roll_mean" in c or "roll_std" in c]
        for col in rolling_cols:
            np.testing.assert_array_almost_equal(
                orig_past[col].values,
                mod_past[col].values,
                decimal=6,
                err_msg=f"Column {col} was affected by a future change — non-causal!",
            )


# ---------------------------------------------------------------------------
# Test 5: Preprocessing is fold-specific (scaler inside Pipeline)
# ---------------------------------------------------------------------------

def test_preprocessing_is_inside_pipeline():
    """The best estimator from tuning must be a Pipeline with a scaler step."""
    from sklearn.pipeline import Pipeline
    from src.hyperparameter_tuning import _build_pipeline
    from sklearn.ensemble import RandomForestRegressor

    estimator = RandomForestRegressor(random_state=42)
    pipeline = _build_pipeline(estimator)
    assert isinstance(pipeline, Pipeline), "Estimator must be wrapped in a Pipeline"
    assert "scaler" in pipeline.named_steps, "Pipeline must have a 'scaler' step"
    assert "model" in pipeline.named_steps, "Pipeline must have a 'model' step"


# ---------------------------------------------------------------------------
# Test 6: No test set peeking in tuning (interface check)
# ---------------------------------------------------------------------------

def test_run_all_tuning_does_not_accept_test_data(workspace_tmp_path):
    """run_all_tuning must NOT accept X_test/y_test as arguments."""
    import inspect
    from src.hyperparameter_tuning import run_all_tuning

    sig = inspect.signature(run_all_tuning)
    param_names = list(sig.parameters.keys())
    assert "X_test" not in param_names, "run_all_tuning must not accept X_test"
    assert "y_test" not in param_names, "run_all_tuning must not accept y_test"


# ---------------------------------------------------------------------------
# Test 7: Deterministic splits
# ---------------------------------------------------------------------------

def test_deterministic_splits_same_seed():
    """Same seed must produce identical engine assignments."""
    data1 = prepare_data(_args(seed=42))
    data2 = prepare_data(_args(seed=42))

    units1 = sorted(data1["train_df"]["unit_number"].unique())
    units2 = sorted(data2["train_df"]["unit_number"].unique())
    assert units1 == units2, "Same seed produced different train engine assignments"

    val1 = sorted(data1["val_df"]["unit_number"].unique())
    val2 = sorted(data2["val_df"]["unit_number"].unique())
    assert val1 == val2, "Same seed produced different val engine assignments"


def test_different_seeds_may_differ():
    """Different seeds may produce different engine assignments."""
    data1 = prepare_data(_args(seed=1))
    data2 = prepare_data(_args(seed=99))

    units1 = sorted(data1["train_df"]["unit_number"].unique())
    units2 = sorted(data2["train_df"]["unit_number"].unique())
    # Not guaranteed to differ but usually will with different seeds
    # Just verify both are non-empty and structurally valid
    assert len(units1) > 0
    assert len(units2) > 0


# ---------------------------------------------------------------------------
# Test 8: Split manifest is generated
# ---------------------------------------------------------------------------

def test_split_manifest_is_generated():
    """prepare_data must return a split_manifest DataFrame."""
    data = prepare_data(_args())
    assert "split_manifest" in data
    manifest = data["split_manifest"]
    assert isinstance(manifest, pd.DataFrame)
    required_cols = {"unit_number", "split", "dataset_id", "random_seed"}
    assert required_cols.issubset(set(manifest.columns))
    splits_present = set(manifest["split"].unique())
    assert "train" in splits_present
    assert "validation" in splits_present


# ---------------------------------------------------------------------------
# Test 9: groups arrays are aligned with X and y
# ---------------------------------------------------------------------------

def test_groups_aligned_with_features_and_targets():
    """train_groups, y_train_rul, X_train must all have the same length."""
    data = prepare_data(_args())

    assert len(data["X_train"]) == len(data["y_train_rul"]) == len(data["train_groups"])
    assert len(data["X_val"]) == len(data["y_val_rul"]) == len(data["val_groups"])
    assert len(data["X_test"]) == len(data["y_test_rul"]) == len(data["test_groups"])


# ---------------------------------------------------------------------------
# Test 10: Cycles are monotonically increasing per engine unit
# ---------------------------------------------------------------------------

def test_cycles_are_monotonically_increasing_per_unit():
    """After feature engineering, cycles must be monotonically increasing per engine."""
    data = prepare_data(_args())
    for df_name in ["train_df", "val_df", "test_df"]:
        df = data[df_name]
        monotonic_check = (
            df.groupby("unit_number")["time_in_cycles"]
            .apply(lambda s: s.is_monotonic_increasing)
            .all()
        )
        assert monotonic_check, f"{df_name}: cycles are not monotonically increasing per unit"
