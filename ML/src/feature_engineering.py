"""Feature engineering utilities for C-MAPSS time-series data."""

import pandas as pd
import numpy as np
from src.preprocessing import INFORMATIVE_SENSORS, OP_SETTINGS, NON_FEATURE_COLS


def add_rolling_features(df: pd.DataFrame, window_sizes=(5, 10, 20)) -> pd.DataFrame:
    """
    Compute rolling mean and rolling std for each informative sensor,
    per engine unit to avoid cross-unit leakage.

    Args:
        df: DataFrame with unit_number, time_in_cycles, and sensor columns
        window_sizes: Tuple of rolling window sizes in cycles

    Returns:
        DataFrame with additional rolling feature columns.
    """
    df = df.copy()
    sensor_cols = [c for c in INFORMATIVE_SENSORS if c in df.columns]
    
    for sensor in sensor_cols:
        for w in window_sizes:
            df[f"{sensor}_roll_mean_{w}"] = (
                df.groupby("unit_number")[sensor]
                .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
            )
            df[f"{sensor}_roll_std_{w}"] = (
                df.groupby("unit_number")[sensor]
                .transform(lambda x: x.rolling(window=w, min_periods=1).std().fillna(0))
            )
    return df


def add_degradation_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute degradation delta: current sensor value minus its rolling mean (window=10).
    A positive delta on a rising sensor indicates accelerating degradation.
    """
    df = df.copy()
    sensor_cols = [c for c in INFORMATIVE_SENSORS if c in df.columns]
    
    for sensor in sensor_cols:
        roll_mean_col = f"{sensor}_roll_mean_10"
        if roll_mean_col in df.columns:
            df[f"{sensor}_delta"] = df[sensor] - df[roll_mean_col]
        else:
            df[f"{sensor}_delta"] = (
                df.groupby("unit_number")[sensor]
                .transform(lambda x: x - x.rolling(window=10, min_periods=1).mean())
            )
    return df


def add_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add anomaly-specific degradation features based on SHAP analysis results.

    Top anomaly-contributing sensors (from FD004 SHAP): 14, 13, 11.
    These features capture non-linear degradation patterns that linear rolling
    stats miss:

    - trend_slope_N: linear regression slope over last N cycles (degradation rate)
    - delta_delta (2nd derivative): rate-of-change of the delta feature
    - expanding_var: cumulative variance (increases as system degrades)

    All computed per engine unit to avoid cross-unit data leakage.
    """
    df = df.copy()
    # Top sensors identified from SHAP analysis
    key_sensors = [s for s in ["sensor_13", "sensor_11", "sensor_14", "sensor_8"] if s in df.columns]

    for sensor in key_sensors:
        # --- Expanding variance: grows as sensor becomes noisier near failure ---
        df[f"{sensor}_expanding_var"] = (
            df.groupby("unit_number")[sensor]
            .transform(lambda x: x.expanding(min_periods=1).var().fillna(0))
        )

        # --- Second derivative (delta of delta): acceleration of degradation ---
        delta_col = f"{sensor}_delta"
        if delta_col in df.columns:
            df[f"{sensor}_delta2"] = (
                df.groupby("unit_number")[delta_col]
                .transform(lambda x: x.diff().fillna(0))
            )

        # --- Trend slope over last 10 cycles: linear rate of change ---
        def _slope(x: "pd.Series") -> "pd.Series":
            result = np.zeros(len(x))
            vals = x.values
            for i in range(len(vals)):
                window = vals[max(0, i - 9): i + 1]
                n = len(window)
                if n < 2:
                    result[i] = 0.0
                else:
                    t = np.arange(n)
                    # Least-squares slope: (n*sum(t*y) - sum(t)*sum(y)) / (n*sum(t^2) - sum(t)^2)
                    num = n * np.dot(t, window) - t.sum() * window.sum()
                    den = n * np.dot(t, t) - t.sum() ** 2
                    result[i] = num / den if den != 0 else 0.0
            return pd.Series(result, index=x.index)

        df[f"{sensor}_trend_slope"] = (
            df.groupby("unit_number")[sensor].transform(_slope)
        )

    return df


def engineer_features(df: pd.DataFrame, window_sizes=(5, 10, 20)) -> pd.DataFrame:
    """
    Master feature engineering function. Applies rolling stats, degradation deltas,
    expanding mean features, and anomaly-specific degradation features in a
    strictly causal manner (no future data leakage).
    """
    df = df.copy()
    # Sort first to prevent any rolling window misalignment
    df = df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)

    # Assert monotonic cycles per unit
    cycle_is_monotonic = (
        df.groupby("unit_number")["time_in_cycles"]
        .apply(lambda values: values.is_monotonic_increasing)
        .all()
    )
    assert cycle_is_monotonic, "Cycles are not monotonic per engine unit"

    df = add_rolling_features(df, window_sizes=window_sizes)
    df = add_degradation_trends(df)

    # Add expanding mean for informative sensors (causal aggregates)
    sensor_cols = [c for c in INFORMATIVE_SENSORS if c in df.columns]
    for sensor in sensor_cols:
        df[f"{sensor}_expanding_mean"] = (
            df.groupby("unit_number")[sensor]
            .transform(lambda series: series.expanding(min_periods=1).mean())
        )

    # Add anomaly-specific features (expanding variance, 2nd derivative, trend slope)
    df = add_anomaly_features(df)

    return df




def build_sequence_windows(df: pd.DataFrame, feature_cols: list, sequence_length: int = 30):
    """
    Construct 3D sliding window tensors suitable for LSTM / GRU / 1D CNN / Transformer models.
    Each sample is a sequence of `sequence_length` consecutive cycles from one engine unit.
    The label is the RUL at the last cycle of the window.

    Args:
        df: Preprocessed DataFrame with feature and RUL columns
        feature_cols: List of feature column names to include in each window
        sequence_length: Number of cycles per sequence window

    Returns:
        X: numpy array of shape (N, sequence_length, num_features)
        y_rul: numpy array of shape (N,) — RUL targets
        y_anomaly: numpy array of shape (N,) — anomaly targets
    """
    X_seqs, y_rul_list, y_anomaly_list = [], [], []

    for unit_id, group in df.groupby("unit_number"):
        group = group.sort_values("time_in_cycles")
        features = group[feature_cols].values
        rul_vals = group["RUL"].values
        anomaly_vals = group["anomaly"].values
        
        n_rows = len(features)
        if n_rows < sequence_length:
            # Pad short sequences with the first row values
            pad = np.tile(features[0], (sequence_length - n_rows, 1))
            features = np.vstack([pad, features])
            rul_vals = np.concatenate([np.full(sequence_length - n_rows, rul_vals[0]), rul_vals])
            anomaly_vals = np.concatenate([np.zeros(sequence_length - n_rows, dtype=int), anomaly_vals])
        
        for i in range(sequence_length, len(features) + 1):
            X_seqs.append(features[i - sequence_length:i])
            y_rul_list.append(rul_vals[i - 1])
            y_anomaly_list.append(anomaly_vals[i - 1])

    return (
        np.array(X_seqs, dtype=np.float32),
        np.array(y_rul_list, dtype=np.float32),
        np.array(y_anomaly_list, dtype=np.int32)
    )


def get_flat_features(df: pd.DataFrame, feature_cols: list):
    """
    Return flat (2D) feature matrix and targets for scikit-learn models.
    Uses only the last cycle per rolling group (no sequence windowing needed).
    
    Returns:
        X: (N, num_features) float64 numpy array
        y_rul: (N,) float64 array
        y_anomaly: (N,) int32 array
    """
    X = df[feature_cols].values.astype(np.float64)
    y_rul = df["RUL"].values.astype(np.float64)
    y_anomaly = df["anomaly"].values.astype(np.int32)
    return X, y_rul, y_anomaly
