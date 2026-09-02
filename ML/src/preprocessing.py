"""Preprocessing helpers for sensor selection and train-only normalization."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.utils import get_project_root
import os

# 14 informative sensors identified in the project design spec
INFORMATIVE_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21"
]

OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]

NON_FEATURE_COLS = ["unit_number", "time_in_cycles", "RUL", "RUL_raw", "anomaly"]


def select_informative_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the 7 near-constant/non-informative sensors from the dataframe.
    Only retains the 14 informative sensors + operational settings + metadata columns.
    """
    keep_cols = NON_FEATURE_COLS + OP_SETTINGS + INFORMATIVE_SENSORS
    # Only keep columns that exist in the dataframe
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def normalize_per_unit(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Apply Min-Max normalization to sensor and op_setting columns.
    Scaler is fit on training data only to prevent data leakage.
    Returns (scaled_train_df, scaled_test_df, scaler).
    """
    feature_cols = [c for c in (OP_SETTINGS + INFORMATIVE_SENSORS) if c in train_df.columns]
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df, scaler


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return all numeric feature columns (sensors + op_settings + engineered features), excluding metadata."""
    exclude = set(NON_FEATURE_COLS)
    return [c for c in df.columns if c not in exclude]
