"""Smoke test for data loading, preprocessing, and feature engineering.

Example:
    python smoke_test.py
"""

import logging
import sys


sys.path.insert(0, ".")

from src.data_loader import load_cmapss_data
from src.feature_engineering import build_sequence_windows, engineer_features, get_flat_features
from src.logging_config import configure_logging
from src.preprocessing import get_feature_columns, normalize_per_unit, select_informative_sensors
from src.utils import set_seed


LOGGER = logging.getLogger("mechsage.smoke_test")


def main() -> int:
    """Run a lightweight data/feature smoke test.

    Returns:
        Process exit code.
    """
    configure_logging()
    LOGGER.info("Starting data pipeline smoke test")
    set_seed(42)
    train_df, test_df = load_cmapss_data("FD001")
    LOGGER.info("Train shape: %s | Test shape: %s", train_df.shape, test_df.shape)
    LOGGER.info("Train RUL min=%s max=%s", train_df["RUL"].min(), train_df["RUL"].max())
    LOGGER.info("Anomaly rate: %.1f%%", train_df["anomaly"].mean() * 100)

    train_df = engineer_features(select_informative_sensors(train_df))
    test_df = engineer_features(select_informative_sensors(test_df))
    train_df, test_df, _ = normalize_per_unit(train_df, test_df)
    feature_columns = get_feature_columns(train_df)

    X_train, _, _ = get_flat_features(train_df, feature_columns)
    X_sequence, _, _ = build_sequence_windows(train_df, feature_columns, sequence_length=30)
    LOGGER.info("Feature count: %d", len(feature_columns))
    LOGGER.info("Flat X_train: %s", X_train.shape)
    LOGGER.info("Sequence X_train: %s", X_sequence.shape)
    LOGGER.info("Smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
