"""Smoke test for the supervised LightGBM anomaly model.

Example:
    python smoke_test_lgbm.py
"""

import logging
import sys


sys.path.insert(0, ".")

from src.data_loader import load_cmapss_data
from src.feature_engineering import engineer_features, get_flat_features
from src.logging_config import configure_logging
from src.preprocessing import get_feature_columns, normalize_per_unit, select_informative_sensors
from src.train_anomaly_models import train_lightgbm_anomaly
from src.utils import set_seed


LOGGER = logging.getLogger("mechsage.smoke_test_lgbm")


def main() -> int:
    """Train LightGBM anomaly detector on FD001 as a smoke test.

    Returns:
        Process exit code.
    """
    configure_logging()
    set_seed(42)
    train_df, test_df = load_cmapss_data("FD001")
    train_df = engineer_features(select_informative_sensors(train_df))
    test_df = engineer_features(select_informative_sensors(test_df))
    train_df, test_df, _ = normalize_per_unit(train_df, test_df)
    feature_columns = get_feature_columns(train_df)

    X_train, _, y_train_anomaly = get_flat_features(train_df, feature_columns)
    X_test, _, y_test_anomaly = get_flat_features(test_df, feature_columns)

    LOGGER.info("Training LightGBM anomaly classifier")
    _, _, metrics = train_lightgbm_anomaly(X_train, y_train_anomaly, X_test, y_test_anomaly, seed=42)
    for key, value in metrics.items():
        LOGGER.info("%s: %s", key, value)
    LOGGER.info("LightGBM anomaly smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
