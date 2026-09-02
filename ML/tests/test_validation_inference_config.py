"""Tests for production utilities: config loading, validation, and inference."""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import load_config
from src.data_validation import validate_dataframe
from src.inference import load_model, predict, predict_from_csv


def test_load_config_reads_default_yaml() -> None:
    """Default YAML config should load with expected production sections."""
    config = load_config()

    assert config.runtime.seed == 42
    assert config.paths.logs_dir == "logs"
    assert "quick" in config.models


def test_validate_dataframe_reports_missing_duplicates_and_schema() -> None:
    """Validation should surface common data quality findings."""
    frame = pd.DataFrame(
        {
            "unit_number": [1, 1, 1],
            "time_in_cycles": [1, 1, 2],
            "RUL": [10.0, 10.0, 9.0],
            "RUL_raw": [10.0, 10.0, 9.0],
            "anomaly": [0, 0, 1],
            "sensor_2": [0.1, 0.1, 100.0],
        }
    )

    report = validate_dataframe(
        frame,
        dataset_name="unit-test",
        required_columns=["unit_number", "time_in_cycles", "RUL", "RUL_raw", "anomaly"],
        target_columns=["RUL", "anomaly"],
    )

    assert report.row_count == 3
    assert report.duplicate_rows == 1
    assert report.errors == []
    assert report.outlier_counts["sensor_2"] >= 0


def test_model_loading_and_prediction_helpers(workspace_tmp_path) -> None:
    """Inference helpers should load a saved model and write predictions."""
    X = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [1.0, 2.0, 3.0]})
    y = np.array([1.0, 3.0, 5.0])
    model = LinearRegression().fit(X, y)
    model_path = workspace_tmp_path / "linear.joblib"
    input_path = workspace_tmp_path / "features.csv"
    output_path = workspace_tmp_path / "predictions.csv"
    joblib.dump(model, model_path)
    X.to_csv(input_path, index=False)

    loaded = load_model(str(model_path))
    predictions = predict(loaded, X)
    written_path = predict_from_csv(str(model_path), str(input_path), str(output_path))

    assert len(predictions) == len(X)
    assert written_path == str(output_path)
    assert output_path.exists()
    assert "prediction" in pd.read_csv(output_path).columns
