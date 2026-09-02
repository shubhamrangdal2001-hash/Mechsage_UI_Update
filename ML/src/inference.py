"""Inference utilities for locally saved MechSage models.

Example:
    >>> from src.inference import load_model, predict
    >>> model = load_model("models/best_rul_model_FD001.joblib")
    >>> predictions = predict(model, feature_matrix)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.exceptions import ModelInferenceError


def load_model(model_path: str) -> Any:
    """Load a joblib model from disk.

    Args:
        model_path: Path to a ``.joblib`` model file.

    Returns:
        Deserialized model object.

    Raises:
        ModelInferenceError: If the model path is invalid or cannot be read.
    """
    path = Path(model_path)
    if not path.exists():
        raise ModelInferenceError(f"Model file does not exist: {model_path}")
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - convert library error to domain error.
        raise ModelInferenceError(f"Unable to load model from {model_path}") from exc


def predict(model: Any, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Generate predictions from a loaded model.

    Args:
        model: Fitted estimator implementing ``predict``.
        features: Feature matrix as a DataFrame or numpy array.

    Returns:
        Prediction array.
    """
    if not hasattr(model, "predict"):
        raise ModelInferenceError("Model object does not implement predict().")
    try:
        return np.asarray(model.predict(features))
    except Exception as exc:  # noqa: BLE001
        raise ModelInferenceError("Prediction failed.") from exc


def predict_from_csv(model_path: str, input_csv: str, output_csv: str) -> str:
    """Load a model, score a CSV file, and write predictions.

    Args:
        model_path: Path to a saved joblib model.
        input_csv: CSV containing feature columns.
        output_csv: Destination CSV path.

    Returns:
        The output CSV path.
    """
    model = load_model(model_path)
    frame = pd.read_csv(input_csv)
    frame["prediction"] = predict(model, frame)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return str(output_path)
