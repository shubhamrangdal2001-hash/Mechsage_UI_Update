"""
backend/services/inference_service.py
---------------------------------------
Singleton service that loads all 8 model artifacts at FastAPI startup.
Exposes a single predict() method used by both the simulation loop
and the on-demand /api/v1/infer endpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Ensure repo root is on sys.path so we can import inference.*
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.core.logging import get_logger

logger = get_logger(__name__)

DATASETS = ("FD001", "FD002", "FD003", "FD004")


class InferenceService:
    """
    Loads and owns all 8 .joblib model artifacts.
    Thread-safe for read operations (models are read-only after load).
    """

    def __init__(self) -> None:
        self._models: Dict[str, Dict[str, Any]] = {}
        self._builders: Dict[str, Any] = {}
        self._loaded = False

    async def startup(self) -> None:
        """Called during FastAPI lifespan startup. Loads all models + scalers."""
        logger.info("Loading inference models and fitting scalers...")
        try:
            from inference.predictor import load_all_models
            from inference.feature_builder import build_builders

            self._models = load_all_models()
            self._builders = build_builders(list(DATASETS))
            self._loaded = True
            logger.info(
                "Inference service ready",
                datasets=list(DATASETS),
                model_count=len(self._models) * 2,
            )
        except Exception as exc:
            logger.error("Failed to load inference models", error=str(exc))
            raise

    def is_ready(self) -> bool:
        return self._loaded

    def predict(
        self,
        dataset_id: str,
        raw_row: pd.Series,
        cycle: int,
        machine_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Run RUL + Anomaly prediction for one cycle.

        Args:
            dataset_id:  FD001–FD004
            raw_row:     pd.Series from the synthetic generator
            cycle:       Current cycle number
            machine_id:  Engine identifier string

        Returns:
            Structured payload dict (matches agent_feed schema), or None
            if the builder is still in warm-up phase.
        """
        if not self._loaded:
            raise RuntimeError("InferenceService not started — call await startup() first")

        from inference.predictor import predict_cycle
        from inference.agent_feed import build_payload

        builder = self._builders[dataset_id]
        feature_row = builder.add_cycle(raw_row)

        if feature_row is None:
            return None  # still warming up

        prediction = predict_cycle(
            dataset_id=dataset_id,
            feature_row=feature_row,
            models=self._models,
            cycle=cycle,
            machine_id=machine_id,
        )
        return build_payload(prediction, raw_row)

    def get_builder(self, dataset_id: str) -> Any:
        return self._builders.get(dataset_id)

    def get_models(self) -> Dict[str, Dict[str, Any]]:
        return self._models


# Global singleton
inference_service = InferenceService()
