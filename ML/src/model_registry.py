"""Local model versioning helpers for trained model artifacts."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.mlflow_tracking import write_json


def get_git_commit() -> Optional[str]:
    """Return the current git commit hash when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def build_model_version(
    model_name: str,
    dataset_version: str,
    run_id: Optional[str] = None,
) -> dict[str, Optional[str]]:
    """Create a model version metadata payload.

    Args:
        model_name: Human-readable model name.
        dataset_version: Dataset hash/version used for training.
        run_id: Optional MLflow run ID.

    Returns:
        JSON-serializable model version dictionary.
    """
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "model_name": model_name,
        "version": f"{model_name}-{dataset_version}-{timestamp}",
        "training_timestamp_utc": timestamp,
        "dataset_version": dataset_version,
        "git_commit": get_git_commit(),
        "mlflow_run_id": run_id,
    }


def save_model_card(path: str, payload: dict[str, object]) -> str:
    """Persist model version metadata as JSON."""
    write_json(path, payload)
    return path
