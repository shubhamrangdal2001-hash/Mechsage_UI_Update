"""Configuration loading utilities for the MechSage pipeline.

Example:
    >>> from src.config import load_config
    >>> config = load_config()
    >>> config.runtime.seed
    42
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from src.exceptions import ConfigurationError
from src.utils import get_project_root


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths used by the project."""

    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    models_dir: str = "models"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    mlruns_dir: str = "mlruns"


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime defaults for reproducible executions."""

    seed: int = 42
    profile: str = "quick"
    validation_size: float = 0.2
    tracking_uri: Optional[str] = None


@dataclass(frozen=True)
class DataConfig:
    """Dataset labeling and sampling configuration."""

    dataset: str = "FD001"
    rul_clip: int = 125
    anomaly_threshold: int = 30
    max_train_units: dict[str, Optional[int]] = field(default_factory=dict)
    max_test_units: dict[str, Optional[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering configuration."""

    rolling_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    informative_sensors: list[str] = field(default_factory=list)
    operational_settings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameter tuning defaults."""

    cv: int = 3
    n_trials: int = 5
    tuning_methods: list[str] = field(default_factory=lambda: ["grid", "random", "optuna"])
    tuning_models: list[str] = field(default_factory=lambda: ["RandomForest"])


@dataclass(frozen=True)
class LoggingConfig:
    """Centralized logging configuration."""

    level: str = "INFO"
    file_name: str = "pipeline.log"


@dataclass(frozen=True)
class ProjectConfig:
    """Typed project configuration loaded from YAML."""

    name: str = "MechSage NASA C-MAPSS Pipeline"
    experiment_prefix: str = "MechSage"
    paths: PathConfig = field(default_factory=PathConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _default_config_path() -> Path:
    """Return the default YAML configuration path."""
    return Path(get_project_root()) / "configs" / "default.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML config file.

    Args:
        path: Absolute or relative path to a YAML file.

    Returns:
        Parsed YAML dictionary.

    Raises:
        ConfigurationError: If the file cannot be read or parsed.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML configuration: {path}") from exc


def load_config(config_path: Optional[str] = None) -> ProjectConfig:
    """Load project configuration from YAML and environment variables.

    Args:
        config_path: Optional path to a YAML config. Defaults to
            ``configs/default.yaml`` inside the project.

    Returns:
        A typed ``ProjectConfig`` instance.
    """
    path = Path(config_path).resolve() if config_path else _default_config_path()
    raw = _read_yaml(path)
    project = raw.get("project", {})

    runtime_raw = raw.get("runtime", {})
    env_seed = os.getenv("MECHSAGE_RANDOM_SEED")
    env_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_seed:
        runtime_raw["seed"] = int(env_seed)
    if env_tracking_uri:
        runtime_raw["tracking_uri"] = env_tracking_uri

    return ProjectConfig(
        name=project.get("name", "MechSage NASA C-MAPSS Pipeline"),
        experiment_prefix=project.get("experiment_prefix", "MechSage"),
        paths=PathConfig(**raw.get("paths", {})),
        runtime=RuntimeConfig(**runtime_raw),
        data=DataConfig(**raw.get("data", {})),
        features=FeatureConfig(**raw.get("features", {})),
        training=TrainingConfig(**raw.get("training", {})),
        models=raw.get("models", {}),
        logging=LoggingConfig(**raw.get("logging", {})),
    )
