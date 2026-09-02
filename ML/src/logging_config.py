"""Central logging setup for command-line runs and tests."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from src.utils import get_project_root


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(
    logs_dir: Optional[str] = None,
    file_name: str = "pipeline.log",
    level: str = "INFO",
) -> Path:
    """Configure stream and file logging.

    Args:
        logs_dir: Directory where log files should be written.
        file_name: Name of the log file.
        level: Logging level such as ``INFO`` or ``DEBUG``.

    Returns:
        Path to the active log file.
    """
    root = Path(get_project_root())
    log_directory = Path(logs_dir) if logs_dir else root / "logs"
    if not log_directory.is_absolute():
        log_directory = root / log_directory
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / file_name

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.INFO)
    return log_path
