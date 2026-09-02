"""Pytest configuration for local package imports and workspace temp folders."""

import os
import sys
import uuid
from pathlib import Path

import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def workspace_tmp_path():
    root = Path(PROJECT_ROOT) / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    return path
