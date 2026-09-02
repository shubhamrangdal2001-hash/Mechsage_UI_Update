"""Integration test that runs the CLI pipeline end to end."""

import subprocess
import sys
from pathlib import Path


def test_main_quick_pipeline_runs_end_to_end(workspace_tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    tracking_uri = (workspace_tmp_path / "mlruns").resolve().as_uri()

    command = [
        sys.executable,
        "main.py",
        "--profile",
        "quick",
        "--dataset",
        "FD001",
        "--max-train-units",
        "6",
        "--max-test-units",
        "4",
        "--models",
        "LinearRegression",
        "--anomaly-models",
        "IsolationForest",
        "--tune",
        "--tuning-methods",
        "optuna",
        "--tuning-models",
        "RandomForest",
        "--n-trials",
        "1",
        "--cv",
        "2",
        "--experiment-name",
        "pytest_integration",
        "--tracking-uri",
        tracking_uri,
    ]

    result = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Pipeline complete." in result.stdout
    assert (project_root / "reports" / "model_comparison_FD001.csv").exists()
    assert (project_root / "reports" / "final_report_FD001.md").exists()
