"""Command-line entrypoint for the MechSage C-MAPSS ML pipeline.

Example:
    python main.py --profile quick --dataset FD001 --tune
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import ProjectConfig, load_config
from src.exceptions import MechSageError
from src.logging_config import configure_logging
from src.pipeline import run_pipeline


LOGGER = logging.getLogger("mechsage.main")


def _base_parser() -> argparse.ArgumentParser:
    """Create a parser for early config-file discovery."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None, help="Optional YAML configuration file path")
    return parser


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments using config values as defaults.

    Returns:
        Parsed command-line namespace.
    """
    known_args, _ = _base_parser().parse_known_args()
    config = load_config(known_args.config)

    parser = argparse.ArgumentParser(description="MechSage NASA C-MAPSS ML pipeline with MLflow tracking")
    parser.add_argument("--config", default=known_args.config, help="Optional YAML configuration file path")
    parser.add_argument("--dataset", default=config.data.dataset, choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--profile", default=config.runtime.profile, choices=["quick", "standard", "full"],
                        help="quick is for local smoke tests, full trains the complete sklearn model set")
    parser.add_argument("--seed", type=int, default=config.runtime.seed)
    parser.add_argument("--validation-size", type=float, default=config.runtime.validation_size)
    parser.add_argument("--rul-clip", type=int, default=config.data.rul_clip)
    parser.add_argument("--anomaly-threshold", type=int, default=config.data.anomaly_threshold)
    parser.add_argument("--rolling-windows", type=int, nargs="+", default=config.features.rolling_windows)
    parser.add_argument("--max-train-units", type=int, default=None,
                        help="Limit engine units for fast local/debug runs")
    parser.add_argument("--max-test-units", type=int, default=None,
                        help="Limit test engine units for fast local/debug runs")
    parser.add_argument("--models", nargs="*", default=None,
                        choices=["LinearRegression", "RandomForest", "LightGBM", "XGBoost", "CatBoost", "LSTM", "GRU", "CNN_1D", "Transformer"])
    parser.add_argument("--skip-anomaly", action="store_true")
    parser.add_argument("--anomaly-models", nargs="*", default=None,
                        choices=["IsolationForest", "LightGBM_Anomaly", "OneClassSVM", "Autoencoder", "LSTM_Autoencoder"])
    parser.add_argument("--tune", action="store_true",
                        help="Run GridSearchCV, RandomizedSearchCV, and/or Optuna tuning")
    parser.add_argument("--tuning-methods", nargs="+", default=config.training.tuning_methods,
                        choices=["grid", "random", "optuna"])
    parser.add_argument("--tuning-models", nargs="*", default=None,
                        choices=["RandomForest", "LightGBM", "XGBoost", "CatBoost"])
    parser.add_argument("--n-trials", type=int, default=config.training.n_trials,
                        help="Optuna trials and RandomizedSearchCV iterations per model")
    parser.add_argument("--cv", type=int, default=config.training.cv)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--tracking-uri", default=config.runtime.tracking_uri,
                        help="Optional MLflow tracking URI; defaults to local ./mlruns")
    args = parser.parse_args()
    if args.max_train_units is None:
        args.max_train_units = config.data.max_train_units.get(args.profile)
    if args.max_test_units is None:
        args.max_test_units = config.data.max_test_units.get(args.profile)
    args.project_config = config
    return args


def main() -> int:
    """Run the pipeline from the command line.

    Returns:
        Process exit code.
    """
    args = parse_args()
    config: ProjectConfig = args.project_config
    log_path = configure_logging(
        logs_dir=config.paths.logs_dir,
        file_name=config.logging.file_name,
        level=config.logging.level,
    )
    LOGGER.info("Logging initialized: %s", log_path)

    try:
        result = run_pipeline(args)
    except MechSageError as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        return 1
    except Exception:
        LOGGER.exception("Unexpected pipeline failure")
        return 1

    best_rul = result.get("best_rul")
    best_anomaly = result.get("best_anomaly")
    LOGGER.info("Pipeline complete.")
    if best_rul:
        LOGGER.info("Best RUL model: %s | test RMSE=%.4f", best_rul["Model name"], best_rul["Test score"])
    if best_anomaly:
        LOGGER.info("Best anomaly model: %s | test F1=%.4f", best_anomaly["Model name"], best_anomaly["Test score"])
    LOGGER.info("Comparison table: %s", result["report_paths"]["comparison_csv"])
    LOGGER.info("Final report: %s", result["report_paths"]["final_markdown"])
    LOGGER.info("View MLflow with: mlflow ui --backend-store-uri ./mlruns")
    LOGGER.info("MLflow URL: http://127.0.0.1:5000")

    # Create new report folder and save updated metrics
    if best_anomaly and best_rul:
        import json
        metrics_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_report")
        os.makedirs(metrics_folder, exist_ok=True)
        report_data = {
            "F1_score": best_anomaly["Test metrics dict"]["F1_Score"],
            "precision": best_anomaly["Test metrics dict"]["Precision"],
            "recall": best_anomaly["Test metrics dict"]["Recall"],
            "RMSE": best_rul["Test metrics dict"]["RMSE"]
        }
        report_file = os.path.join(metrics_folder, f"updated_metrics_{args.dataset}.json")
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=4)
        LOGGER.info("Saved updated metrics to: %s", report_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
