# MechSage NASA C-MAPSS ML/MLOps Pipeline

Production-oriented machine learning pipeline for NASA C-MAPSS turbofan Remaining Useful Life (RUL) regression and anomaly detection. The project includes reproducible configuration, data validation, MLflow tracking, hyperparameter tuning, evaluation reports, feature importance artifacts, model version cards, and pytest coverage.

## Architecture

```text
NASA_CMAPSS_RUL_Project/
  configs/default.yaml          # Runtime, path, model, feature, and tuning defaults
  data/                         # Local raw C-MAPSS files
  src/
    config.py                   # Typed YAML/environment configuration loader
    logging_config.py           # Centralized console + file logging
    exceptions.py               # Project-specific custom exceptions
    data_loader.py              # C-MAPSS loading and dataset version hashing
    data_validation.py          # Missing/duplicate/schema/outlier validation
    preprocessing.py            # Sensor selection and scaling helpers
    feature_engineering.py      # Rolling stats, deltas, sequence windows
    evaluate.py                 # Regression, classification, anomaly metrics
    mlflow_tracking.py          # MLflow params, metrics, models, artifacts
    hyperparameter_tuning.py    # GridSearchCV, RandomizedSearchCV, Optuna
    pipeline.py                 # End-to-end orchestration
    model_registry.py           # Model version metadata cards
    inference.py                # Load model and generate predictions
    visualization.py            # Residual, ROC, and PR curve plots
    train_rul_models.py         # RUL model definitions/helpers
    train_anomaly_models.py     # Anomaly model definitions/helpers
    utils.py                    # Seeds, paths, plotting helpers
  tests/                        # Unit, integration, validation, tuning, inference tests
  artifacts/                    # Generated plots, predictions, confusion matrices
  models/                       # Saved local model files
  reports/                      # CSV/JSON/Markdown reports and model cards
  logs/                         # Pipeline logs
  mlruns/                       # MLflow tracking store
  main.py                       # CLI entrypoint
  pyproject.toml                # Packaging and tooling config
  setup.py                      # Editable install support
  requirements.txt
```

## Installation

```powershell
cd D:\Capstone\MechSage\NASA_CMAPSS_RUL_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Configuration

Defaults live in:

```text
configs/default.yaml
```

You can override with CLI flags:

```powershell
python main.py --config configs/default.yaml --profile quick --dataset FD001
```

Supported environment overrides:

```text
MECHSAGE_RANDOM_SEED
MLFLOW_TRACKING_URI
```

## Run Tests

```powershell
python -m pytest
```

Test coverage includes:

- Metrics
- Data loading and feature generation
- Data validation
- Model loading and prediction
- Hyperparameter tuning
- CLI integration

## Train and Evaluate

Quick local run:

```powershell
python main.py --profile quick --dataset FD001 --tune --tuning-methods optuna --tuning-models RandomForest --n-trials 1 --cv 2
```

Full experiment run:

```powershell
python main.py --profile full --dataset FD001 --tune --tuning-methods grid random optuna --n-trials 10 --cv 3
```

## MLflow

Start the UI:

```powershell
mlflow ui --backend-store-uri ./mlruns
```

Open:

```text
http://127.0.0.1:5000
```

MLflow tracks:

- Dataset version
- Train/validation/test split details
- Preprocessing and feature engineering
- Hyperparameters
- Tuning trial metrics
- Train, validation, and test metrics
- Models
- Prediction CSV files
- Confusion matrices
- Feature importance plots
- Residual plots
- ROC and precision-recall curves

## Outputs

```text
reports/model_comparison_FD001.csv
reports/final_report_FD001.md
reports/final_report_FD001.json
reports/data_validation_FD001.json
reports/best_rul_metrics_FD001.json
reports/best_anomaly_metrics_FD001.json
reports/best_rul_model_card_FD001.json
reports/best_anomaly_model_card_FD001.json
artifacts/
models/
logs/pipeline.log
mlruns/
```

## Inference

Use the inference helpers from Python:

```python
from src.inference import load_model, predict

model = load_model("models/best_rul_model_FD001.joblib")
predictions = predict(model, feature_matrix)
```

Or score a CSV:

```python
from src.inference import predict_from_csv

predict_from_csv(
    "models/best_rul_model_FD001.joblib",
    "features.csv",
    "reports/predictions.csv",
)
```

## Metrics

Regression:

- MAE
- MSE
- RMSE
- R2
- Adjusted R2
- MAPE
- NASA PHM08 score

Anomaly/classification:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- False positive rate
- False negative rate
- Confusion matrix

## Troubleshooting

If dependencies are missing:

```powershell
pip install -r requirements.txt
```

If MLflow does not open, verify port 5000 is free:

```powershell
netstat -ano | Select-String ":5000"
```

If the dataset cannot be found, place C-MAPSS files in either:

```text
NASA_CMAPSS_RUL_Project/data/
```

or the parent workspace data directory:

```text
D:\Capstone\MechSage\data\NASA C-MAPSS-1 Turbofan Engine Degradation Dataset
```
