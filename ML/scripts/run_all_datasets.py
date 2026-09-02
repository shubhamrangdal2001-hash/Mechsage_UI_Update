"""
Run the MechSage pipeline on all 4 NASA C-MAPSS datasets and collect
ROC-AUC and PR-AUC for anomaly detection across FD001–FD004.
"""

import json
import os
import sys
import types
from pathlib import Path

# ── ensure project is on path ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import prepare_data, run_pipeline


DATASETS = ["FD001", "FD002", "FD003", "FD004"]

def make_args(dataset: str):
    return types.SimpleNamespace(
        dataset=dataset,
        profile="standard",
        seed=42,
        validation_size=0.20,
        rul_clip=125,
        anomaly_threshold=30,
        rolling_windows=(5, 10, 20),
        max_train_units=None,
        max_test_units=None,
        models=["LinearRegression", "RandomForest"],
        anomaly_models=["IsolationForest", "LightGBM_Anomaly"],
        tune=False,
        tuning_models=[],
        tuning_methods=[],
        n_trials=5,
        cv=3,
        skip_anomaly=False,
        experiment_name=None,
        tracking_uri=None,
    )


results = {}

for ds in DATASETS:
    print(f"\n{'='*60}")
    print(f"  Running pipeline for {ds}")
    print(f"{'='*60}")
    try:
        out = run_pipeline(make_args(ds))
        rows = out["rows"]

        ds_metrics = {"dataset": ds, "anomaly_models": {}, "rul_models": {}}

        for row in rows:
            name = row["Model name"]
            task = row.get("Task", "")
            test_m = row.get("Test metrics dict", {})

            if task == "Anomaly Detection":
                ds_metrics["anomaly_models"][name] = {
                    "ROC_AUC":  round(test_m.get("ROC_AUC", float("nan")), 4),
                    "PR_AUC":   round(test_m.get("PR_AUC",  float("nan")), 4),
                    "F1_Score": round(test_m.get("F1_Score", float("nan")), 4),
                    "Precision":round(test_m.get("Precision", float("nan")), 4),
                    "Recall":   round(test_m.get("Recall",   float("nan")), 4),
                }
            elif task == "RUL Regression":
                ds_metrics["rul_models"][name] = {
                    "RMSE": round(test_m.get("RMSE", float("nan")), 4),
                    "MAE":  round(test_m.get("MAE",  float("nan")), 4),
                    "R2":   round(test_m.get("R2",   float("nan")), 4),
                }

        results[ds] = ds_metrics

    except Exception as exc:
        print(f"  ERROR on {ds}: {exc}")
        results[ds] = {"error": str(exc)}


# ── Print summary table ──────────────────────────────────────────────────────
print("\n\n" + "="*70)
print("  ANOMALY DETECTION — ROC-AUC / PR-AUC SUMMARY (Test Set)")
print("="*70)

header = f"{'Dataset':<8} {'Model':<25} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8}"
print(header)
print("-" * len(header))

for ds, data in results.items():
    if "error" in data:
        print(f"{ds:<8}  ERROR: {data['error'][:50]}")
        continue
    for model_name, m in data.get("anomaly_models", {}).items():
        print(f"{ds:<8} {model_name:<25} {m['ROC_AUC']:>8.4f} {m['PR_AUC']:>8.4f} {m['F1_Score']:>8.4f}")

print("\n" + "="*70)
print("  RUL REGRESSION — RMSE SUMMARY (Test Set)")
print("="*70)

header2 = f"{'Dataset':<8} {'Model':<25} {'RMSE':>8} {'MAE':>8} {'R2':>8}"
print(header2)
print("-" * len(header2))

for ds, data in results.items():
    if "error" in data:
        continue
    for model_name, m in data.get("rul_models", {}).items():
        print(f"{ds:<8} {model_name:<25} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['R2']:>8.4f}")

# ── Save to JSON ─────────────────────────────────────────────────────────────
out_path = PROJECT_ROOT / "reports" / "all_datasets_results.json"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nFull results saved: {out_path}")
