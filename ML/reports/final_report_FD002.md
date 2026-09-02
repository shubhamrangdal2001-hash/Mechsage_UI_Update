# Final ML Report - FD002

- Dataset version: `26efe536f436`
- Features used: `141`
- Train rows: `1428`
- Validation rows: `468`
- Test rows: `1132`

## Best Models
- Best RUL model: `LinearRegression` with test RMSE `29.9854`
- Best anomaly model: `LightGBM_Anomaly` with test F1 `0.6504`

## Model Interpretability (SHAP Analysis)


## MLflow

Run `mlflow ui --backend-store-uri ./mlruns` from this project folder, then open http://127.0.0.1:5000.

## Comparison Table

```
      Model name                                                                                                                                                                                                                                                                                                                                                                                   Best hyperparameters  Train score  Validation score  Test score Main metric  Runtime                    MLflow run ID                                                                                                             Model artifact path              Task                                             Local model path
LinearRegression                                                                                                                                                                                                                                                                                                               {"copy_X": true, "fit_intercept": true, "n_jobs": null, "positive": false, "tol": 1e-06}    12.370480         27.346541   29.985369        RMSE 0.043630 47955ccfc03c4400892a49c7012cd7d8 file:///D:/Capstone/MechSage/NASA_CMAPSS_RUL_Project/mlruns/203621584362175723/47955ccfc03c4400892a49c7012cd7d8/artifacts/model    RUL Regression D:\Capstone\MechSage\ML\models\LinearRegression_FD002.joblib
    RandomForest {"bootstrap": true, "ccp_alpha": 0.0, "criterion": "squared_error", "max_depth": 5, "max_features": 1.0, "max_leaf_nodes": null, "max_samples": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": null, "n_estimators": 25, "n_jobs": 1, "oob_score": false, "random_state": 42, "verbose": 0, "warm_start": false}    16.888882         32.873836   35.311167        RMSE 0.887362 68d71a63db484f85a698b6deb2abb546 file:///D:/Capstone/MechSage/NASA_CMAPSS_RUL_Project/mlruns/203621584362175723/68d71a63db484f85a698b6deb2abb546/artifacts/model    RUL Regression     D:\Capstone\MechSage\ML\models\RandomForest_FD002.joblib
LightGBM_Anomaly                                    {"best_threshold": 0.91, "class_weight": "balanced", "colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 5, "min_child_samples": 30, "n_estimators": 300, "n_jobs": -1, "num_leaves": 31, "random_state": 42, "reg_alpha": 0.1, "reg_lambda": 1.0, "smote_applied": true, "subsample": 0.8, "threshold_source": "validation_f1_sweep_0.01", "verbose": -1}     1.000000          0.609929    0.650407    F1_Score 4.673653 788124a77a5c4b029725bbbb420928e2 file:///D:/Capstone/MechSage/NASA_CMAPSS_RUL_Project/mlruns/203621584362175723/788124a77a5c4b029725bbbb420928e2/artifacts/model Anomaly Detection D:\Capstone\MechSage\ML\models\LightGBM_Anomaly_FD002.joblib
```