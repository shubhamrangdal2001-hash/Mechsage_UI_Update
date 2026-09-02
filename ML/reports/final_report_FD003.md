# Final ML Report - FD003

- Dataset version: `bb1be79f3434`
- Features used: `141`
- Train rows: `2416`
- Validation rows: `659`
- Test rows: `1211`

## Best Models
- Best RUL model: `RandomForest` with test RMSE `15.5592`
- Best anomaly model: `LightGBM_Anomaly` with test F1 `0.0000`

## Model Interpretability (SHAP Analysis)


## MLflow

Run `mlflow ui --backend-store-uri ./mlruns` from this project folder, then open http://127.0.0.1:5000.

## Comparison Table

```
      Model name                                                                                                                                                                                                                                                                                                                                                                                   Best hyperparameters  Train score  Validation score  Test score Main metric  Runtime                    MLflow run ID                                                                                                             Model artifact path              Task                                             Local model path
LinearRegression                                                                                                                                                                                                                                                                                                               {"copy_X": true, "fit_intercept": true, "n_jobs": null, "positive": false, "tol": 1e-06}     8.040457         19.709101   17.623062        RMSE 0.019924 3879ea90308f4fffa447a7b55c1315eb file:///D:/Capstone/MechSage/NASA_CMAPSS_RUL_Project/mlruns/478754518221550792/3879ea90308f4fffa447a7b55c1315eb/artifacts/model    RUL Regression D:\Capstone\MechSage\ML\models\LinearRegression_FD003.joblib
    RandomForest {"bootstrap": true, "ccp_alpha": 0.0, "criterion": "squared_error", "max_depth": 5, "max_features": 1.0, "max_leaf_nodes": null, "max_samples": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": null, "n_estimators": 25, "n_jobs": 1, "oob_score": false, "random_state": 42, "verbose": 0, "warm_start": false}     5.933699         14.895848   15.559231        RMSE 1.188962 63af2ce600944855a5ab40d471ff4b56 file:///D:/Capstone/MechSage/NASA_CMAPSS_RUL_Project/mlruns/478754518221550792/63af2ce600944855a5ab40d471ff4b56/artifacts/model    RUL Regression     D:\Capstone\MechSage\ML\models\RandomForest_FD003.joblib
LightGBM_Anomaly                                    {"best_threshold": 0.53, "class_weight": "balanced", "colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 5, "min_child_samples": 30, "n_estimators": 300, "n_jobs": -1, "num_leaves": 31, "random_state": 42, "reg_alpha": 0.1, "reg_lambda": 1.0, "smote_applied": true, "subsample": 0.8, "threshold_source": "validation_f1_sweep_0.01", "verbose": -1}     1.000000          0.764706    0.000000    F1_Score 5.112425 45c3132512554f9d9d28010d1db34340 file:///D:/Capstone/MechSage/NASA_CMAPSS_RUL_Project/mlruns/478754518221550792/45c3132512554f9d9d28010d1db34340/artifacts/model Anomaly Detection D:\Capstone\MechSage\ML\models\LightGBM_Anomaly_FD003.joblib
```