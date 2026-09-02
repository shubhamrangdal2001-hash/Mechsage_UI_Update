# Final ML Report - FD004

- Dataset version: `071b815ff8fc`
- Features used: `141`
- Train rows: `2231`
- Validation rows: `633`
- Test rows: `1231`

## Best Models
- Best RUL model: `LinearRegression` with test RMSE `17.9755`
- Best anomaly model: `LightGBM_Anomaly` with test F1 `0.3864`

## Model Interpretability (SHAP Analysis)


## MLflow

Run `mlflow ui --backend-store-uri ./mlruns` from this project folder, then open http://127.0.0.1:5000.

## Comparison Table

```
      Model name                                                                                                                                                                                                                                                                                                                                                                                   Best hyperparameters  Train score  Validation score  Test score Main metric  Runtime                    MLflow run ID                                                                                                             Model artifact path              Task                                             Local model path
LinearRegression                                                                                                                                                                                                                                                                                                               {"copy_X": true, "fit_intercept": true, "n_jobs": null, "positive": false, "tol": 1e-06}    11.568920         15.412125   17.975497        RMSE 0.017506 b99bc92bd1b345f383fa6cb347b7fd46 file:///D:/Capstone/MechSage/NASA_CMAPSS_RUL_Project/mlruns/203867875341978885/b99bc92bd1b345f383fa6cb347b7fd46/artifacts/model    RUL Regression D:\Capstone\MechSage\ML\models\LinearRegression_FD004.joblib
    RandomForest {"bootstrap": true, "ccp_alpha": 0.0, "criterion": "squared_error", "max_depth": 5, "max_features": 1.0, "max_leaf_nodes": null, "max_samples": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": null, "n_estimators": 25, "n_jobs": 1, "oob_score": false, "random_state": 42, "verbose": 0, "warm_start": false}    17.346512         28.317498   22.588651        RMSE 1.227410 e736077ee7a14193b3eb8a8b85ef3db0 file:///D:/Capstone/MechSage/NASA_CMAPSS_RUL_Project/mlruns/203867875341978885/e736077ee7a14193b3eb8a8b85ef3db0/artifacts/model    RUL Regression     D:\Capstone\MechSage\ML\models\RandomForest_FD004.joblib
LightGBM_Anomaly                                    {"best_threshold": 0.14, "class_weight": "balanced", "colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 5, "min_child_samples": 30, "n_estimators": 300, "n_jobs": -1, "num_leaves": 31, "random_state": 42, "reg_alpha": 0.1, "reg_lambda": 1.0, "smote_applied": true, "subsample": 0.8, "threshold_source": "validation_f1_sweep_0.01", "verbose": -1}     1.000000          0.622951    0.386364    F1_Score 4.858409 4cf215014b9147ea9539e12aac38a1fc file:///D:/Capstone/MechSage/NASA_CMAPSS_RUL_Project/mlruns/203867875341978885/4cf215014b9147ea9539e12aac38a1fc/artifacts/model Anomaly Detection D:\Capstone\MechSage\ML\models\LightGBM_Anomaly_FD004.joblib
```