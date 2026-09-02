import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import make_regression, make_classification
from src.shap_explain import run_shap_analysis, extract_top_features

def test_shap_analysis_regression():
    # 1. Generate synthetic regression data
    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    feature_names = [f"sensor_{i}" for i in range(5)]
    
    # 2. Train a simple model
    model = RandomForestRegressor(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)
    
    # 3. Create dataset dictionary
    data = {
        "X_train": X,
        "X_test": X,
        "feature_cols": feature_names
    }
    
    # 4. Run SHAP analysis in a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = run_shap_analysis(
            model=model,
            data=data,
            task_type="rul",
            model_name="RandomForest",
            dataset_id="FD001",
            save_dir=tmp_dir
        )
        
        # 5. Assertions
        assert "plots" in result
        assert "top_features" in result
        
        plots = result["plots"]
        assert "summary_bar" in plots
        assert "summary_beeswarm" in plots
        assert "dependence" in plots
        
        # Verify files exist
        assert os.path.exists(plots["summary_bar"])
        assert os.path.exists(plots["summary_beeswarm"])
        assert os.path.exists(plots["dependence"])
        
        # Verify top features list
        top_features = result["top_features"]
        assert len(top_features) > 0
        assert len(top_features) <= 5
        assert isinstance(top_features[0], tuple)
        assert isinstance(top_features[0][0], str)
        assert isinstance(top_features[0][1], float)

def test_shap_analysis_classification():
    # 1. Generate synthetic classification data
    X, y = make_classification(n_samples=50, n_features=5, n_informative=3, random_state=42)
    feature_names = [f"sensor_{i}" for i in range(5)]
    
    # 2. Train a classifier
    model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)
    
    # 3. Create dataset dictionary
    data = {
        "X_train": X,
        "X_test": X,
        "feature_cols": feature_names
    }
    
    # 4. Run SHAP analysis in a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = run_shap_analysis(
            model=model,
            data=data,
            task_type="anomaly",
            model_name="RandomForestClassifier",
            dataset_id="FD001",
            save_dir=tmp_dir
        )
        
        # 5. Assertions
        assert "plots" in result
        assert "top_features" in result
        
        plots = result["plots"]
        assert "summary_bar" in plots
        assert "summary_beeswarm" in plots
        assert "dependence" in plots
        
        # Verify files exist
        assert os.path.exists(plots["summary_bar"])
        assert os.path.exists(plots["summary_beeswarm"])
        assert os.path.exists(plots["dependence"])
        
        top_features = result["top_features"]
        assert len(top_features) > 0
