"""SHAP (Shapley Additive exPlanations) explainability module for model interpretation."""

import logging
import os
import matplotlib
matplotlib.use('Agg')  # Ensure no GUI backends are used
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

LOGGER = logging.getLogger("mechsage.shap_explain")

def normalize_shap_values(shap_values, X_sample_df):
    """Normalize shap_values to handle Explanation objects, lists, 3D arrays, etc.
    
    Ensures we have either a 2D numpy array or a 2D Explanation object.
    """
    if hasattr(shap_values, "values"):
        # It's an Explanation object
        if len(shap_values.values.shape) == 3:  # (samples, features, classes)
            # Extract class 1 for binary classification (anomaly)
            return shap_values[..., 1]
        return shap_values
    
    if isinstance(shap_values, list):
        # List of classes
        if len(shap_values) == 2:
            return shap_values[1]
        return shap_values[0]
    
    if isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 3:
            # (samples, features, classes)
            return shap_values[:, :, 1]
        return shap_values
        
    return shap_values

def run_shap_analysis(
    model,
    data: dict,
    task_type: str,  # "rul" or "anomaly"
    model_name: str,
    dataset_id: str,
    save_dir: str,
) -> dict:
    """Generate SHAP explainability plots for the given model and save them.
    
    Args:
        model: The trained estimator model (e.g. RandomForest, LightGBM, PyTorch model)
        data: The dataset context dictionary containing train/test data
        task_type: "rul" or "anomaly"
        model_name: Name of the model
        dataset_id: C-MAPSS dataset id (e.g. "FD001")
        save_dir: Directory where the generated plots will be saved.
        
    Returns:
        A dict containing paths of saved plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    feature_names = data.get("feature_cols", [])
    if not feature_names:
        LOGGER.warning("No feature_cols found in data dictionary. Skipping SHAP analysis.")
        return {}
        
    X_train = data.get("X_train")
    X_test = data.get("X_test")
    
    if X_train is None or X_test is None:
        LOGGER.warning("X_train or X_test not found in data dictionary. Skipping SHAP analysis.")
        return {}

    # Downsample datasets to speed up computation
    rng = np.random.default_rng(42)
    bg_size = min(100, len(X_train))
    bg_indices = rng.choice(len(X_train), size=bg_size, replace=False)
    bg_data = X_train[bg_indices]
    
    test_size = min(200, len(X_test))
    test_indices = rng.choice(len(X_test), size=test_size, replace=False)
    X_sample = X_test[test_indices]
    
    X_train_df = pd.DataFrame(bg_data, columns=feature_names)
    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
    
    model_class = model.__class__.__name__
    explainer = None
    shap_values = None
    is_pytorch = "nn.Module" in str(type(model)) or hasattr(model, "forward")
    
    # 1. Handle PyTorch Models
    if is_pytorch:
        LOGGER.info(f"Using DeepExplainer for PyTorch model: {model_name}")
        try:
            X_train_seq = data.get("X_train_seq")
            X_test_seq = data.get("X_test_seq")
            if X_train_seq is not None and X_test_seq is not None:
                # Sequence data shape: (N, seq_len, features)
                bg_seq = torch.tensor(X_train_seq[:30], dtype=torch.float32)
                test_seq = torch.tensor(X_test_seq[:10], dtype=torch.float32)
                
                # Check if device is GPU and put model/data on it
                device = next(model.parameters()).device
                bg_seq = bg_seq.to(device)
                test_seq = test_seq.to(device)
                
                model.eval()
                # Run DeepExplainer
                explainer = shap.DeepExplainer(model, bg_seq)
                raw_shap = explainer.shap_values(test_seq)
                
                # Extract shap values
                if isinstance(raw_shap, list):
                    raw_shap = raw_shap[0]
                
                # Average over sequence length axis (axis 1)
                # raw_shap shape is (N, seq_len, features)
                if isinstance(raw_shap, torch.Tensor):
                    raw_shap = raw_shap.cpu().numpy()
                shap_values_2d = raw_shap.mean(axis=1)
                
                # Also average inputs over sequence length axis
                X_sample_2d = test_seq.cpu().numpy().mean(axis=1)
                X_sample_df = pd.DataFrame(X_sample_2d, columns=feature_names)
                
                shap_values = shap_values_2d
            else:
                LOGGER.warning("PyTorch sequence data not found in data dict. Skipping SHAP analysis.")
                return {}
        except Exception as e:
            LOGGER.warning(f"PyTorch DeepExplainer failed: {e}. Skipping deep learning SHAP analysis.")
            return {}
            
    # 2. Handle Machine Learning Models (scikit-learn, xgboost, lightgbm, catboost)
    else:
        try:
            # Tree-based explainers (very fast)
            if any(t in model_class for t in ["Forest", "LGBM", "XGB", "CatBoost", "Tree", "Isolation"]):
                LOGGER.info(f"Initializing TreeExplainer for model: {model_name} ({model_class})")
                if "Isolation" in model_class:
                    # IsolationForest requires background data or might have issues in some shap versions
                    explainer = shap.TreeExplainer(model, data=X_train_df)
                else:
                    explainer = shap.TreeExplainer(model)
            # Linear explainers
            elif "Linear" in model_class or "Ridge" in model_class or "Lasso" in model_class:
                LOGGER.info(f"Initializing LinearExplainer for model: {model_name} ({model_class})")
                explainer = shap.LinearExplainer(model, X_train_df)
            # Fallback to model-agnostic Explainer
            else:
                LOGGER.info(f"Initializing generic Explainer for model: {model_name} ({model_class})")
                explainer = shap.Explainer(model, X_train_df)
                
            # Compute SHAP values
            try:
                # New API
                shap_values = explainer(X_sample_df)
            except Exception:
                # Old API fallback
                shap_values = explainer.shap_values(X_sample_df)
                
        except Exception as e:
            LOGGER.warning(f"SHAP initialization or value computation failed for {model_name}: {e}. Trying generic KernelExplainer.")
            try:
                # Final fallback: KernelExplainer on small sample
                explainer = shap.KernelExplainer(model.predict, shap.sample(X_train_df, 10))
                shap_values = explainer.shap_values(X_sample_df)
            except Exception as ex:
                LOGGER.error(f"All SHAP explainers failed for {model_name}: {ex}")
                return {}

    # Normalize SHAP values (extract anomaly class if binary classifier, etc.)
    shap_values = normalize_shap_values(shap_values, X_sample_df)
    
    # Check if we have valid shap values
    if shap_values is None:
        LOGGER.warning("Could not calculate SHAP values. Skipping plot generation.")
        return {}

    # Generate Plots
    saved_plots = {}
    
    # 1. Summary Bar Plot
    try:
        plt.figure(figsize=(10, 6))
        if hasattr(shap_values, "values"):
            shap.plots.bar(shap_values, max_display=15, show=False)
        else:
            shap.summary_plot(shap_values, X_sample_df, plot_type="bar", max_display=15, show=False)
            
        plt.title(f"SHAP Global Feature Importance - {model_name} ({task_type.upper()})")
        plt.tight_layout()
        bar_path = os.path.join(save_dir, f"best_{task_type}_shap_summary_bar_{dataset_id}.png")
        plt.savefig(bar_path, dpi=180)
        plt.close()
        saved_plots["summary_bar"] = bar_path
    except Exception as e:
        LOGGER.error(f"Error plotting SHAP summary bar: {e}")
        plt.close()
        
    # 2. Summary Beeswarm Plot
    try:
        plt.figure(figsize=(10, 6))
        if hasattr(shap_values, "values"):
            try:
                shap.plots.beeswarm(shap_values, max_display=15, show=False)
            except Exception:
                shap.summary_plot(shap_values.values, X_sample_df, max_display=15, show=False)
        else:
            shap.summary_plot(shap_values, X_sample_df, max_display=15, show=False)
            
        plt.title(f"SHAP Beeswarm Plot - {model_name} ({task_type.upper()})")
        plt.tight_layout()
        beeswarm_path = os.path.join(save_dir, f"best_{task_type}_shap_summary_beeswarm_{dataset_id}.png")
        plt.savefig(beeswarm_path, dpi=180)
        plt.close()
        saved_plots["summary_beeswarm"] = beeswarm_path
    except Exception as e:
        LOGGER.error(f"Error plotting SHAP beeswarm: {e}")
        plt.close()

    # 3. Dependence Plot for the Top Feature
    try:
        if hasattr(shap_values, "values"):
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
        if len(mean_abs_shap.shape) > 1:
            mean_abs_shap = mean_abs_shap.mean(axis=tuple(range(1, len(mean_abs_shap.shape))))
            
        top_idx = int(np.argmax(mean_abs_shap))
        top_feature = feature_names[top_idx]
        
        plt.figure(figsize=(8, 5))
        vals = shap_values.values if hasattr(shap_values, "values") else shap_values
        shap.dependence_plot(top_feature, vals, X_sample_df, show=False)
        plt.title(f"SHAP Dependence Plot: {top_feature} - {model_name}")
        plt.tight_layout()
        dependence_path = os.path.join(save_dir, f"best_{task_type}_shap_dependence_{dataset_id}.png")
        plt.savefig(dependence_path, dpi=180)
        plt.close()
        saved_plots["dependence"] = dependence_path
    except Exception as e:
        LOGGER.error(f"Error plotting SHAP dependence: {e}")
        plt.close()

    return {
        "plots": saved_plots,
        "top_features": extract_top_features(shap_values, feature_names, top_n=5)
    }

def extract_top_features(shap_values, feature_names: list[str], top_n: int = 5) -> list[tuple[str, float]]:
    """Extract the top N features by mean absolute SHAP value."""
    try:
        if hasattr(shap_values, "values"):
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
        if len(mean_abs_shap.shape) > 1:
            mean_abs_shap = mean_abs_shap.mean(axis=tuple(range(1, len(mean_abs_shap.shape))))
            
        indices = np.argsort(mean_abs_shap)[::-1][:top_n]
        return [(feature_names[i], float(mean_abs_shap[i])) for i in indices]
    except Exception as e:
        LOGGER.error(f"Failed to extract top features from SHAP: {e}")
        return []
