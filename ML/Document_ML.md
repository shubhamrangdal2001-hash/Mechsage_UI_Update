# MechSage — Complete Model Guide

> **What this file is.** A single reference for every ML and DL model used in this project —
> what it is, why we chose it, its hyperparameters, and how it fits into the overall
> Ironside predictive-maintenance pipeline.

---

## Table of Contents

1. [Project Context](#1-project-context)
2. [Where Models Fit in the Pipeline](#2-where-models-fit-in-the-pipeline)
3. [Data & Feature Engineering](#3-data--feature-engineering)
4. [Task A — RUL Regression Models](#4-task-a--rul-regression-models)
   - [4.1 ML Models (Scikit-learn / Tree-based)](#41-ml-models-scikit-learn--tree-based)
   - [4.2 DL Models (PyTorch)](#42-dl-models-pytorch)
5. [Task B — Anomaly Detection Models](#5-task-b--anomaly-detection-models)
   - [5.1 Unsupervised ML Models](#51-unsupervised-ml-models)
   - [5.2 Supervised ML Model](#52-supervised-ml-model)
   - [5.3 DL Autoencoder Models](#53-dl-autoencoder-models)
6. [Hyperparameter Tuning Strategy](#6-hyperparameter-tuning-strategy)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Model Selection Summary](#8-model-selection-summary)

---

## 1. Project Context

**Client:** Ironside Manufacturing (fictional capstone persona)
**Problem:** Machines degrade silently until they fail. We need to catch that degradation *early*.
**Data:** NASA C-MAPSS turbofan run-to-failure dataset (the approved public proxy for industrial machinery sensor streams).
**Goal:** Predict *how many cycles of useful life remain* (RUL) and *flag anomalous behaviour* for each asset in the fleet.

The ML/DL models in this codebase are **tools** that the agentic copilot calls — they are not the final product by themselves.

---

## 2. Where Models Fit in the Pipeline

```
Raw Sensor Streams (C-MAPSS)
         |
         v
  Feature Engineering
  (rolling stats, degradation trends, sequence windows)
         |
    +----+---------------------------+
    |                               |
    v                               v
Task A: RUL Regression          Task B: Anomaly Detection
(How much life is left?)        (Is this machine behaving abnormally?)
    |                               |
    +----------------+--------------+
                     |
                     v
            Diagnosis Agent (LLM)
            reads RUL + anomaly scores
            -> explains cause in plain language
            -> drafts work order
```

Both tasks run on the **same preprocessed data**. RUL is the headline north-star metric; anomaly detection is the cheap always-on monitoring layer.

---

## 3. Data & Feature Engineering

### Dataset: NASA C-MAPSS

| Property | Value |
|---|---|
| Subsets | FD001, FD002, FD003, FD004 |
| Engines (training) | 100-260 per subset |
| Sensors | 21 total -> 14 informative (7 near-constant dropped) |
| Op settings | 3 (flight condition parameters) |
| RUL cap | 125 cycles (piece-wise linear label) |
| Anomaly label | RUL <= 30 -> anomaly = 1 |

### 14 Informative Sensors Selected

```python
INFORMATIVE_SENSORS = [
    "sensor_2",  "sensor_3",  "sensor_4",  "sensor_7",
    "sensor_8",  "sensor_9",  "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21"
]
```

**Why drop 7 sensors?** Sensors 1, 5, 6, 10, 16, 18, 19 are near-constant across all cycles — they carry zero predictive signal. Including them adds noise, increases training time, and risks SHAP misattribution.

### Engineered Features (per sensor)

| Feature | Formula | Why |
|---|---|---|
| Rolling mean (w=5,10,20) | Mean over last w cycles per engine | Smooths sensor noise; captures trend |
| Rolling std (w=5,10,20) | Std over last w cycles per engine | Captures volatility increase near failure |
| Degradation delta | sensor - rolling_mean_10 | Flags deviation from recent baseline |
| Expanding mean | Running average from cycle 1 | Long-range drift signal |

**Causal guarantee:** All rolling windows are grouped by `unit_number` and sorted by `time_in_cycles`. No future data leaks into any feature window.

### Normalisation

- **MinMax scaling** applied on training data only (`fit_transform` on train, `transform` on test).
- Wrapped inside a `sklearn.Pipeline` during hyperparameter tuning so the scaler is never fitted on validation or test folds.

### Sequence Windows (for DL models only)

```
sequence_length = 30 cycles
X shape: (N, 30, num_features)   <- 3D tensor
y shape: (N,)                    <- RUL at the last cycle of each window
```

Short engines (< 30 cycles) are **front-padded** with the first row to avoid discarding data.

---

## 4. Task A — RUL Regression Models

### 4.1 ML Models (Scikit-learn / Tree-based)

Input: **2D flat feature matrix** `(N, num_features)` — one row per cycle.

---

#### Linear Regression *(baseline)*

**What it is:** Ordinary least-squares linear model.

**Why we use it:** A sanity-check baseline. If any of our real models cannot beat this, something is wrong with the pipeline. Also interpretable: coefficients directly show which features drive RUL.

**Hyperparameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `fit_intercept` | True (default) | Allows a constant offset |
| No tuning needed | — | OLS has a closed-form solution |

**Limitation:** Assumes a linear relationship between sensor readings and RUL. Real degradation is non-linear, so this underperforms but is useful as a lower-bound comparison.

---

#### Random Forest Regressor

**What it is:** An ensemble of decision trees, each trained on a bootstrap sample. Final prediction is the **average** across all trees.

**Why we use it:**
- Handles non-linear relationships naturally.
- Robust to outliers and irrelevant features.
- Built-in feature importance (Gini-based) — useful for explaining which sensors matter most.
- Low risk of overfitting due to averaging.

**Default hyperparameters (used in baseline training):**

| Parameter | Value | Meaning |
|---|---|---|
| `n_estimators` | 100 | Number of trees |
| `max_depth` | 10 | Max depth per tree (prevents overfit) |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Use all CPU cores |

**Tuning search space (Optuna):**

| Parameter | Range | Why |
|---|---|---|
| `n_estimators` | 80 - 350 | More trees -> better averaging up to a point |
| `max_depth` | 2 - 18 | Depth controls bias-variance tradeoff |
| `min_samples_leaf` | 1 - 8 | Prevents overly specific leaf nodes |
| `max_features` | "sqrt" or None | Controls correlation between trees |

---

#### XGBoost Regressor

**What it is:** Gradient boosted trees trained **sequentially** — each new tree corrects the residuals of the previous ones. Uses second-order gradient information.

**Why we use it:**
- State-of-the-art for tabular regression benchmarks.
- Handles missing values natively.
- `subsample` and `colsample_bytree` act as built-in regularisation.
- PHM08 published baselines used XGBoost-class models — our results are comparable.

**Default hyperparameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `n_estimators` | 200 | Number of boosting rounds |
| `learning_rate` | 0.05 | Step size shrinkage |
| `max_depth` | 6 | Tree depth |
| `subsample` | 0.8 | Row sampling per tree |
| `colsample_bytree` | 0.8 | Column sampling per tree |
| `random_state` | 42 | Reproducibility |

**Tuning search space (Optuna):**

| Parameter | Range | Why |
|---|---|---|
| `n_estimators` | 100 - 450 | More rounds with lower LR = better generalisation |
| `learning_rate` | 0.02 - 0.2 (log-scale) | Log-scale because small values matter more |
| `max_depth` | 3 - 10 | Deeper = more complex; risk of overfit |
| `subsample` | 0.7 - 1.0 | Row-level stochasticity |
| `colsample_bytree` | 0.7 - 1.0 | Feature-level stochasticity |

---

#### LightGBM Regressor

**What it is:** Gradient boosted trees using **leaf-wise** (best-first) growth instead of level-wise. Uses histogram binning for speed.

**Why we use it:**
- **Fastest** of the boosting trio — critical for fleet-scale monitoring.
- `num_leaves` controls model complexity more directly than `max_depth`.
- Often matches or beats XGBoost on medium datasets with proper tuning.

**Default hyperparameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `n_estimators` | 200 | Boosting rounds |
| `learning_rate` | 0.05 | Step size |
| `max_depth` | 6 | Tree depth |
| `subsample` | 0.8 | Row sampling |
| `colsample_bytree` | 0.8 | Feature sampling |
| `verbose` | -1 | Silent mode |

**Tuning search space (Optuna):**

| Parameter | Range | Why |
|---|---|---|
| `n_estimators` | 100 - 450 | — |
| `learning_rate` | 0.02 - 0.2 (log) | — |
| `max_depth` | 4 - 12 | -1 = no limit |
| `num_leaves` | 15 - 150 | Key LightGBM capacity control |
| `subsample` | 0.7 - 1.0 | — |

---

#### CatBoost Regressor

**What it is:** Gradient boosted trees with **ordered boosting** and symmetric trees.

**Why we use it:**
- Handles categorical features natively (useful if Ironside real data has machine type codes).
- Ordered boosting reduces target leakage within training.
- Often the most calibrated predictor with minimal tuning.

**Default hyperparameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `iterations` | 200 | Equivalent to n_estimators |
| `learning_rate` | 0.05 | Step size |
| `depth` | 6 | Symmetric tree depth |
| `random_seed` | 42 | Reproducibility |
| `verbose` | 0 | Silent mode |

**Tuning search space (Optuna):**

| Parameter | Range | Why |
|---|---|---|
| `iterations` | 100 - 450 | — |
| `learning_rate` | 0.02 - 0.2 (log) | — |
| `depth` | 4 - 10 | CatBoost depths are typically shallower |
| `l2_leaf_reg` | 0.1 - 10.0 (log) | L2 regularisation on leaf values |

---

### 4.2 DL Models (PyTorch)

Input: **3D sequence tensor** `(N, seq_len=30, num_features)`
Training: Adam optimizer + MSELoss + ReduceLROnPlateau scheduler (patience=5, factor=0.5)

---

#### LSTM Regressor

**What it is:** Long Short-Term Memory network — a recurrent network with gated memory cells that capture **long-range temporal dependencies** in sensor sequences.

**Why we use it:**
- C-MAPSS degradation is a temporal process — LSTM captures this naturally.
- Gates (input, forget, output) let the model selectively remember long-ago events.
- The dominant DL architecture in published C-MAPSS baselines — results are directly comparable.

**Architecture:**
```
Input -> LSTM (input_size -> 128, 2 layers, dropout=0.2) -> Dropout -> Linear(128 -> 1) -> RUL
```

**Hyperparameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `input_size` | num_features (dynamic) | Set at runtime from data |
| `hidden_size` | 128 | LSTM hidden state dimension |
| `num_layers` | 2 | Stacked LSTM layers |
| `dropout` | 0.2 | Between layers + before FC |
| `epochs` | 50 | Training iterations |
| `batch_size` | 64 | Samples per gradient update |
| `lr` | 1e-3 | Adam learning rate |
| scheduler | ReduceLROnPlateau (patience=5, factor=0.5) | Halves LR on plateau |

**Why hidden_size=128?** Feature space is ~50-80 features after engineering. 128 units gives enough capacity without overparameterising.

**Why 2 layers?** Layer 1 extracts low-level temporal patterns; Layer 2 models higher-level degradation trends. More layers don't help at C-MAPSS scale.

---

#### GRU Regressor

**What it is:** Gated Recurrent Unit — a streamlined RNN with two gates (reset + update) instead of LSTM's three. Fewer parameters, faster training.

**Why we use it:**
- GRU often matches LSTM on 30-cycle sequences while training faster.
- Useful comparison — if GRU achieves the same RMSE at lower cost, it's better for the cheap monitoring path.

**Architecture:**
```
Input -> GRU (input_size -> 128, 2 layers, dropout=0.2) -> Dropout -> Linear(128 -> 1) -> RUL
```

**Hyperparameters:** Identical to LSTM — intentional so the comparison is fair (only the cell type differs).

---

#### CNN-1D Regressor

**What it is:** 1D Convolutional Neural Network applied along the time dimension. Extracts **local patterns** via sliding filters.

**Why we use it:**
- Captures local temporal patterns (e.g. sharp spikes, slope changes over 3 cycles).
- **Much faster** to train than LSTM/GRU — no sequential dependencies.
- Two layers allow hierarchical pattern extraction.

**Architecture:**
```
Input (batch, seq_len, features)
  -> permute to (batch, features, seq_len)
  -> Conv1d(input_size -> 64, kernel=3, padding=1) -> ReLU
  -> Conv1d(64 -> 128, kernel=3, padding=1) -> ReLU
  -> AdaptiveAvgPool1d(1)  <- global average pooling
  -> Dropout(0.2)
  -> Linear(128 -> 1) -> RUL
```

**Hyperparameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `num_filters` | 64 | Conv filters in layer 1 (doubled in layer 2) |
| `kernel_size` | 3 | Local window = 3 consecutive cycles per filter |
| `dropout` | 0.2 | Before final FC layer |
| `epochs`, `batch_size`, `lr` | Same as LSTM | Fair comparison |

---

#### Transformer Regressor

**What it is:** Lightweight Transformer encoder using **multi-head self-attention** over the 30-cycle sequence. Every timestep can directly attend to every other timestep.

**Why we use it:**
- Captures **global** temporal dependencies — not limited by recurrence.
- State-of-the-art in many time-series tasks; included as the most modern DL baseline.

**Architecture:**
```
Input -> Linear(input_size -> d_model=64)
  -> TransformerEncoder(d_model=64, nhead=4, num_layers=2, dropout=0.1, batch_first=True)
  -> permute -> AdaptiveAvgPool1d(1)
  -> Dropout(0.1)
  -> Linear(64 -> 1) -> RUL
```

**Hyperparameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `d_model` | 64 | Attention embedding dimension |
| `nhead` | 4 | Attention heads (64/4 = 16 dims per head) |
| `num_layers` | 2 | Stacked encoder layers |
| `dropout` | 0.1 | Lower than RNNs |
| `epochs`, `batch_size`, `lr` | Same as LSTM | Fair comparison |

**Why d_model=64 (not 128)?** Transformers have quadratic attention complexity. At seq_len=30 this is fine, but smaller d_model prevents overparameterisation on ~10k training samples.

---

## 5. Task B — Anomaly Detection Models

**Label definition:** `anomaly = 1` if `RUL <= 30` cycles.
Class distribution: ~15% positive, ~85% negative — imbalanced but manageable.

---

### 5.1 Unsupervised ML Models

These models do **not use anomaly labels** during training. They learn only from normal (healthy) data.

---

#### Isolation Forest

**What it is:** Ensemble of random decision trees that **isolates** anomalies. Anomalies are rare and different, so they get isolated in fewer splits. Short path length = high anomaly score.

**Why we use it:**
- No labels needed — trains purely on nominal sensor readings.
- Works well in high-dimensional feature spaces.
- Extremely fast at inference — suitable for the always-on monitoring loop.
- Classic industry-standard unsupervised anomaly detector.

**Hyperparameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `n_estimators` | 200 | Number of isolation trees |
| `contamination` | 0.05 | Expected fraction of anomalies (score scaling only) |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | All CPU cores |

Threshold: 95th percentile of validation anomaly scores.

---

#### One-Class SVM

**What it is:** SVM variant that fits a **decision boundary around normal data** in a high-dimensional kernel space. Points outside the boundary are anomalies.

**Why we use it:**
- Complements Isolation Forest — different geometric assumption.
- RBF kernel handles non-linear normal data distributions.
- Useful as a second-opinion detector.

**Hyperparameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `kernel` | "rbf" | Radial Basis Function kernel |
| `gamma` | "scale" | Auto-scaled to 1/(n_features * X.var()) |
| `nu` | 0.05 | Upper bound on anomaly fraction |

Note: Training capped at **10,000 samples** from nominal set due to SVM's O(n^2) complexity.

---

### 5.2 Supervised ML Model

---

#### LightGBM Anomaly Classifier

**What it is:** Binary classifier (0=nominal, 1=anomaly) trained on the **full labelled training set** using RUL-derived labels.

**Why we use it:**
- Supervised learning dramatically outperforms unsupervised methods when labels exist.
- `class_weight='balanced'` compensates for the 85/15 imbalance automatically.
- `predict_proba` gives calibrated probability scores for ROC-AUC and PR-AUC.
- Threshold is tuned on validation F1, not defaulted to 0.5.

**Hyperparameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `n_estimators` | 300 | Boosting rounds |
| `learning_rate` | 0.05 | Step size |
| `max_depth` | 8 | Slightly deeper — classification benefits from more interactions |
| `num_leaves` | 63 | Leaf capacity (<=2^max_depth) |
| `subsample` | 0.8 | Row sampling |
| `colsample_bytree` | 0.8 | Feature sampling |
| `class_weight` | "balanced" | Upweights minority anomaly class |
| `best_threshold` | tuned on val F1 | Swept 0.10-0.90 in 0.05 steps |

---

### 5.3 DL Autoencoder Models

Autoencoders train **only on nominal (healthy) data**. At inference, anomalous data has high reconstruction error because the model has never seen degraded patterns.

---

#### Feed-Forward Autoencoder

**What it is:** Bottleneck neural network — Encoder compresses input to latent vector; Decoder reconstructs original. High reconstruction error = anomaly.

**Why we use it:**
- Fast, effective for flat feature vectors.
- Bottleneck forces learning of a compressed healthy-behaviour representation.
- Per-feature reconstruction error can identify which sensors are misbehaving.

**Architecture:**
```
Encoder: Input -> Linear(N, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 16) -> ReLU
Decoder: Latent(16) -> Linear(16, 32) -> ReLU -> Linear(32, 64) -> ReLU -> Linear(64, N)
Loss: MSELoss (reconstruction error on nominal training data only)
```

**Hyperparameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `encoding_dim` | 16 | Latent space size |
| `epochs` | 50 | Training passes |
| `batch_size` | 64 | Samples per update |
| `lr` | 1e-3 | Adam learning rate |
| Anomaly threshold | 95th percentile | Set on reconstruction errors |

---

#### LSTM Autoencoder

**What it is:** Same concept as the feed-forward autoencoder but operating on **sequences**. LSTM encoder compresses a 30-cycle sequence to a hidden state; decoder reconstructs the full sequence.

**Why we use it:**
- Captures **temporal anomalies** — not just that a sensor is out of range, but that its *pattern over time* is abnormal.
- Reconstruction error computed over all timesteps and features: `mean((recon - input)^2 over time and features)`.

**Architecture:**
```
Encoder LSTM: (input_size, hidden=64, 1 layer) -> last hidden state h
Decoder LSTM: repeat h for seq_len timesteps -> (hidden=64, 1 layer) -> Linear(64 -> input_size)
Loss: MSELoss over all (timestep, feature) positions
```

**Hyperparameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `hidden_size` | 64 | Encoder/decoder hidden state |
| `num_layers` | 1 | Single LSTM layer (enough for short sequences) |
| `epochs` | 50 | Training passes |
| `batch_size` | 64 | Sequences per update |
| `lr` | 1e-3 | Adam learning rate |
| Anomaly threshold | 95th percentile | Set on reconstruction errors |

---

## 6. Hyperparameter Tuning Strategy

Three methods are implemented and compared:

### Method 1: Grid Search (GridSearchCV)
- Exhaustively tests **all combinations** in a discrete parameter grid.
- Uses `GroupKFold` so all cycles from one engine stay in one fold — prevents temporal leakage.
- Best for small grids.

### Method 2: Randomized Search (RandomizedSearchCV)
- Randomly samples `n_iter` combinations from a wider distribution.
- Faster than Grid Search for large parameter spaces.

### Method 3: Optuna (Bayesian Optimization) — RECOMMENDED
- Uses Tree-structured Parzen Estimator (TPE) to intelligently sample the space.
- Each trial informs the next — focuses search on promising regions.
- Best result in fewer trials than random search.
- Every trial logged as a nested MLflow run.

### Anti-Leakage Guarantees

```python
# 1. Pipeline ensures scaler only sees training fold data:
Pipeline([("scaler", StandardScaler()), ("model", estimator)])

# 2. GroupKFold: engine unit IDs are groups — no engine appears in both train and val:
GroupKFold(n_splits=5)

# 3. Forbidden features that contain future information (raises ValueError if present):
FORBIDDEN_FEATURES = {
    "cycle_norm", "normalized_cycle", "life_fraction",
    "percentage_life_used", "max_cycle", "final_cycle"
}
```

### Tuning Objective
- **RUL models:** Minimise RMSE on validation set.
- **Anomaly models:** Maximise F1 on validation set (threshold swept 0.1-0.9 in 0.05 steps).
- **Test set is never seen during tuning** — final evaluation only.

---

## 7. Evaluation Metrics

### RUL Regression Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **RMSE** | sqrt(mean((y_hat - y)^2)) | Primary metric — average error in cycles |
| **MAE** | mean(abs(y_hat - y)) | Less sensitive to outlier predictions |
| **R2** | 1 - SS_res/SS_tot | Proportion of variance explained |
| **Adjusted R2** | 1 - (1-R2)(n-1)/(n-k-1) | Penalises extra features |
| **MAPE** | mean(abs(y_hat - y)/y) x 100 | Percentage error |
| **NASA Score** | Asymmetric exponential penalty | **PHM08 official benchmark metric** |

**NASA Score — why it matters:**
```
delta = y_hat - y
score = exp(-delta/13) - 1   if delta < 0  (early prediction, less severe)
score = exp(delta/10) - 1    if delta >= 0  (late prediction, MORE severe)
```
Lower is better. Late predictions (underestimating RUL) are penalised more than early ones — matches the real cost: missing a failure is much worse than over-maintaining.

### Anomaly Detection Metrics

| Metric | Meaning |
|---|---|
| **Precision** | Of all flagged anomalies, how many were real? (Controls false alarms) |
| **Recall** | Of all real anomalies, how many did we catch? (Guards against missed failures) |
| **F1 Score** | Harmonic mean of Precision and Recall — primary metric |
| **ROC-AUC** | Area under Receiver Operating Characteristic curve |
| **PR-AUC** | Area under Precision-Recall curve — better for imbalanced data |
| **False Positive Rate** | How often we cry wolf |
| **False Negative Rate** | How often we miss a real failure (MOST DANGEROUS) |

---

## 8. Model Selection Summary

### For RUL Regression

| Model | Type | Strengths | Use Case |
|---|---|---|---|
| LinearRegression | ML | Interpretable, fast | Sanity check baseline only |
| RandomForest | ML | Robust, feature importance | Quick strong baseline |
| **LightGBM** | ML | **Fastest, accurate** | **Fleet-scale production RUL monitoring** |
| XGBoost | ML | Accurate, benchmark-proven | Challenger to LightGBM |
| CatBoost | ML | Calibrated, handles categoricals | When Ironside data has machine type codes |
| LSTM | DL | Best temporal modelling | On-demand detailed diagnosis |
| GRU | DL | Faster LSTM alternative | Cheap DL monitoring path |
| CNN-1D | DL | Local pattern detection | Fast DL baseline |
| Transformer | DL | Global attention | Frontier comparison |

**Recommended for production:**
- Cheap always-on monitoring: **LightGBM** (fast, low cost per asset)
- On-demand detailed diagnosis: **LSTM** (best temporal accuracy)

### For Anomaly Detection

| Model | Type | Strengths | Use Case |
|---|---|---|---|
| **LightGBM Classifier** | Supervised ML | **Best performance with labels** | Primary detector when training labels exist |
| Isolation Forest | Unsupervised ML | No labels needed, fast | Cold-start / new machine types |
| One-Class SVM | Unsupervised ML | Different inductive bias | Second-opinion detector |
| Autoencoder | Unsupervised DL | Learns healthy distribution | When feature interactions matter |
| LSTM Autoencoder | Unsupervised DL | Sequential anomaly detection | When temporal patterns define normality |

**Recommended for production:**
- Primary: **LightGBM Classifier** (supervised, best F1/ROC-AUC)
- Fallback (new assets, no labels): **Isolation Forest** (no labels needed, fast)

---

*This guide reflects the code in:*
- `src/train_rul_models.py` — all RUL model definitions and training
- `src/train_anomaly_models.py` — all anomaly model definitions and training
- `src/hyperparameter_tuning.py` — Grid/Random/Optuna tuning with anti-leakage guarantees
- `src/feature_engineering.py` — rolling features, delta features, sequence windows
- `src/preprocessing.py` — sensor selection, normalisation
- `src/evaluate.py` — all evaluation metrics including NASA Score
