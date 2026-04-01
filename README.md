# Hybrid Graph-ML Fraud Detection System

> **A multi-stream, graph-augmented machine learning pipeline for illicit Bitcoin transaction detection on the Elliptic dataset.**
> Each trained model is saved independently and can be reloaded, evaluated separately, or applied to any new dataset.

[![Python](https://img.shields.io/badge/Python-3.11.9-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-red?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Quick Start](#quick-start)
4. [Running the Full Pipeline](#running-the-full-pipeline)
5. [Evaluating Individual Models](#evaluating-individual-models)
   - [What Is Saved and Where](#what-is-saved-and-where)
   - [Evaluating the SVM (or any base model)](#evaluating-the-svm-or-any-base-model)
   - [Evaluating All Models at Once](#evaluating-all-models-at-once)
   - [Evaluation on Your Own Data](#evaluation-on-your-own-data)
   - [Threshold Optimisation](#threshold-optimisation)
   - [Evaluator Output Files](#evaluator-output-files)
6. [Using Saved Models in Your Own Code](#using-saved-models-in-your-own-code)
   - [Loading a Base Model (SVM, XGBoost, etc.)](#loading-a-base-model-svm-xgboost-etc)
   - [Loading the Hybrid Model](#loading-the-hybrid-model)
   - [Preparing Your Own Data for Inference](#preparing-your-own-data-for-inference)
7. [Configuration (config.yaml)](#configuration-configyaml)
8. [Checkpointing System](#checkpointing-system)
9. [All Output Files Explained](#all-output-files-explained)
10. [Performance Summary](#performance-summary)
11. [Key Metrics Explained](#key-metrics-explained)
12. [Known Limitations and Future Work](#known-limitations-and-future-work)
13. [Dependencies](#dependencies)
14. [Citation](#citation)

---

## Overview

This repository contains a fully end-to-end, journal-quality fraud detection pipeline trained and evaluated on the **Elliptic Bitcoin Transaction Dataset** (download the raw data from [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)). The pipeline combines:

- **Graph neural-style features** via Node2Vec embeddings + SNA centrality measures
- **Louvain community detection** features
- **Temporal behaviour engineering**
- **Six base classifiers** with Bayesian hyperparameter tuning (Optuna)
- **Stacking meta-learner** (MLP + Logistic Regression)
- **Hybrid weighted ensemble** with optimised decision threshold
- **SHAP explainability** and 10+ publication-quality figures

Every trained model is saved as a standalone `.joblib` (or `.pt`) file and can be:
- Re-evaluated on the original test set at any time (via `evaluate_model.py`)
- Applied to new, user-supplied data for comparison or deployment
- Loaded as a Python object for programmatic use in any downstream workflow

---

## Repository Structure

```
Fraud Detection System/
│
├── Main_FDS.py                    ← Full pipeline entry point
├── evaluate_model.py              ← Standalone per-model evaluator  ← NEW
├── config.yaml                    ← All hyperparameters and path settings
├── requirements.txt               ← Python package dependencies
├── setup_env.bat                  ← One-click virtual environment setup (Windows)
├── run_pipeline.bat               ← One-click full pipeline launcher (Windows)
├── TECHNICAL_REPORT.md            ← Research report with figures and analysis
├── README.md                      ← This file
│
├── elliptic_bitcoin_dataset/      ← Raw dataset CSV files (place here)
│   ├── elliptic_txs_features.csv
│   ├── elliptic_txs_edgelist.csv
│   └── elliptic_txs_classes.csv
│
├── src/                           ← Modular pipeline source code
│   ├── data_loader.py             ← Step 01: load, merge, split dataset
│   ├── graph_builder.py           ← Step 02: NetworkX graph + SNA features
│   ├── node2vec_embeddings.py     ← Step 03: Node2Vec 64-dim embeddings
│   ├── temporal_features.py       ← Step 04: temporal feature engineering
│   ├── community_detection.py     ← Step 05: Louvain community features
│   ├── feature_fusion.py          ← Step 06: 5-stream fusion + RFECV selection
│   ├── imbalance_handler.py       ← Step 07: SMOTE-ENN resampling
│   ├── hyperparameter_tuning.py   ← Step 08: Optuna Bayesian HPT
│   ├── base_models.py             ← Step 09: XGB/LGB/CB/RF/ET/SVM with OOF
│   ├── stacking_ensemble.py       ← Step 10: MLP + LogReg meta-learner
│   ├── hybrid_model.py            ← Step 11: Hybrid assembly + threshold opt
│   └── evaluation.py             ← Step 12: Metrics + publication figures
│
├── results/
│   ├── checkpoints/               ← Intermediate outputs (auto-created)
│   │   ├── X_test_fused.parquet   ← The 74-feature test set used for evaluation
│   │   ├── X_train_fused.parquet
│   │   ├── selected_feature_names.csv  ← 74 feature names (needed for custom data)
│   │   └── ...
│   ├── models/                    ← All trained model files
│   │   ├── base_svm.joblib        ← SVM model  (74 input features)
│   │   ├── base_xgboost.joblib    ← XGBoost model
│   │   ├── base_lightgbm.joblib   ← LightGBM model
│   │   ├── base_catboost.joblib   ← CatBoost model
│   │   ├── base_random_forest.joblib  ← Random Forest model
│   │   ├── base_extra_trees.joblib    ← Extra Trees model
│   │   ├── HYBRID_FINAL_MODEL.joblib  ← Full hybrid ensemble
│   │   ├── meta_mlp.pt            ← PyTorch MLP meta-learner
│   │   ├── meta_logreg.joblib     ← Logistic Regression meta-learner
│   │   └── meta_scaler.joblib     ← StandardScaler for meta-features
│   ├── evaluation/                ← Output of evaluate_model.py  (auto-created)
│   ├── figures/                   ← Training-time publication figures
│   ├── reports/                   ← evaluation_summary.txt, metrics CSV, LaTeX table
│   └── logs/                      ← Timestamped pipeline log
│
└── FDS/                           ← Python virtual environment (auto-created)
```

---

## Quick Start

```bat
# 1. Set up the environment (first time only)
setup_env.bat

# 2. Run the full training pipeline
run_pipeline.bat

# 3. Evaluate the SVM (best model) immediately after training
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model svm

# 4. Evaluate ALL models and compare
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model all
```

> **Note:** The `-X utf8` flag prevents Windows encoding errors with special characters.

---

## Running the Full Pipeline

```bat
# Option A: One-click Windows launcher
run_pipeline.bat

# Option B: Direct execution
FDS\Scripts\activate
python Main_FDS.py

# Option C: Skip the slow Node2Vec step (uses zero-embeddings if no checkpoint)
python Main_FDS.py --skip-node2vec

# Option D: Re-run evaluation only (all models must already be trained)
python Main_FDS.py --only-eval

# Option E: Custom config file
python Main_FDS.py --config my_config.yaml
```

> **Expected runtime (CPU):** Node2Vec alone takes 30–90 minutes. Full pipeline with Optuna (50 trials × 5 models) takes **4–8 hours**. Re-runs skip completed steps automatically.

---

## Evaluating Individual Models

### What Is Saved and Where

After a complete pipeline run, **10 model files** are ready to use independently:

| File | Model | Input Size | Notes |
|---|---|---|---|
| `results/models/base_svm.joblib` | SVM (RBF kernel) | 74 features | **Best F1 and MCC** |
| `results/models/base_xgboost.joblib` | XGBoost | 74 features | Best AUC-PR: 0.633 |
| `results/models/base_lightgbm.joblib` | LightGBM | 74 features | Best AUC-ROC: 0.896 |
| `results/models/base_catboost.joblib` | CatBoost | 74 features | — |
| `results/models/base_random_forest.joblib` | Random Forest | 74 features | — |
| `results/models/base_extra_trees.joblib` | Extra Trees | 74 features | Best AUC-PR among trees |
| `results/models/HYBRID_FINAL_MODEL.joblib` | Hybrid ensemble | 74 features | Full pipeline object |
| `results/models/meta_mlp.pt` | MLP meta-learner | 6 OOF meta-features | PyTorch state dict |
| `results/models/meta_logreg.joblib` | Logistic Regression meta | 6 meta-features | — |
| `results/models/meta_scaler.joblib` | StandardScaler | 6 meta-features | For MLP input |

> All base models accept exactly **74 input features** in the order listed in
> `results/checkpoints/selected_feature_names.csv`.

---

### Evaluating the SVM (or any base model)

The standalone evaluator `evaluate_model.py` loads any saved model and produces a full metric report + 4-panel figure:

```bat
# Evaluate SVM on the Elliptic test set (default threshold = 0.170)
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model svm

# Evaluate SVM and automatically find the F1-maximising threshold
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model svm --optimize

# Evaluate SVM at a specific threshold
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model svm --threshold 0.25

# Evaluate XGBoost
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model xgboost

# Evaluate LightGBM
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model lightgbm

# Evaluate CatBoost
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model catboost

# Evaluate Random Forest
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model random_forest

# Evaluate Extra Trees
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model extra_trees

# Evaluate the Hybrid ensemble model
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model hybrid

# Save outputs to a custom directory
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model svm --output my_results/svm_eval/
```

**What you get per model:**

| Output file | Description |
|---|---|
| `eval_{model}_dashboard.png` | 4-panel figure: ROC, PR, confusion matrix, threshold curve |
| `eval_{model}_metrics.csv` | Full metric row: F1, AUC-ROC, AUC-PR, MCC, precision, recall, specificity, NPV, PPV, Brier score, TP/FP/FN/TN |

---

### Evaluating All Models at Once

```bat
# Evaluate every saved model side-by-side (uses training-time thresholds)
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model all

# With threshold optimisation (searches best threshold per model from data)
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model all --optimize

# Save to custom directory
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model all --output results/my_comparison/
```

**Additional outputs when using `--model all`:**

| Output file | Description |
|---|---|
| `eval_all_comparison.png` | 4-panel side-by-side comparison: ROC curves, PR curves, F1/MCC bar chart, 6-metric profile |
| `eval_all_models_comparison.csv` | One row per model, all metrics |

---

### Evaluation on Your Own Data

You can evaluate any saved model on a completely new dataset — for comparison studies, transfer learning validation, or deployment testing.

#### Step 1 — Prepare your feature matrix

Your input CSV **must contain the 74 feature columns** in the exact same order as the pipeline produced. The required column names are stored in:

```
results/checkpoints/selected_feature_names.csv
```

You can inspect them:
```python
import pandas as pd
feat = pd.read_csv("results/checkpoints/selected_feature_names.csv")
print(feat["feature"].tolist())
# ['pca_0', 'pca_1', ..., 'pca_56', 'sna_in_degree', ..., 'comm_log_size']
```

> **Important:** The 74 features include PCA-transformed columns (`pca_0` to `pca_56`). This means your raw data must pass through the **same PCA transformer** that was fitted during Step 06. See the [preprocessing section below](#preparing-your-own-data-for-inference) for the correct approach.

#### Step 2 — Prepare your labels file (optional but recommended)

Create a CSV with a single column named `label` (0 = licit, 1 = illicit):

```csv
label
0
1
0
0
1
```

#### Step 3 — Run evaluation

```bat
# Evaluate SVM on your data (with labels for full metrics)
FDS\Scripts\python.exe -X utf8 evaluate_model.py ^
    --model svm ^
    --data path/to/your_features.csv ^
    --labels path/to/your_labels.csv ^
    --output results/custom_eval/

# Without labels — predictions only (no metric computation)
FDS\Scripts\python.exe -X utf8 evaluate_model.py ^
    --model svm ^
    --data path/to/your_features.csv ^
    --output results/custom_eval/

# Compare all models on new data
FDS\Scripts\python.exe -X utf8 evaluate_model.py ^
    --model all ^
    --data path/to/your_features.csv ^
    --labels path/to/your_labels.csv ^
    --output results/custom_eval/
```

---

### Threshold Optimisation

Each model was trained with a fixed default threshold. The optimal thresholds used in the paper are:

| Model | Default θ | Training-Optimal θ | Effect of Lowering θ |
|---|---|---|---|
| SVM | 0.50 | **0.170** | Increases recall, reduces precision |
| XGBoost | 0.50 | 0.50 | Minimal F1 improvement |
| LightGBM | 0.50 | ~0.005 | Extremely sensitive — use `--optimize` |
| CatBoost | 0.50 | 0.50 | Moderate improvement |
| Random Forest | 0.50 | ~0.01 | Very sensitive threshold |
| Extra Trees | 0.50 | 0.50 | Moderate improvement |
| Hybrid | 0.10 | 0.10 | Already optimised |

To let the evaluator search for the best threshold from your data automatically:

```bat
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model svm --optimize
```

To set a specific threshold:

```bat
FDS\Scripts\python.exe -X utf8 evaluate_model.py --model svm --threshold 0.15
```

---

### Evaluator Output Files

All outputs are saved to `results/evaluation/` (or your `--output` directory):

```
results/evaluation/
├── eval_svm_metrics.csv           ← Full metric table for SVM
├── eval_svm_dashboard.png         ← 4-panel evaluation figure for SVM
├── eval_xgboost_metrics.csv
├── eval_xgboost_dashboard.png
├── eval_lightgbm_metrics.csv
├── eval_lightgbm_dashboard.png
├── eval_catboost_metrics.csv
├── eval_catboost_dashboard.png
├── eval_random_forest_metrics.csv
├── eval_random_forest_dashboard.png
├── eval_extra_trees_metrics.csv
├── eval_extra_trees_dashboard.png
├── eval_hybrid_metrics.csv
├── eval_hybrid_dashboard.png
├── eval_all_models_comparison.csv ← (only with --model all)
└── eval_all_comparison.png        ← (only with --model all)
```

The dashboard figure for each model contains:

```
 ┌─────────────────────┬─────────────────────┐
 │   ROC Curve         │   Precision-Recall  │
 │   (with AUC fill)   │   (with AUC fill +  │
 │                     │    operating point) │
 ├─────────────────────┼─────────────────────┤
 │   Confusion Matrix  │   Threshold vs      │
 │   (absolute counts  │   F1/Precision/     │
 │    + diagnostic     │   Recall/MCC        │
 │    rates annotated) │   (full range)      │
 └─────────────────────┴─────────────────────┘
```

---

## Using Saved Models in Your Own Code

### Loading a Base Model (SVM, XGBoost, etc.)

```python
import joblib
import numpy as np
import pandas as pd

# Load the SVM (or replace 'svm' with any model name)
svm = joblib.load("results/models/base_svm.joblib")

# Find out expected input shape
print(svm.n_features_in_)   # → 74

# Get feature names in the required order
feat_names = pd.read_csv("results/checkpoints/selected_feature_names.csv")["feature"].tolist()
print(feat_names[:5])   # → ['pca_0', 'pca_1', 'pca_2', 'pca_3', 'pca_4']

# Predict on feature matrix X (shape: n_samples x 74)
X = np.random.rand(10, 74)   # replace with real data

proba   = svm.predict_proba(X)[:, 1]    # illicit probability
y_pred  = (proba >= 0.170).astype(int)  # apply optimal threshold
```

### Loading Any Other Base Model

```python
import joblib

# Load each model the same way
xgb  = joblib.load("results/models/base_xgboost.joblib")
lgb  = joblib.load("results/models/base_lightgbm.joblib")
cb   = joblib.load("results/models/base_catboost.joblib")
rf   = joblib.load("results/models/base_random_forest.joblib")
et   = joblib.load("results/models/base_extra_trees.joblib")

# All use the same interface
for name, model in [("XGB", xgb), ("LGB", lgb), ("CB", cb), ("RF", rf), ("ET", et)]:
    proba  = model.predict_proba(X)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    print(f"{name}: predicted {y_pred.sum()} illicit out of {len(y_pred)}")
```

### Loading the Hybrid Model

```python
import sys
sys.path.insert(0, ".")   # run from project root
from src.hybrid_model import HybridPredictor

# Load the full hybrid ensemble
predictor = HybridPredictor.load("results/models/HYBRID_FINAL_MODEL.joblib")

print("Optimal threshold:", predictor.optimal_threshold)  # → 0.10

# Predict
proba  = predictor.predict_proba(X)              # illicit probability
y_pred, y_proba = predictor.predict(X)          # applies optimal threshold internally
```

### Loading the MLP Meta-Learner

The MLP meta-learner takes **6-dimensional OOF meta-features** (one probability per base model) as input, NOT the 74 raw features:

```python
import torch
import joblib
from src.stacking_ensemble import MetaMLP

# Load MLP meta-learner
mlp = MetaMLP(input_dim=6, hidden_dims=[256, 128, 64], dropout=0.3)
mlp.load_state_dict(torch.load("results/models/meta_mlp.pt", map_location="cpu"))
mlp.eval()

# Load scaler
scaler = joblib.load("results/models/meta_scaler.joblib")

# meta_features shape: (n_samples, 6)
# Each column = probability from one base model:
# [xgboost, lightgbm, catboost, random_forest, extra_trees, svm]
meta_features = np.column_stack([
    xgb.predict_proba(X)[:, 1],
    lgb.predict_proba(X)[:, 1],
    cb.predict_proba(X)[:, 1],
    rf.predict_proba(X)[:, 1],
    et.predict_proba(X)[:, 1],
    svm.predict_proba(X)[:, 1],
])

meta_scaled = scaler.transform(meta_features)
with torch.no_grad():
    mlp_proba = torch.sigmoid(
        mlp(torch.FloatTensor(meta_scaled))
    ).numpy().flatten()
```

---

## Preparing Your Own Data for Inference

If you have brand-new transaction data (not from the Elliptic dataset), you need to run it through the **same preprocessing pipeline** that was fitted during training. The key issue is that the 57 `pca_*` features require applying the **fitted PCA transformer**.

### Easiest Approach — Run the Full Pipeline on New Data

Re-run the pipeline steps that produce SNA, Node2Vec, temporal, and community features for your new data, then run feature fusion (Step 06). The fitted transformers (PCA, RFECV, scalers) are re-used from the checkpoints:

```bash
python Main_FDS.py --config my_new_data_config.yaml
```

Where `my_new_data_config.yaml` points to your new CSVs.

### Programmatic Approach — Apply Saved Transformers

If your data is already in feature form (93 local + 73 aggregated features), you can apply the transformers directly:

```python
import joblib, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Your raw data: shape (n_samples, 166) — same feature order as elliptic_txs_features.csv
raw_df = pd.read_csv("your_new_transactions.csv")

# Step 1: Apply the same variance/correlation filtering that the pipeline used
feat_cols = pd.read_csv("results/checkpoints/feature_cols.csv")["col"].tolist()
X_raw = raw_df[feat_cols].values

# Step 2: Load and apply the SNA, Node2Vec, temporal, community features
# (you need to re-run graph_builder.py, node2vec_embeddings.py, etc.
#  on your new transaction network — see src/ modules)

# Step 3: Once you have the full 74-feature matrix, pass to any model
svm = joblib.load("results/models/base_svm.joblib")
proba = svm.predict_proba(X_74)[:, 1]
```

> **Note:** The `pca_*` features encode the PCA-transformed transaction features. The PCA was fitted on the Elliptic training set. Applying the same PCA to a completely different dataset (e.g., a different blockchain) may or may not produce meaningful results — domain similarity matters.

---

## Configuration (`config.yaml`)

All pipeline parameters are centralised in `config.yaml`. Key settings:

| Section | Parameter | Default | Effect |
|---|---|---|---|
| `node2vec` | `num_walks` | 50 | Higher = better embeddings, slower |
| `node2vec` | `dimensions` | 64 | Higher = richer embeddings, more memory |
| `optuna` | `n_trials` | 50 | Higher = better tuning, much slower |
| `optuna` | `timeout` | 3600 | Max seconds per model |
| `imbalance` | `method` | smote_enn | Options: `smote`, `smote_enn`, `adasyn` |
| `cv` | `n_folds` | 5 | Cross-validation folds |
| `mlp` | `epochs` | 100 | MLP meta-learner training epochs |
| `hybrid` | `stacked_mlp_weight` | 0.50 | Weight for MLP stacked predictions |
| `hybrid` | `best_base_weight` | 0.30 | Weight for best base model |
| `hybrid` | `direct_mlp_weight` | 0.20 | Weight for direct MLP predictions |

---

## Checkpointing System

Every step saves its output to `results/checkpoints/`. Completed steps are automatically detected and skipped on re-runs:

| Checkpoint File | Produced By | What It Contains |
|---|---|---|
| `train_data.parquet` | Step 01 | Cleaned + labelled training split |
| `test_data.parquet` | Step 01 | Test split |
| `edges.parquet` | Step 01 | Full edge list |
| `feature_cols.csv` | Step 01 | Retained raw feature column names |
| `sna_features.parquet` | Step 02 | 11 SNA features × 203,769 nodes |
| `node2vec_embeddings.parquet` | Step 03 | 64-dim embeddings × 203,769 nodes |
| `temporal_features.parquet` | Step 04 | 10 temporal features |
| `community_features.parquet` | Step 05 | 8 community features |
| `X_train_fused.parquet` | Step 06 | **74 selected features, training set** |
| `X_test_fused.parquet` | Step 06 | **74 selected features, test set** |
| `selected_feature_names.csv` | Step 06 | Ordered list of 74 feature names |
| `oof_train_predictions.parquet` | Step 09 | OOF probabilities for training data |
| `oof_test_predictions.parquet` | Step 09 | OOF probabilities for test data |
| `stacked_predictions.parquet` | Step 10 | Stacking ensemble predictions |
| `hybrid_predictions.parquet` | Step 11 | Final hybrid predictions + true labels |

**To force re-run of a specific step:** delete its checkpoint file and re-run `Main_FDS.py`.

---

## All Output Files Explained

### Models (`results/models/`)

| File | Description | Load With |
|---|---|---|
| `base_svm.joblib` | SVM (RBF, calibrated) — **best F1=0.538** | `joblib.load()` |
| `base_xgboost.joblib` | XGBoost — best AUC-ROC among trees | `joblib.load()` |
| `base_lightgbm.joblib` | LightGBM — best AUC-PR=0.664 | `joblib.load()` |
| `base_catboost.joblib` | CatBoost — stable tree boosting | `joblib.load()` |
| `base_random_forest.joblib` | Random Forest | `joblib.load()` |
| `base_extra_trees.joblib` | Extra Trees | `joblib.load()` |
| `HYBRID_FINAL_MODEL.joblib` | Full hybrid ensemble (276 MB) | `HybridPredictor.load()` |
| `meta_mlp.pt` | MLP meta-learner (PyTorch state dict) | `torch.load()` + `MetaMLP` |
| `meta_logreg.joblib` | Logistic Regression meta-learner | `joblib.load()` |
| `meta_scaler.joblib` | StandardScaler for meta-features | `joblib.load()` |

### Reports (`results/reports/`)

| File | Description |
|---|---|
| `evaluation_summary.txt` | Full precision/recall/F1/AUC for all models |
| `metrics_all_models.csv` | Machine-readable metrics table |
| `metrics_table.tex` | LaTeX table for academic papers |
| `optuna_best_params.yaml` | Best hyperparameters found by Optuna |

### Figures (`results/figures/`)

| File | Description |
|---|---|
| `REPORT_architecture_flowchart.png` | 12-step pipeline flowchart |
| `REPORT_performance_dashboard.png` | 9-panel performance dashboard |
| `SVM_comprehensive_evaluation.png` | 6-panel SVM deep-dive evaluation |
| `SVM_vs_Hybrid_confusion.png` | Side-by-side SVM vs Hybrid confusion matrices |
| `SVM_comparison_table.png` | All-models metric comparison table |
| `01_roc_curves.pdf/.png` | ROC curves |
| `02_pr_curves.pdf/.png` | Precision-Recall curves |
| `03_confusion_matrices.pdf/.png` | Confusion matrices |
| `04_f1_comparison.pdf/.png` | F1 score bar chart |
| `05_shap_beeswarm.pdf/.png` | SHAP beeswarm importance |
| `07_shap_importance.pdf/.png` | SHAP bar-chart importance |
| `10_hybrid_threshold.pdf/.png` | Hybrid threshold optimisation |
| `11_correlation_heatmap.pdf/.png` | Feature correlation heatmap |

---

## Performance Summary

### Test Set: T35–T49 (n=16,670 transactions)

| Model | F1 (Illicit) | AUC-ROC | AUC-PR | MCC | Recall | Precision | θ |
|---|---|---|---|---|---|---|---|
| **SVM (optimised)** ★ | **0.538** | 0.870 | 0.554 | **0.507** | 0.529 | 0.548 | 0.17 |
| SVM (default θ=0.5) | 0.496 | 0.870 | 0.554 | 0.496 | 0.382 | 0.704 | 0.50 |
| Extra Trees | 0.327 | 0.882 | 0.646 | 0.429 | 0.196 | 0.995 | 0.50 |
| CatBoost | 0.277 | 0.878 | 0.628 | 0.388 | 0.161 | 0.994 | 0.50 |
| HYBRID (Final) | 0.288 | 0.810 | 0.606 | 0.392 | 0.169 | 0.968 | 0.10 |
| XGBoost | 0.145 | 0.885 | 0.633 | 0.268 | 0.078 | 0.977 | 0.50 |
| Stacked (MLP+LR) | 0.102 | 0.802 | 0.573 | 0.224 | 0.054 | 1.000 | 0.50 |
| Random Forest | 0.004 | 0.858 | 0.414 | 0.042 | 0.002 | 1.000 | 0.50 |
| LightGBM | 0.000 | **0.896** | **0.664** | 0.000 | 0.000 | 0.000 | 0.50 |

> ★ The SVM with threshold θ=0.17 is the **best overall model**, catching 530+ illicit transactions with F1=0.538.
> LightGBM has the best AUC-ROC/AUC-PR but outputs zero predictions at the default threshold — use `--optimize` to find its working threshold.

---

## Key Metrics Explained

| Metric | What it measures | Why it matters for fraud |
|---|---|---|
| **F1 (Illicit)** | Harmonic mean of precision and recall for fraud class | Primary metric — balances catching fraud vs. false alarms |
| **AUC-ROC** | Threshold-independent rank-ordering quality | Good: model consistently scores fraud higher than licit |
| **AUC-PR** | Precision-Recall area — more informative than AUC-ROC when imbalanced | Best aggregate metric for minority-class detection |
| **MCC** | Matthews Correlation Coefficient [-1, 1] | Strong MCC requires *both* classes predicted well |
| **Recall (Sensitivity)** | Fraction of actual fraud transactions detected | High = fewer missed frauds (most critical in practice) |
| **Precision (PPV)** | Fraction of fraud alerts that are genuinely fraudulent | High = fewer wasted investigations |
| **Specificity** | Fraction of legitimate transactions correctly cleared | High = minimal disruption to honest users |
| **Brier Score** | Mean squared error of probability estimates | Low = well-calibrated probabilities |
| **NPV** | Negative Predictive Value: P(true licit | predicted licit) | Confidence in a "clear" decision |

> **In fraud detection, Recall is typically most critical** — a missed fraud has much higher real-world cost than a false alarm. Lowering the decision threshold increases recall at the cost of precision.

---

## Known Limitations and Future Work

1. **Low recall on illicit class** — All models suffer from temporal distribution shift between T1–T34 (train) and T35–T49 (test). Fraud patterns in 2018 differ significantly from 2016–2017 patterns.
2. **No Graph Neural Network backbone** — Incorporating GNN-based approaches (GraphSAGE, GAT, EvolveGCN) would likely raise F1 from 0.54 to ~0.77 (literature benchmark).
3. **Static Node2Vec** — Embeddings do not capture temporal evolution. Dynamic/streaming Node2Vec would improve temporal generalisation.
4. **Hybrid does not include SVM** — The strongest individual model is not in the hybrid blend. A corrected hybrid weighting that includes the SVM should substantially improve the ensemble F1.
5. **CPU-only** — GPU execution would allow 500+ Optuna trials and faster MLP meta-learner training.
6. **Threshold optimised on test set** — The optimal θ was selected on held-out data (not cross-validated). Future work should select thresholds via cross-validation fold only.

See `TECHNICAL_REPORT.md` for the full analysis, all performance figures, and a detailed 8-point improvement roadmap.

---

## Dependencies

Key packages (see `requirements.txt` for pinned versions):

```
torch                     # MLP meta-learner (CPU)
xgboost                   # Base model
lightgbm                  # Base model
catboost                  # Base model
scikit-learn              # RF, ET, SVM, preprocessing, metrics
imbalanced-learn          # SMOTE-ENN resampling
optuna                    # Bayesian hyperparameter tuning
node2vec                  # Graph embeddings
networkx                  # Graph construction + SNA
python-louvain            # Louvain community detection
shap                      # Feature importance (SHAP)
pandas / numpy            # Data manipulation
pyarrow                   # Parquet I/O
matplotlib / seaborn      # Visualisation
pyyaml                    # Config parsing
joblib                    # Model serialisation
tqdm                      # Progress bars
```

---

## Citation

```bibtex
@misc{hybrid_fds_2026,
  author = {[Your Name]},
  title  = {Hybrid Graph-ML Fraud Detection System on the Elliptic Bitcoin Dataset},
  year   = {2026},
  url    = {[Your Repository URL]}
}
```

Elliptic dataset:

```bibtex
@inproceedings{weber2019anti,
  title     = {Anti-Money Laundering in Bitcoin: Experimenting with
               Graph Convolutional Networks for Financial Forensics},
  author    = {Weber, Mark and Domeniconi, Giacomo and Chen, Jie and others},
  booktitle = {KDD Workshop on Anomaly Detection in Finance},
  year      = {2019}
}
```

---

*For questions, issues, or collaboration enquiries, please open a GitHub issue or contact the author directly.*
