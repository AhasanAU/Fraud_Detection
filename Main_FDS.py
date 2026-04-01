"""
Main_FDS.py
===========
Entry point for the Hybrid Fraud Detection System.

Usage
-----
  # Run full pipeline (from project root):
  python Main_FDS.py

  # Skip Node2Vec (use existing checkpoint or zero embeddings):
  python Main_FDS.py --skip-node2vec

  # Only run evaluation on a saved model:
  python Main_FDS.py --only-eval

  # Use a custom config file:
  python Main_FDS.py --config my_config.yaml

  # Load and use the saved model later:
  >>> from src.hybrid_model import HybridPredictor
  >>> predictor = HybridPredictor.load('results/models/HYBRID_FINAL_MODEL.joblib')
  >>> y_pred, y_proba = predictor.predict(X_new)

Project Structure
-----------------
  src/data_loader.py          ← Step 01: load & merge dataset
  src/graph_builder.py        ← Step 02: NetworkX graph + SNA features
  src/node2vec_embeddings.py  ← Step 03: Node2Vec (50 walks × length 20)
  src/temporal_features.py    ← Step 04: time-step feature engineering
  src/community_detection.py  ← Step 05: Louvain community features
  src/feature_fusion.py       ← Step 06: 5-stream fusion + RFECV selection
  src/imbalance_handler.py    ← Step 07: SMOTE-ENN resampling
  src/base_models.py          ← Step 08: XGB/LGB/CB/RF/ET/SVM with OOF
  src/stacking_ensemble.py    ← Step 09: MLP + LogReg meta-learner
  src/hybrid_model.py         ← Step 10: Hybrid assembly + threshold opt
  src/hyperparameter_tuning.py← Step 11: Optuna Bayesian HPT
  src/evaluation.py           ← Step 12: Metrics + 10 publication figures
  src/pipeline.py             ← Orchestration + checkpointing
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline, parse_args

if __name__ == "__main__":
    args = run_pipeline.__code__  # trigger load check
    args = parse_args()
    predictor, metrics = run_pipeline(config_path=args.config, args=args)
