# -*- coding: utf-8 -*-
"""
evaluate_model.py
=================
Standalone evaluation script for any saved model in results/models/.

Supports:
  - Evaluating any single saved model on the original Elliptic test set
  - Evaluating any single saved model on CUSTOM user-supplied data
  - Comparing multiple models side-by-side on any dataset
  - Saving a full metric report and figures for every run

Usage Examples
--------------
  # Evaluate SVM on original Elliptic test set:
  python evaluate_model.py --model svm

  # Evaluate all saved models and compare:
  python evaluate_model.py --model all

  # Evaluate on custom CSV data:
  python evaluate_model.py --model svm --data path/to/your_data.csv --labels path/to/labels.csv

  # Set a custom decision threshold:
  python evaluate_model.py --model svm --threshold 0.17

  # Save results to a custom directory:
  python evaluate_model.py --model all --output my_results/

Available --model choices:
  svm | xgboost | lightgbm | catboost | random_forest | extra_trees | hybrid | all
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import yaml

from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
MODELS    = ROOT / "results" / "models"
CK        = ROOT / "results" / "checkpoints"
FEAT_CSV  = CK  / "selected_feature_names.csv"

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "svm":          "#ff9f43",
    "xgboost":      "#00d4ff",
    "lightgbm":     "#ff6b6b",
    "catboost":     "#ffd93d",
    "random_forest":"#6bcb77",
    "extra_trees":  "#a855f7",
    "hybrid":       "white",
}

# ── Optimal thresholds found during training ──────────────────────────────────
DEFAULT_THRESHOLDS = {
    "svm":          0.170,
    "xgboost":      0.500,
    "lightgbm":     0.500,
    "catboost":     0.500,
    "random_forest":0.500,
    "extra_trees":  0.500,
    "hybrid":       0.100,
}


# =============================================================================
# Model loader
# =============================================================================

def load_model(name: str):
    """
    Load a saved model by name.

    Parameters
    ----------
    name : str
        One of: svm, xgboost, lightgbm, catboost, random_forest, extra_trees, hybrid

    Returns
    -------
    model object with a .predict_proba(X) method
    """
    if name == "hybrid":
        # Hybrid predictor wraps everything internally
        sys.path.insert(0, str(ROOT))
        from src.hybrid_model import HybridPredictor
        predictor = HybridPredictor.load(str(MODELS / "HYBRID_FINAL_MODEL.joblib"))
        return predictor

    path = MODELS / f"base_{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Run the full pipeline first: python Main_FDS.py"
        )
    return joblib.load(path)


# =============================================================================
# Data loader — Elliptic test set (default)
# =============================================================================

def load_elliptic_test():
    """Load the original Elliptic Bitcoin test set from checkpoints."""
    fused = pd.read_parquet(CK / "X_test_fused.parquet")
    feat_names = pd.read_csv(FEAT_CSV)["feature"].tolist()
    X = fused.drop(columns=["txId", "label"]).values
    y = fused["label"].values
    tx_ids = fused["txId"].values
    return X, y, tx_ids, feat_names


# =============================================================================
# Data loader — custom user data
# =============================================================================

def load_custom_data(data_path: str, labels_path: str = None):
    """
    Load a user-supplied feature matrix and optional labels.

    The feature CSV must have exactly the 74 columns listed in
    results/checkpoints/selected_feature_names.csv (in the same order),
    OR it must have a 'txId' column that will be used to align.

    Parameters
    ----------
    data_path : str
        Path to CSV file.  Must contain the 74 selected feature columns.
        May optionally contain a 'txId' column (used for alignment only).
    labels_path : str, optional
        Path to a CSV with columns: txId (or row-order), label (0=licit, 1=illicit).
        If omitted, evaluation metrics cannot be computed — only predictions are returned.

    Returns
    -------
    X      : np.ndarray  shape (n, 74)
    y      : np.ndarray or None
    tx_ids : np.ndarray or None
    feat_names : list[str]
    """
    feat_names = pd.read_csv(FEAT_CSV)["feature"].tolist()
    df = pd.read_csv(data_path)

    # Extract txId if present
    tx_ids = df["txId"].values if "txId" in df.columns else np.arange(len(df))

    # Check feature columns
    missing = [f for f in feat_names if f not in df.columns]
    if missing:
        raise ValueError(
            f"Custom data is missing {len(missing)} required feature columns.\n"
            f"First 5 missing: {missing[:5]}\n\n"
            f"Your data must contain all 74 feature columns listed in:\n"
            f"  {FEAT_CSV}\n\n"
            f"See README.md §'Using Models on New Data' for preprocessing instructions."
        )

    X = df[feat_names].values

    # Load labels
    y = None
    if labels_path:
        ldf = pd.read_csv(labels_path)
        if "label" not in ldf.columns:
            raise ValueError("Labels file must have a 'label' column (0=licit, 1=illicit).")
        y = ldf["label"].values

    return X, y, tx_ids, feat_names


# =============================================================================
# Metric computation
# =============================================================================

def compute_metrics(y_true, y_proba, threshold: float, model_name: str) -> dict:
    """Compute the full metric suite for one model at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    cm  = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    metrics = {
        "model":         model_name,
        "threshold":     threshold,
        "accuracy":      accuracy_score(y_true, y_pred),
        "precision":     precision_score(y_true, y_pred, zero_division=0),
        "recall":        recall_score(y_true, y_pred, zero_division=0),
        "f1_illicit":    f1_score(y_true, y_pred, zero_division=0),
        "f1_macro":      f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted":   f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "auc_roc":       roc_auc_score(y_true, y_proba),
        "auc_pr":        average_precision_score(y_true, y_proba),
        "mcc":           matthews_corrcoef(y_true, y_pred),
        "brier":         brier_score_loss(y_true, y_proba),
        "specificity":   TN / (TN + FP) if (TN + FP) > 0 else 0.0,
        "sensitivity":   TP / (TP + FN) if (TP + FN) > 0 else 0.0,
        "ppv":           TP / (TP + FP) if (TP + FP) > 0 else 0.0,
        "npv":           TN / (TN + FN) if (TN + FN) > 0 else 0.0,
        "tp": int(TP), "fp": int(FP), "fn": int(FN), "tn": int(TN),
    }
    return metrics


# =============================================================================
# Threshold optimizer
# =============================================================================

def find_best_threshold(y_true, y_proba, metric: str = "f1") -> float:
    """Find the threshold in [0.01, 0.99] that maximises the chosen metric."""
    best_t, best_v = 0.5, 0.0
    for t in np.arange(0.01, 1.0, 0.005):
        yp = (y_proba >= t).astype(int)
        if metric == "f1":
            v = f1_score(y_true, yp, zero_division=0)
        elif metric == "mcc":
            v = matthews_corrcoef(y_true, yp)
        else:
            v = f1_score(y_true, yp, zero_division=0)
        if v > best_v:
            best_v, best_t = v, t
    return round(float(best_t), 3)


# =============================================================================
# Single-model evaluation with figures
# =============================================================================

def evaluate_single(name: str, X: np.ndarray, y: np.ndarray,
                    threshold: float = None, output_dir: Path = None,
                    optimize_threshold: bool = False) -> dict:
    """
    Load and fully evaluate one model. Saves a 4-panel figure + CSV report.

    Parameters
    ----------
    name            : model key (svm / xgboost / etc.)
    X               : feature matrix (n_samples, 74)
    y               : true labels — required for metric computation
    threshold       : fixed decision threshold; None uses the training default
    output_dir      : directory to save figures and CSV
    optimize_threshold: if True, find the F1-maximising threshold from data

    Returns
    -------
    dict of metric values
    """
    print(f"\n{'='*64}")
    print(f"  Evaluating: {name.upper()}")
    print(f"{'='*64}")

    model = load_model(name)
    print(f"  Model loaded from results/models/base_{name}.joblib")

    # ── Predict ──────────────────────────────────────────────────────────────
    if name == "hybrid":
        proba = model.predict_proba(X)
    else:
        proba = model.predict_proba(X)[:, 1]

    # ── Threshold ────────────────────────────────────────────────────────────
    if y is not None and optimize_threshold:
        threshold = find_best_threshold(y, proba)
        print(f"  Optimised threshold (F1-maximising): {threshold:.3f}")
    elif threshold is None:
        threshold = DEFAULT_THRESHOLDS.get(name, 0.5)
        print(f"  Using default threshold: {threshold:.3f}")
    else:
        print(f"  Using user-supplied threshold: {threshold:.3f}")

    y_pred = (proba >= threshold).astype(int)

    # ── Metrics ──────────────────────────────────────────────────────────────
    if y is not None:
        metrics = compute_metrics(y, proba, threshold, name)

        print(f"\n  Classification Report:")
        print(classification_report(y, y_pred,
                                    target_names=["Licit (0)", "Illicit (1)"],
                                    digits=4))
        print(f"  AUC-ROC :  {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR  :  {metrics['auc_pr']:.4f}")
        print(f"  MCC     :  {metrics['mcc']:.4f}")
        print(f"  Brier   :  {metrics['brier']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}  |  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  PPV (Precision): {metrics['ppv']:.4f}  |  NPV: {metrics['npv']:.4f}")
        print(f"\n  Confusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = cm.ravel()
        print(f"    TN={TN:,}  FP={FP:,}")
        print(f"    FN={FN:,}  TP={TP:,}")

        # ── Save CSV ─────────────────────────────────────────────────────────
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([metrics]).to_csv(
                out_dir / f"eval_{name}_metrics.csv", index=False
            )
            print(f"\n  Metrics saved: {out_dir}/eval_{name}_metrics.csv")

        # ── Figures ──────────────────────────────────────────────────────────
        _plot_single_model(name, proba, y, y_pred, threshold, metrics, output_dir)

        return metrics
    else:
        # No labels — just return predictions
        pred_df = pd.DataFrame({"proba_illicit": proba, "prediction": y_pred})
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_df.to_csv(out_dir / f"predictions_{name}.csv", index=False)
            print(f"  Predictions saved: {out_dir}/predictions_{name}.csv")
        return {"model": name, "predictions": y_pred, "probabilities": proba}


# =============================================================================
# Plotting helper
# =============================================================================

def _plot_single_model(name, proba, y_true, y_pred, threshold, metrics, output_dir):
    """Generate a 4-panel evaluation figure for one model."""
    colour = PALETTE.get(name, "#58a6ff")

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── (A) ROC ──────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#161b22")
    fpr, tpr, _ = roc_curve(y_true, proba)
    ra = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color=colour, lw=2.5, label=f"AUC-ROC = {ra:.4f}")
    ax1.plot([0, 1], [0, 1], "gray", ls=":", lw=1)
    ax1.fill_between(fpr, tpr, alpha=0.08, color=colour)
    ax1.set_xlabel("False Positive Rate", color="#8b949e", fontsize=9)
    ax1.set_ylabel("True Positive Rate", color="#8b949e", fontsize=9)
    ax1.set_title(f"ROC Curve — {name.upper()}", color="white", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, facecolor="#0d1117", labelcolor="white")
    ax1.tick_params(colors="#8b949e", labelsize=7)
    ax1.spines[:].set_color("#30363d")

    # ── (B) PR ───────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161b22")
    prec, rec, _ = precision_recall_curve(y_true, proba)
    pa = auc(rec, prec)
    ax2.plot(rec, prec, color=colour, lw=2.5, label=f"AUC-PR = {pa:.4f}")
    ax2.fill_between(rec, prec, alpha=0.08, color=colour)
    # Mark operating point
    ax2.scatter([metrics["recall"]], [metrics["precision"]],
                color=colour, s=150, zorder=6, marker="*",
                label=f"θ={threshold:.3f}  (P={metrics['precision']:.3f}, R={metrics['recall']:.3f})")
    ax2.set_xlabel("Recall", color="#8b949e", fontsize=9)
    ax2.set_ylabel("Precision", color="#8b949e", fontsize=9)
    ax2.set_title(f"Precision-Recall Curve — {name.upper()}", color="white", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, facecolor="#0d1117", labelcolor="white")
    ax2.tick_params(colors="#8b949e", labelsize=7)
    ax2.spines[:].set_color("#30363d")

    # ── (C) Confusion Matrix ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#161b22")
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    im = ax3.imshow(cm, cmap="YlOrRd", aspect="auto")
    cell_text = [
        [f"TN = {TN:,}\nSpec = {metrics['specificity']:.4f}", f"FP = {FP:,}\nFall-out = {FP/(TN+FP):.4f}"],
        [f"FN = {FN:,}\nMiss Rate = {FN/(FN+TP):.4f}", f"TP = {TP:,}\nSens = {metrics['sensitivity']:.4f}"],
    ]
    for r in range(2):
        for c in range(2):
            col = "white" if cm[r, c] > cm.max() * 0.5 else "#0d1117"
            ax3.text(c, r, cell_text[r][c], ha="center", va="center",
                     fontsize=10, color=col, fontweight="bold")
    ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Predicted Licit", "Predicted Illicit"], color="#8b949e", fontsize=9)
    ax3.set_yticklabels(["True Licit", "True Illicit"], color="#8b949e", fontsize=9)
    ax3.set_title(f"Confusion Matrix @ θ = {threshold:.3f}", color="white", fontsize=11, fontweight="bold")
    ax3.tick_params(colors="#8b949e")
    ax3.spines[:].set_color("#30363d")
    plt.colorbar(im, ax=ax3).ax.tick_params(colors="#8b949e")
    ax3.set_xlabel(
        f"F1={metrics['f1_illicit']:.4f}  |  MCC={metrics['mcc']:.4f}  |  "
        f"PPV={metrics['ppv']:.4f}  |  NPV={metrics['npv']:.4f}",
        color="#ffd93d", fontsize=8, fontweight="bold"
    )

    # ── (D) Threshold vs Metrics ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#161b22")
    thr_range = np.arange(0.005, 0.995, 0.005)
    f1s, ps, rs, mccs = [], [], [], []
    for t in thr_range:
        yp = (proba >= t).astype(int)
        f1s.append(f1_score(y_true, yp, zero_division=0))
        ps.append(precision_score(y_true, yp, zero_division=0))
        rs.append(recall_score(y_true, yp, zero_division=0))
        mccs.append(matthews_corrcoef(y_true, yp))
    ax4.plot(thr_range, f1s,  color="#00d4ff", lw=2.2, label="F1 (Illicit)")
    ax4.plot(thr_range, ps,   color="#ffd93d", lw=1.5, ls="--", label="Precision")
    ax4.plot(thr_range, rs,   color="#ff6b6b", lw=1.5, ls="--", label="Recall")
    ax4.plot(thr_range, mccs, color="#6bcb77", lw=1.5, ls=":",  label="MCC")
    ax4.axvline(threshold, color="white", ls=":", lw=1.5, label=f"θ = {threshold:.3f}")
    best_idx = int(np.argmax(f1s))
    ax4.scatter(thr_range[best_idx], f1s[best_idx], color="#00d4ff", s=80, zorder=5)
    ax4.set_xlabel("Decision Threshold (θ)", color="#8b949e", fontsize=9)
    ax4.set_ylabel("Score", color="#8b949e", fontsize=9)
    ax4.set_title(f"Threshold Optimisation — {name.upper()}", color="white", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=8, facecolor="#0d1117", labelcolor="white")
    ax4.tick_params(colors="#8b949e", labelsize=7)
    ax4.spines[:].set_color("#30363d")
    ax4.set_ylim(-0.05, 1.05)

    fig.suptitle(
        f"{name.upper()} Model — Comprehensive Evaluation\n"
        f"F1={metrics['f1_illicit']:.4f}  AUC-ROC={metrics['auc_roc']:.4f}  "
        f"AUC-PR={metrics['auc_pr']:.4f}  MCC={metrics['mcc']:.4f}  θ={threshold:.3f}",
        color="white", fontsize=13, fontweight="bold"
    )

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"eval_{name}_dashboard.png",
                    dpi=150, bbox_inches="tight", facecolor="#0d1117")
        print(f"  Figure saved: {out_dir}/eval_{name}_dashboard.png")
    plt.close(fig)


# =============================================================================
# Multi-model comparison
# =============================================================================

def evaluate_all(X: np.ndarray, y: np.ndarray,
                 optimize_thresholds: bool = False,
                 output_dir: Path = None) -> pd.DataFrame:
    """
    Evaluate every saved base model plus the Hybrid and produce a
    comparative report and figure.
    """
    model_names = ["xgboost", "lightgbm", "catboost",
                   "random_forest", "extra_trees", "svm", "hybrid"]
    all_metrics = []
    all_probas  = {}

    for name in model_names:
        try:
            model = load_model(name)
            if name == "hybrid":
                proba = model.predict_proba(X)
            else:
                proba = model.predict_proba(X)[:, 1]

            t = (find_best_threshold(y, proba) if optimize_thresholds
                 else DEFAULT_THRESHOLDS.get(name, 0.5))

            m = compute_metrics(y, proba, t, name)
            all_metrics.append(m)
            all_probas[name] = proba
            print(f"  {name:15s}  F1={m['f1_illicit']:.4f}  AUC-ROC={m['auc_roc']:.4f}"
                  f"  AUC-PR={m['auc_pr']:.4f}  MCC={m['mcc']:.4f}  θ={t:.3f}")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    df = pd.DataFrame(all_metrics)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "eval_all_models_comparison.csv", index=False)
        print(f"\n  Comparison table saved: {out_dir}/eval_all_models_comparison.csv")

    # ── Comparison figure ─────────────────────────────────────────────────────
    _plot_comparison(all_probas, y, df, output_dir)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FULL COMPARISON TABLE")
    print("=" * 80)
    cols = ["model", "threshold", "f1_illicit", "auc_roc", "auc_pr", "mcc",
            "precision", "recall", "specificity", "accuracy"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return df


def _plot_comparison(all_probas, y_true, metrics_df, output_dir):
    """4-panel comparison figure across all models."""
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.33)

    # (A) ROC comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#161b22")
    for name, proba in all_probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ra = auc(fpr, tpr)
        lw = 3.0 if name == "svm" else 1.5
        ax1.plot(fpr, tpr, color=PALETTE.get(name, "white"), lw=lw,
                 label=f"{name} ({ra:.3f})")
    ax1.plot([0, 1], [0, 1], "gray", ls=":", lw=1)
    ax1.set_title("ROC Curves — All Models", color="white", fontsize=11, fontweight="bold")
    ax1.set_xlabel("FPR", color="#8b949e", fontsize=9)
    ax1.set_ylabel("TPR", color="#8b949e", fontsize=9)
    ax1.legend(fontsize=7.5, facecolor="#0d1117", labelcolor="white", loc="lower right")
    ax1.tick_params(colors="#8b949e", labelsize=7); ax1.spines[:].set_color("#30363d")

    # (B) PR comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161b22")
    for name, proba in all_probas.items():
        prec, rec, _ = precision_recall_curve(y_true, proba)
        pa = auc(rec, prec)
        lw = 3.0 if name == "svm" else 1.5
        ax2.plot(rec, prec, color=PALETTE.get(name, "white"), lw=lw,
                 label=f"{name} (AP={pa:.3f})")
    ax2.set_title("Precision-Recall Curves — All Models", color="white", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Recall", color="#8b949e", fontsize=9)
    ax2.set_ylabel("Precision", color="#8b949e", fontsize=9)
    ax2.legend(fontsize=7.5, facecolor="#0d1117", labelcolor="white", loc="upper right")
    ax2.tick_params(colors="#8b949e", labelsize=7); ax2.spines[:].set_color("#30363d")

    # (C) F1 and MCC bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#161b22")
    names  = metrics_df["model"].tolist()
    f1s    = metrics_df["f1_illicit"].tolist()
    mccs_v = metrics_df["mcc"].tolist()
    x = np.arange(len(names))
    ax3.bar(x - 0.2, f1s,    0.38, color=[PALETTE.get(n, "#58a6ff") for n in names], alpha=0.85, label="F1 (Illicit)")
    ax3.bar(x + 0.2, mccs_v, 0.38, color=[PALETTE.get(n, "#58a6ff") for n in names], alpha=0.45, label="MCC", edgecolor="white", linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=40, ha="right", fontsize=8, color="#8b949e")
    ax3.set_ylabel("Score", color="#8b949e", fontsize=9)
    ax3.set_title("F1 (Illicit) vs MCC — All Models", color="white", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8, facecolor="#0d1117", labelcolor="white")
    ax3.tick_params(colors="#8b949e", labelsize=7); ax3.spines[:].set_color("#30363d")
    for i, (f1, mc) in enumerate(zip(f1s, mccs_v)):
        ax3.text(i - 0.2, f1 + 0.01, f"{f1:.3f}", ha="center", fontsize=6.5, color="white")
        ax3.text(i + 0.2, mc + 0.01, f"{mc:.3f}", ha="center", fontsize=6.5, color="#8b949e")

    # (D) Multi-metric profile
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#161b22")
    metric_keys  = ["f1_illicit", "auc_roc", "auc_pr", "mcc", "recall", "precision"]
    metric_labels = ["F1 (Illicit)", "AUC-ROC", "AUC-PR", "MCC", "Recall", "Precision"]
    x_pos = np.arange(len(metric_keys))
    for _, row in metrics_df.iterrows():
        vals = [row[k] for k in metric_keys]
        lw = 3.0 if row["model"] == "svm" else 1.5
        ax4.plot(x_pos, vals, color=PALETTE.get(row["model"], "white"), lw=lw,
                 marker="o", markersize=4, label=row["model"])
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metric_labels, color="#8b949e", fontsize=8.5)
    ax4.set_ylim(-0.05, 1.05)
    ax4.set_ylabel("Score", color="#8b949e", fontsize=9)
    ax4.set_title("6-Metric Profile — All Models", color="white", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=7.5, facecolor="#0d1117", labelcolor="white", ncol=2)
    ax4.tick_params(colors="#8b949e", labelsize=7); ax4.spines[:].set_color("#30363d")
    ax4.grid(axis="y", color="#30363d", lw=0.5, alpha=0.5)

    fig.suptitle("All Models — Comprehensive Comparison", color="white",
                 fontsize=14, fontweight="bold")

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "eval_all_comparison.png",
                    dpi=150, bbox_inches="tight", facecolor="#0d1117")
        print(f"  Comparison figure saved: {out_dir}/eval_all_comparison.png")
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone model evaluator for the Hybrid FDS pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model", default="svm",
        choices=["svm", "xgboost", "lightgbm", "catboost",
                 "random_forest", "extra_trees", "hybrid", "all"],
        help="Which model to evaluate. Use 'all' to compare every saved model.",
    )
    p.add_argument(
        "--data", default=None,
        help="Path to a custom CSV file (must contain all 74 selected feature columns)."
             " If omitted, uses the Elliptic test set from results/checkpoints/.",
    )
    p.add_argument(
        "--labels", default=None,
        help="Path to a CSV with a 'label' column (0=licit, 1=illicit). "
             "Required for metric computation when --data is provided.",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Fixed decision threshold in [0, 1]. If omitted, uses the optimal "
             "training threshold for each model. Use --optimize to search automatically.",
    )
    p.add_argument(
        "--optimize", action="store_true",
        help="Search for the F1-maximising threshold from the provided data.",
    )
    p.add_argument(
        "--output", default="results/evaluation",
        help="Directory to save evaluation figures and CSVs. Default: results/evaluation/",
    )
    return p.parse_args()


def main():
    import sys, io
    # Ensure UTF-8 output on Windows (avoids cp1252 encoding errors)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    out_dir = Path(args.output)

    print("\n" + "=" * 64)
    print("  Hybrid FDS -- Standalone Model Evaluator")
    print("=" * 64)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.data:
        print(f"\nLoading custom data from: {args.data}")
        X, y, tx_ids, feat_names = load_custom_data(args.data, args.labels)
    else:
        print("\nLoading Elliptic Bitcoin test set from checkpoints...")
        X, y, tx_ids, feat_names = load_elliptic_test()

    print(f"  Data shape: {X.shape}")
    if y is not None:
        print(f"  Labels: {int((y==1).sum()):,} illicit, {int((y==0).sum()):,} licit")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if args.model == "all":
        print(f"\nEvaluating all models...")
        df = evaluate_all(X, y,
                          optimize_thresholds=args.optimize,
                          output_dir=out_dir)
    else:
        metrics = evaluate_single(
            args.model, X, y,
            threshold=args.threshold,
            output_dir=out_dir,
            optimize_threshold=args.optimize,
        )

    print(f"\nAll outputs saved to: {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
