"""
pipeline/evaluation.py
───────────────────────
Metrics computation and plotting for the BTC + Global M2 pipeline.

Charts produced
───────────────
1. Confusion matrix
2. Equity curve (long when p_up > 0.5, else flat)
3. Feature importance bar chart (LightGBM, highlighting M2 features)
4. SHAP beeswarm (top 20 features)
5. Regime accuracy (DA per liquidity-growth bucket)  ← M2-specific
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    logger.warning("matplotlib not installed; plots will be skipped.")

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    logger.warning("shap not installed; SHAP plots will be skipped.")


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, p_up: np.ndarray) -> dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : binary ground-truth labels (0/1)
    p_up   : predicted probability of being up

    Returns
    -------
    dict with keys: accuracy, f1, auc, directional_accuracy
    """
    y_pred = (p_up >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, p_up),
    }
    logger.info(
        "Metrics  acc=%.4f  f1=%.4f  AUC=%.4f",
        metrics["accuracy"], metrics["f1"], metrics["auc"],
    )
    return metrics


# ── Helper: save figure ────────────────────────────────────────────────────────

def _save(fig: "plt.Figure", filename: str, config: dict) -> None:
    if not config.get("save_plots", True):
        plt.close(fig)
        return
    out_dir: str = config.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    logger.info("Saved plot: %s", path)
    plt.close(fig)


# ── 1. Confusion matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    p_up: np.ndarray,
    config: dict,
    filename: str = "confusion_matrix.png",
) -> None:
    if not _HAS_MPL:
        return
    y_pred = (p_up >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Down", "Up"]); ax.set_yticklabels(["Down", "Up"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    _save(fig, filename, config)


# ── 2. Equity curve ────────────────────────────────────────────────────────────

def plot_equity_curve(
    y_true: np.ndarray,
    p_up: np.ndarray,
    config: dict,
    filename: str = "equity_curve.png",
) -> None:
    if not _HAS_MPL:
        return
    signal = (p_up >= 0.5).astype(float)
    # Daily PnL = signal × (2*y_true - 1)  (+1 correct, -1 incorrect)
    pnl = signal * (2 * y_true.astype(float) - 1)
    equity = np.cumsum(pnl)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity, label="Strategy equity")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Cumulative PnL (unit bets)")
    ax.set_title("Strategy Equity Curve")
    ax.legend()
    _save(fig, filename, config)


# ── 3. Feature importance ──────────────────────────────────────────────────────

def plot_feature_importance(
    lgbm_model,
    lgbm_features: List[str],
    config: dict,
    top_n: int = 30,
    filename: str = "feature_importance.png",
) -> None:
    if not _HAS_MPL:
        return
    importance = lgbm_model.feature_importances_
    feat_imp = (
        pd.Series(importance, index=lgbm_features)
        .sort_values(ascending=False)
        .head(top_n)
    )

    is_m2 = feat_imp.index.str.startswith("M2_") | feat_imp.index.str.startswith("m2_")
    colours = ["#d62728" if m else "#1f77b4" for m in is_m2]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.3)))
    feat_imp.plot.barh(ax=ax, color=colours)
    ax.set_xlabel("Importance (split count)")
    ax.set_title(f"Top {top_n} LightGBM Features\n(red = Global M2 features)")
    ax.invert_yaxis()
    _save(fig, filename, config)

    # Also save M2-only chart
    m2_imp = (
        pd.Series(importance, index=lgbm_features)
        .loc[lambda s: s.index.str.startswith("M2_") | s.index.str.startswith("m2_")]
        .sort_values(ascending=False)
        .head(top_n)
    )
    if m2_imp.empty:
        return
    fig2, ax2 = plt.subplots(figsize=(10, max(4, len(m2_imp) * 0.35)))
    m2_imp.plot.barh(ax=ax2, color="#d62728")
    ax2.set_xlabel("Importance (split count)")
    ax2.set_title("Global M2 Feature Importance")
    ax2.invert_yaxis()
    _save(fig2, "m2_feature_importance.png", config)


# ── 4. SHAP beeswarm ──────────────────────────────────────────────────────────

def plot_shap_beeswarm(
    lgbm_model,
    Z_test: np.ndarray,
    lgbm_features: List[str],
    config: dict,
    max_display: int = 20,
    filename: str = "shap_beeswarm.png",
) -> None:
    if not (_HAS_MPL and _HAS_SHAP):
        return
    try:
        explainer = shap.TreeExplainer(lgbm_model)
        shap_values = explainer.shap_values(Z_test)
        # For binary classifiers SHAP may return a list [neg_class, pos_class]
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            sv,
            Z_test,
            feature_names=lgbm_features,
            max_display=max_display,
            show=False,
        )
        _save(plt.gcf(), filename, config)
    except Exception as exc:
        logger.warning("SHAP plot failed: %s", exc)


# ── 5. Regime accuracy ─────────────────────────────────────────────────────────

def plot_regime_accuracy(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    p_up: np.ndarray,
    config: dict,
    filename: str = "m2_regime_accuracy.png",
) -> None:
    """
    Bucket test samples by liquidity-growth regime and plot directional
    accuracy per bucket.  Only executed when use_m2_exog=True and the
    regime column is present.
    """
    if not _HAS_MPL:
        return
    regime_col: str = config.get("regime_column", "m2_90d_chg")
    n_bins: int = config.get("regime_n_bins", 3)

    if not config.get("use_m2_exog", True) or regime_col not in df_test.columns:
        logger.info("Regime accuracy plot skipped (regime_col '%s' not available).", regime_col)
        return

    regime_vals = df_test[regime_col].values
    if np.isnan(regime_vals).all():
        logger.warning("All regime values are NaN; skipping regime accuracy plot.")
        return

    labels = ["Low", "Medium", "High"][:n_bins]
    try:
        buckets = pd.qcut(regime_vals, q=n_bins, labels=labels, duplicates="drop")
    except Exception as exc:
        logger.warning("Regime binning failed: %s", exc)
        return

    y_pred = (p_up >= 0.5).astype(int)
    da_per_regime: dict = {}
    categories = buckets.cat.categories if hasattr(buckets, "cat") else buckets.categories
    for label in categories:
        mask = buckets == label
        if mask.sum() < 5:
            continue
        da_per_regime[str(label)] = accuracy_score(y_true[mask], y_pred[mask])

    if not da_per_regime:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    regs = list(da_per_regime.keys())
    das = [da_per_regime[r] for r in regs]
    bars = ax.bar(regs, das, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(regs)])
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Random baseline")
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Liquidity growth regime ({regime_col})")
    ax.set_ylabel("Directional Accuracy")
    ax.set_title("DA per Global M2 Liquidity Regime")
    ax.legend()
    for bar, da in zip(bars, das):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{da:.2%}",
            ha="center",
            fontsize=10,
        )
    _save(fig, filename, config)
