"""
main.py
────────
End-to-end pipeline:
  1. Load BTC data (+ Global M2 if use_m2_exog=True)
  2. Build features and target
  3. Time-series train / val / test split
  4. Scale → SDAE → LightGBM
  5. Evaluate and produce charts
  6. Predict next week's BTC direction

Usage
─────
    python main.py

Environment variables
─────────────────────
    FRED_API_KEY   – required when config["m2_source"] == "bis_fred"

Toggle M2 features off for an ablation run:
    CONFIG["use_m2_exog"] = False  (edit config.py or set here before importing)
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

# ── Load config first so callers can override before importing pipeline ──────
from config import CONFIG
from pipeline.data_loader import load_data
from pipeline.features import build_features_and_target
from pipeline.model import predict, train_model
from pipeline.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_equity_curve,
    plot_feature_importance,
    plot_regime_accuracy,
    plot_shap_beeswarm,
)
from pipeline.sdae import encode_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _split(
    X: pd.DataFrame,
    y: pd.Series,
    df_full: pd.DataFrame,
    config: dict,
):
    """Chronological train / val / test split (no shuffling)."""
    n = len(X)
    n_train = int(n * config["train_ratio"])
    n_val = int(n * config["val_ratio"])

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train: n_train + n_val], y.iloc[n_train: n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val:], y.iloc[n_train + n_val:]

    df_test_rows = df_full.iloc[n_train + n_val: n_train + n_val + len(X_test)].reset_index(drop=True)

    logger.info(
        "Split  train=%d  val=%d  test=%d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, df_test_rows


def run(config: dict = CONFIG) -> dict:
    """
    Execute the full pipeline and return a results dict with metrics.
    """
    os.makedirs(config.get("output_dir", "outputs"), exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("=== Step 1: Load data ===")
    df = load_data(config)
    if df.empty:
        logger.error("No data loaded. Exiting.")
        sys.exit(1)

    # ── 2. Feature engineering ────────────────────────────────────────────────
    logger.info("=== Step 2: Build features ===")
    X, y, feature_cols = build_features_and_target(df, config)
    if len(X) < 200:
        logger.error("Insufficient data rows (%d) after feature engineering.", len(X))
        sys.exit(1)

    # Rebuild aligned df to keep regime columns for evaluation
    horizon = config.get("btc_target_horizon", 7)
    target_col = f"up_{horizon}d"
    df_aligned = df.copy()
    from pipeline.features import _add_btc_features, _add_target
    df_aligned = _add_btc_features(df_aligned, config)
    df_aligned = _add_target(df_aligned, config)
    df_aligned = df_aligned.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    # ── 3. Split ──────────────────────────────────────────────────────────────
    logger.info("=== Step 3: Split ===")
    X_train, y_train, X_val, y_val, X_test, y_test, df_test_rows = _split(
        X, y, df_aligned, config
    )

    # ── 4. Train SDAE + LightGBM ──────────────────────────────────────────────
    logger.info("=== Step 4: Train SDAE + LightGBM ===")
    sdae_model, scaler, lgbm_model, lgbm_features = train_model(
        X_train, y_train, X_val, y_val, feature_cols, config
    )

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    logger.info("=== Step 5: Evaluate ===")
    p_up = predict(X_test, feature_cols, sdae_model, scaler, lgbm_model, config)
    metrics = compute_metrics(y_test.values, p_up)

    plot_confusion_matrix(y_test.values, p_up, config)
    plot_equity_curve(y_test.values, p_up, config)
    plot_feature_importance(lgbm_model, lgbm_features, config)

    # Build Z_test for SHAP
    X_test_sc = scaler.transform(X_test[feature_cols].values)
    Z_test = encode_features(sdae_model, X_test_sc)
    use_m2 = config.get("use_m2_exog", True)
    m2_cols = [c for c in feature_cols if c.startswith("M2_") or c.startswith("m2_")] if use_m2 else []
    if m2_cols:
        m2_idx = [feature_cols.index(c) for c in m2_cols]
        Z_test_full = np.hstack([Z_test, X_test_sc[:, m2_idx]])
    else:
        Z_test_full = Z_test

    plot_shap_beeswarm(lgbm_model, Z_test_full, lgbm_features, config)
    plot_regime_accuracy(df_test_rows, y_test.values, p_up, config)

    # ── 6. Predict next week ──────────────────────────────────────────────────
    logger.info("=== Step 6: Predict next period ===")
    # Use the last available row (most recent M2 + BTC snapshot)
    last_row = X.tail(1)
    p_next = predict(last_row, feature_cols, sdae_model, scaler, lgbm_model, config)
    signal = "UP ↑" if p_next[0] >= 0.5 else "DOWN ↓"
    logger.info(
        "Next %d-day BTC forecast: p_up=%.4f  signal=%s",
        horizon, p_next[0], signal,
    )

    results = {
        "metrics": metrics,
        "next_period_p_up": float(p_next[0]),
        "next_period_signal": signal,
        "feature_count": len(feature_cols),
        "m2_feature_count": len(m2_cols),
    }
    logger.info("=== Pipeline complete ===  results=%s", results)
    return results


if __name__ == "__main__":
    run()
