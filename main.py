"""
main.py
────────
End-to-end pipeline with Purged Walk-Forward CV + MBB Bagging:

  1. Load BTC data (+ Global M2 if use_m2_exog=True, + cross-assets)
  2. Build features and target
  3. Walk-forward CV with Moving Block Bootstrap ensemble per fold
  4. Collect out-of-sample predictions across all folds
  5. Run Binance spot backtester on OOS predictions
  6. Save backtest outputs (trade log, charts, summary JSON)
  7. Train final model on all data
  8. Predict next week's BTC direction

Usage
─────
    python main.py

Environment variables
─────────────────────
    FRED_API_KEY   – required when config["m2_source"] == "bis_fred"
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

from config import CONFIG
from pipeline.data_loader import load_data
from pipeline.features import build_features_and_target
from pipeline.model import predict, predict_ensemble, train_ensemble, train_model
from pipeline.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_shap_beeswarm,
)
from pipeline.walk_forward import create_walk_forward_cv
from pipeline.bootstrap import create_mbb
from pipeline.backtester import run_backtest, save_backtest_outputs, print_backtest_summary
from pipeline.sdae import encode_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(config: dict = CONFIG) -> dict:
    """
    Execute the full pipeline and return a results dict with metrics.
    """
    out_dir = config.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)

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

    logger.info("Feature matrix ready: %d rows × %d features", len(X), len(feature_cols))

    # Convert to numpy arrays for the rest of the pipeline
    X_arr = X.values.astype(np.float64)
    y_arr = y.values.astype(np.float64)

    # ── 3. Walk-Forward CV with MBB Bagging ──────────────────────────────────
    logger.info("=== Step 3: Walk-Forward CV with MBB Bagging ===")
    wf_cv = create_walk_forward_cv(config)
    folds = wf_cv.split(len(X_arr))

    if not folds:
        logger.error("No walk-forward folds generated. Need more data.")
        sys.exit(1)

    mbb = create_mbb(config)

    # Collect OOS predictions across all folds
    all_oos_indices = []
    all_oos_p_up = []
    all_oos_y_true = []
    fold_metrics = []

    for fold in folds:
        logger.info(
            "── Fold %d: train[%d:%d] (%d samples), test[%d:%d] (%d samples) ──",
            fold.fold_idx,
            fold.train_start, fold.train_end, fold.train_end - fold.train_start,
            fold.test_start, fold.test_end, fold.test_end - fold.test_start,
        )

        X_train, y_train, X_test, y_test = wf_cv.get_fold_data(X_arr, y_arr, fold)

        # Split training data into train/val (last 15% for SDAE/LightGBM early stopping)
        val_size = max(50, int(len(X_train) * 0.15))
        X_tr = X_train[:-val_size]
        y_tr = y_train[:-val_size]
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]

        # Generate bootstrap samples from X_tr/y_tr
        bootstrap_samples = mbb.generate_samples(X_tr, y_tr)

        # Train ensemble on bootstrap samples
        models = train_ensemble(
            bootstrap_samples, X_val, y_val, feature_cols, config
        )

        # Predict on test set using ensemble
        p_up_fold = predict_ensemble(X_test, feature_cols, models, config)

        # Compute fold metrics
        fm = compute_metrics(y_test, p_up_fold)
        fm["fold"] = fold.fold_idx
        fm["n_train"] = len(X_train)
        fm["n_test"] = len(X_test)
        fold_metrics.append(fm)

        logger.info(
            "Fold %d results: acc=%.4f, f1=%.4f, auc=%.4f",
            fold.fold_idx, fm["accuracy"], fm["f1"], fm["auc"],
        )

        # Store OOS predictions
        test_indices = list(range(fold.test_start, fold.test_end))
        all_oos_indices.extend(test_indices)
        all_oos_p_up.extend(p_up_fold.tolist())
        all_oos_y_true.extend(y_test.tolist())

    # ── 4. Aggregate OOS results ─────────────────────────────────────────────
    logger.info("=== Step 4: Aggregate OOS results ===")
    oos_p_up = np.array(all_oos_p_up)
    oos_y_true = np.array(all_oos_y_true)
    oos_indices = np.array(all_oos_indices)

    # Handle overlapping folds: keep last prediction for each index
    unique_idx, last_occurrence = np.unique(oos_indices, return_index=False), None
    idx_to_pred = {}
    idx_to_true = {}
    for i, idx in enumerate(all_oos_indices):
        idx_to_pred[idx] = all_oos_p_up[i]
        idx_to_true[idx] = all_oos_y_true[i]

    sorted_oos_idx = sorted(idx_to_pred.keys())
    oos_p_up = np.array([idx_to_pred[i] for i in sorted_oos_idx])
    oos_y_true = np.array([idx_to_true[i] for i in sorted_oos_idx])

    overall_metrics = compute_metrics(oos_y_true, oos_p_up)
    logger.info(
        "Overall OOS metrics (%d samples): acc=%.4f, f1=%.4f, auc=%.4f",
        len(oos_y_true), overall_metrics["accuracy"],
        overall_metrics["f1"], overall_metrics["auc"],
    )

    # Print per-fold metrics summary
    logger.info("Per-fold accuracy: %s",
                [f"fold{fm['fold']}={fm['accuracy']:.4f}" for fm in fold_metrics])

    # Save OOS confusion matrix
    plot_confusion_matrix(oos_y_true, oos_p_up, config, filename="confusion_matrix_oos.png")

    # ── 5. Backtest on OOS predictions ───────────────────────────────────────
    logger.info("=== Step 5: Backtest on OOS predictions ===")

    # Get dates and prices for the OOS indices
    # We need to reconstruct dates from the original dataframe
    # The feature matrix X was built from df after dropna, so indices align
    # Use the Date column from the feature-aligned data
    from pipeline.features import _add_btc_features, _add_target, _add_cross_asset_features, _add_calendar_features
    df_feat = df.copy()
    df_feat = _add_btc_features(df_feat, config)
    df_feat = _add_cross_asset_features(df_feat, config)
    df_feat = _add_calendar_features(df_feat)
    df_feat = _add_target(df_feat, config)

    horizon = config.get("btc_target_horizon", 7)
    target_col = f"up_{horizon}d"

    # Replicate the dropna logic from build_features_and_target
    exclude = {"Open", "High", "Low", "Close", "Volume", "Date"}
    cross_tickers = config.get("cross_asset_tickers", {})
    for name in cross_tickers:
        exclude.add(f"close_{name}")

    all_feat_cols = [
        c for c in df_feat.columns
        if c not in exclude and not c.startswith("up_")
    ]
    df_aligned = df_feat[["Date", "Close"] + all_feat_cols + [target_col]].copy()
    df_aligned = df_aligned.replace([np.inf, -np.inf], np.nan)
    df_aligned = df_aligned.dropna().reset_index(drop=True)

    oos_dates = df_aligned["Date"].iloc[sorted_oos_idx].values
    oos_prices = df_aligned["Close"].iloc[sorted_oos_idx].values.astype(float)

    bt_result = run_backtest(oos_dates, oos_prices, oos_p_up, config)
    print_backtest_summary(bt_result)
    save_backtest_outputs(bt_result, config)

    # ── 6. Train final model on all data ─────────────────────────────────────
    logger.info("=== Step 6: Train final model ===")

    # Use all data with last 15% as validation
    val_size_final = max(50, int(len(X_arr) * 0.15))
    X_train_final = X_arr[:-val_size_final]
    y_train_final = y_arr[:-val_size_final]
    X_val_final = X_arr[-val_size_final:]
    y_val_final = y_arr[-val_size_final:]

    sdae_model, scaler, lgbm_model, lgbm_features = train_model(
        X_train_final, y_train_final, X_val_final, y_val_final, feature_cols, config
    )

    # Feature importance plot from final model
    plot_feature_importance(lgbm_model, lgbm_features, config)

    # SHAP plot from final model
    X_val_sc = scaler.transform(X_val_final)
    Z_val = encode_features(sdae_model, X_val_sc)
    use_m2 = config.get("use_m2_exog", True)
    m2_cols = [c for c in feature_cols if c.startswith("M2_") or c.startswith("m2_")] if use_m2 else []
    if m2_cols:
        m2_idx = [feature_cols.index(c) for c in m2_cols]
        Z_val_full = np.hstack([Z_val, X_val_sc[:, m2_idx]])
    else:
        Z_val_full = Z_val
    plot_shap_beeswarm(lgbm_model, Z_val_full, lgbm_features, config)

    # ── 7. Predict next week ─────────────────────────────────────────────────
    logger.info("=== Step 7: Predict next period ===")
    last_row = X_arr[-1:].copy()
    p_next = predict(last_row, feature_cols, sdae_model, scaler, lgbm_model, config)
    signal = "UP" if p_next[0] >= 0.5 else "DOWN"
    logger.info(
        "Next %d-day BTC forecast: p_up=%.4f  signal=%s",
        horizon, p_next[0], signal,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    results = {
        "oos_metrics": overall_metrics,
        "fold_metrics": fold_metrics,
        "n_folds": len(folds),
        "backtest": {
            "total_return_pct": bt_result.total_return_pct,
            "sharpe_ratio": bt_result.sharpe_ratio,
            "max_drawdown_pct": bt_result.max_drawdown_pct,
            "n_trades": bt_result.n_trades,
            "win_rate": bt_result.win_rate,
        },
        "next_period_p_up": float(p_next[0]),
        "next_period_signal": signal,
        "feature_count": len(feature_cols),
        "m2_feature_count": len(m2_cols),
    }

    logger.info("=== Pipeline complete ===")
    logger.info("OOS accuracy: %.4f", overall_metrics["accuracy"])
    logger.info("Backtest total return: %.2f%%", bt_result.total_return_pct)
    logger.info("Next week signal: %s (p_up=%.4f)", signal, p_next[0])

    return results


if __name__ == "__main__":
    run()
