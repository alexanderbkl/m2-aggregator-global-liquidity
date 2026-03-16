"""
Global configuration for the M2 Aggregator + Bitcoin Price Prediction pipeline.

Adjust these settings to control data sources, feature engineering,
model architecture, and evaluation behaviour.
"""

import os

CONFIG: dict = {
    # ── BTC data ──────────────────────────────────────────────────────────────
    "btc_ticker": "BTC-USD",          # yfinance ticker
    "btc_start_date": "2014-01-01",   # earliest date to fetch BTC OHLCV
    "btc_target_horizon": 7,          # predict price direction N days ahead

    # ── Global M2 feature flag ─────────────────────────────────────────────
    "use_m2_exog": False,              # set False to disable M2 features (ablation)

    # ── M2 data source ────────────────────────────────────────────────────
    "m2_source": "bis_fred",
    "m2_csv_path": os.path.join("data", "global_m2.csv"),
    "fred_api_key": "aaf3121388bab2aba7ad45a91c0790a4",
    "m2_countries": ["US", "EA", "CN", "JP", "GB"],
    "m2_lag_days": [1, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98],
    "m2_growth_windows": [7, 14, 30, 60, 90, 180],

    # ── Feature engineering ────────────────────────────────────────────────
    "btc_rolling_windows": [7, 14, 30, 60, 90],

    # ── Cross-asset features ──────────────────────────────────────────────
    "fetch_cross_assets": True,
    "cross_asset_tickers": {
        "gold": "GC=F",
        "dxy": "DX-Y.NYB",
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
    },

    # ── Feature selection ─────────────────────────────────────────────────
    "feature_selection_k": 40,

    # ── Walk-Forward CV ───────────────────────────────────────────────────
    "wf_min_train_days": 1095,        # 3 years minimum training
    "wf_test_days": 182,              # 6 months test window
    "wf_step_days": 182,              # step forward by test_window
    "wf_purge_days": 7,               # = horizon_days
    "wf_embargo_days": 7,

    # ── Moving Block Bootstrap ────────────────────────────────────────────
    "mbb_block_size": 60,             # 2 months of daily data
    "mbb_n_bootstraps": 5,            # number of bootstrap samples per fold

    # ── SDAE (Stacked Denoising Autoencoder) ──────────────────────────────
    # hidden_dims now computed dynamically from input_dim
    "sdae_noise_factor": 0.1,
    "sdae_dropout": 0.2,
    "sdae_learning_rate": 1e-3,
    "sdae_weight_decay": 1e-5,
    "sdae_epochs": 50,
    "sdae_batch_size": 256,
    "sdae_patience": 15,
    "sdae_log_every_epochs": 10,
    "sdae_log_every_batches": 0,
    "sdae_torch_num_threads": 2,

    # ── LightGBM (conservative for noisy financial data) ──────────────────
    "lgbm_params": {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 50,
        "subsample": 0.6,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 3.0,
        "reg_lambda": 10.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "lgbm_early_stopping_rounds": 40,

    # ── Backtester ────────────────────────────────────────────────────────
    "bt_initial_capital": 10000,
    "bt_maker_fee": 0.001,            # 0.1%
    "bt_taker_fee": 0.001,            # 0.1%
    "bt_slippage_pct": 0.0005,        # 0.05%
    "bt_position_size_pct": 0.95,
    "bt_confidence_threshold": 0.50,

    # ── Evaluation & output ────────────────────────────────────────────────
    "output_dir": "outputs",
    "save_plots": True,
    "regime_column": "m2_90d_chg",
    "regime_n_bins": 3,
}
