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
    "use_m2_exog": True,              # set False to disable M2 features (ablation)

    # ── M2 data source ────────────────────────────────────────────────────
    # "bis_fred"  → build series from FRED API (+ optional BIS supplement)
    # "csv"       → load from a pre-computed CSV at m2_csv_path
    "m2_source": "bis_fred",

    # Path to pre-computed CSV (used when m2_source == "csv").
    # Expected columns: Date (YYYY-MM-DD), M2_global_usd
    "m2_csv_path": os.path.join("data", "global_m2.csv"),

    # FRED API key – read from environment variable FRED_API_KEY by default.
    # You may hard-code it here, but using an env var is recommended.
    # "fred_api_key": os.environ.get("FRED_API_KEY", ""),
    "fred_api_key": "aaf3121388bab2aba7ad45a91c0790a4",

    # Country basket for M2 aggregation (FRED path).
    # Keys must match entries in M2_FRED_SERIES inside m2_liquidity.py.
    "m2_countries": ["US", "EA", "CN", "JP", "GB"],

    # Lag offsets (in calendar days) for the lagged M2 features
    "m2_lag_days": [1, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98],

    # Growth windows (in days) computed on the daily M2 series
    "m2_growth_windows": [7, 30, 90],

    # ── Feature engineering ────────────────────────────────────────────────
    # Rolling windows (in days) for BTC technical features
    "btc_rolling_windows": [7, 14, 30, 60, 90],

    # ── Train / validation / test split ────────────────────────────────────
    # Fraction of the dataset used for training (chronological split)
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    # The remaining fraction goes to the test set

    # ── SDAE (Stacked Denoising Autoencoder) ──────────────────────────────
    "sdae_hidden_dims": [256, 128, 64],   # encoder hidden layer sizes
    "sdae_noise_factor": 0.1,             # Gaussian noise σ added during training
    "sdae_dropout": 0.2,
    "sdae_learning_rate": 1e-3,
    "sdae_weight_decay": 1e-5,
    "sdae_epochs": 50,
    "sdae_batch_size": 256,
    "sdae_patience": 10,
    "sdae_log_every_epochs": 1,
    "sdae_log_every_batches": 0,   # set >0 for very verbose batch-level progress logs
    "sdae_torch_num_threads": None,  # e.g. 1-4 can help on some macOS CPU/OpenMP setups

    # ── LightGBM ──────────────────────────────────────────────────────────
    "lgbm_params": {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "lgbm_early_stopping_rounds": 50,

    # ── Evaluation & output ────────────────────────────────────────────────
    "output_dir": "outputs",
    "save_plots": True,

    # Regime analysis: column used to bucket test samples
    "regime_column": "m2_90d_chg",   # must exist in features when use_m2_exog=True
    "regime_n_bins": 3,               # e.g. low / medium / high liquidity growth
}
