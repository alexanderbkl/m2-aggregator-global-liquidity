"""
pipeline/features.py
─────────────────────
Build the feature matrix and classification target from the merged BTC+M2
DataFrame returned by data_loader.load_data().

BTC technical features
──────────────────────
• Log-returns (1-day, 7-day, 30-day)
• Rolling mean / std of close over configurable windows
• Normalised price range (High-Low)/Close
• Volume change (pct)

Global M2 features (when use_m2_exog=True)
──────────────────────────────────────────
All columns starting with "M2_" or "m2_" are treated as exogenous
macro features and passed through unchanged.

Target
──────
``up_<horizon>d``  = 1 if Close[t + horizon] > Close[t], else 0
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _add_btc_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add technical indicator columns derived from BTC OHLCV data."""
    df = df.copy()
    close = df["Close"]
    windows: List[int] = config.get("btc_rolling_windows", [7, 14, 30, 60, 90])

    # Log-returns at key horizons
    df["log_ret_1d"] = np.log(close / close.shift(1))
    df["log_ret_7d"] = np.log(close / close.shift(7))
    df["log_ret_30d"] = np.log(close / close.shift(30))

    for w in windows:
        df[f"close_ma_{w}d"] = close.rolling(w).mean()
        df[f"close_std_{w}d"] = close.rolling(w).std()
        # Price relative to its rolling mean (Z-score flavour)
        df[f"close_zscore_{w}d"] = (
            (close - df[f"close_ma_{w}d"]) / (df[f"close_std_{w}d"] + 1e-9)
        )

    # Intraday range normalised by close
    if "High" in df.columns and "Low" in df.columns:
        df["norm_range"] = (df["High"] - df["Low"]) / (close + 1e-9)

    # Volume change
    if "Volume" in df.columns:
        df["vol_pct_chg_1d"] = df["Volume"].pct_change(1)
        df["vol_pct_chg_7d"] = df["Volume"].pct_change(7)

    return df


def _add_target(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add a binary classification target: 1 if BTC close is higher
    ``btc_target_horizon`` days ahead, else 0.
    """
    horizon: int = config.get("btc_target_horizon", 7)
    df = df.copy()
    df[f"up_{horizon}d"] = (
        df["Close"].shift(-horizon) > df["Close"]
    ).astype(int)
    return df


def build_features_and_target(
    df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Build feature matrix X and target series y from the merged BTC (+M2) frame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_loader.load_data() – BTC OHLCV optionally merged with M2.
    config : dict
        Pipeline configuration.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (rows aligned with Date, NaN rows dropped).
    y : pd.Series
        Binary target (up_<horizon>d).
    feature_cols : list[str]
        Ordered list of feature column names used in X.
    """
    df = _add_btc_features(df, config)
    df = _add_target(df, config)

    horizon: int = config.get("btc_target_horizon", 7)
    target_col = f"up_{horizon}d"

    # Identify feature columns
    ohlcv = {"Open", "High", "Low", "Close", "Volume", "Date"}
    btc_feat_cols = [c for c in df.columns if c not in ohlcv and not c.startswith("up_")]

    # Separate M2 feature columns
    use_m2: bool = config.get("use_m2_exog", True)
    if use_m2:
        m2_cols = [c for c in btc_feat_cols if c.startswith("M2_") or c.startswith("m2_")]
        non_m2_cols = [c for c in btc_feat_cols if c not in m2_cols]
        feature_cols = non_m2_cols + m2_cols
    else:
        feature_cols = [
            c for c in btc_feat_cols
            if not (c.startswith("M2_") or c.startswith("m2_"))
        ]

    # Drop rows missing target (last `horizon` rows) or any feature
    df_feat = df[["Date"] + feature_cols + [target_col]].copy()
    df_feat = df_feat.dropna()

    X = df_feat[feature_cols].reset_index(drop=True)
    y = df_feat[target_col].reset_index(drop=True)

    logger.info(
        "Feature matrix: %d rows × %d features (%d M2 columns)",
        len(X),
        len(feature_cols),
        len([c for c in feature_cols if c.startswith("M2_") or c.startswith("m2_")]),
    )
    return X, y, feature_cols
