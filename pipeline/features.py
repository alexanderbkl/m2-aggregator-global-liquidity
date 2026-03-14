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
• RSI at 14 and 28-day periods
• MACD line, signal, and histogram (12/26/9 EMA)
• Bollinger Band width and %B position (20-day)
• Average True Range normalised by close (14 and 28-day)
• Rate of change at 3, 14, and 30-day periods

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

# Small constant to prevent division-by-zero throughout this module
_EPS = 1e-9


# ── Technical indicator helpers ────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + _EPS)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


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
            (close - df[f"close_ma_{w}d"]) / (df[f"close_std_{w}d"] + _EPS)
        )

    # Intraday range normalised by close
    if "High" in df.columns and "Low" in df.columns:
        df["norm_range"] = (df["High"] - df["Low"]) / (close + _EPS)

    # Volume change
    if "Volume" in df.columns:
        df["vol_pct_chg_1d"] = df["Volume"].pct_change(1)
        df["vol_pct_chg_7d"] = df["Volume"].pct_change(7)

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["rsi_14"] = _rsi(close, period=14)
    df["rsi_28"] = _rsi(close, period=28)

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_line, macd_signal, macd_hist = _macd(close)
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # ── Bollinger Bands (20-day) ──────────────────────────────────────────────
    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / (bb_ma + _EPS)
    bb_pct = (close - bb_lower) / (bb_upper - bb_lower + _EPS)
    df["bb_width_20"] = bb_width
    df["bb_pct_20"] = bb_pct

    # ── Average True Range ────────────────────────────────────────────────────
    if "High" in df.columns and "Low" in df.columns:
        df["atr_14"] = _atr(df["High"], df["Low"], close, period=14) / (close + _EPS)
        df["atr_28"] = _atr(df["High"], df["Low"], close, period=28) / (close + _EPS)

    # ── Rate of Change (price momentum) ──────────────────────────────────────
    df["roc_3d"] = close.pct_change(3)
    df["roc_14d"] = close.pct_change(14)
    df["roc_30d"] = close.pct_change(30)

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
