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
• Volume change (pct) and volume z-scores
• RSI at 14 and 28-day periods
• MACD line, signal, and histogram (12/26/9 EMA)
• Bollinger Band width and %B position (20-day)
• Average True Range normalised by close (14 and 28-day)
• Rate of change at 3, 14, and 30-day periods

Cross-asset momentum signals (when available)
──────────────────────────────────────────────
• 7d and 30d returns for Gold, DXY, S&P500, NASDAQ
• 30d rolling correlation between BTC and each cross-asset

Volatility regime features
──────────────────────────
• Realized volatility ratio (7d/30d)
• Parkinson volatility estimator
• Garman-Klass volatility estimator

Calendar / cyclical features
────────────────────────────
• Day of week (sin/cos encoded)
• Month of year (sin/cos encoded)
• Quarter indicator
• Bitcoin halving cycle position

Global M2 features (when use_m2_exog=True)
──────────────────────────────────────────
All columns starting with "M2_" or "m2_" are treated as exogenous
macro features and passed through unchanged.

Feature selection
─────────────────
Uses mutual information to select top-K most informative features.

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

# Bitcoin halving dates (approximate block dates)
_HALVING_DATES = [
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-20"),
    pd.Timestamp("2028-04-01"),  # estimated next halving
]


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


def _parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 14) -> pd.Series:
    """Parkinson volatility estimator using high-low range."""
    log_hl = np.log(high / (low + _EPS))
    return np.sqrt((log_hl ** 2).rolling(window).mean() / (4 * np.log(2)))


def _garman_klass_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Garman-Klass volatility estimator."""
    log_hl = np.log(high / (low + _EPS))
    log_co = np.log(close / (open_ + _EPS))
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    return np.sqrt(gk.rolling(window).mean().clip(lower=0))


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

    # Volume change + z-scores
    if "Volume" in df.columns:
        df["vol_pct_chg_1d"] = df["Volume"].pct_change(1)
        df["vol_pct_chg_7d"] = df["Volume"].pct_change(7)
        # Volume z-scores
        vol_ma_7 = df["Volume"].rolling(7).mean()
        vol_std_7 = df["Volume"].rolling(7).std()
        df["volume_zscore_7d"] = (df["Volume"] - vol_ma_7) / (vol_std_7 + _EPS)
        vol_ma_30 = df["Volume"].rolling(30).mean()
        vol_std_30 = df["Volume"].rolling(30).std()
        df["volume_zscore_30d"] = (df["Volume"] - vol_ma_30) / (vol_std_30 + _EPS)

    # ── Price-volume divergence ────────────────────────────────────────────
    if "Volume" in df.columns:
        price_chg_7d = close.pct_change(7)
        vol_chg_7d = df["Volume"].pct_change(7)
        df["pv_divergence_7d"] = price_chg_7d - vol_chg_7d

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

    # ── Volatility regime features ───────────────────────────────────────────
    rv_7d = close.pct_change().rolling(7).std()
    rv_30d = close.pct_change().rolling(30).std()
    df["rv_ratio_7_30"] = rv_7d / (rv_30d + _EPS)

    if "High" in df.columns and "Low" in df.columns:
        df["parkinson_vol_14"] = _parkinson_volatility(df["High"], df["Low"], window=14)
        df["garman_klass_vol_14"] = _garman_klass_volatility(
            df["Open"], df["High"], df["Low"], close, window=14
        )

    return df


def _add_cross_asset_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add cross-asset momentum signals and correlations."""
    df = df.copy()
    close = df["Close"]

    cross_tickers = config.get("cross_asset_tickers", {})
    for name in cross_tickers:
        col = f"close_{name}"
        if col not in df.columns:
            continue

        asset = df[col]

        # 7d and 30d returns for the cross-asset
        df[f"{name}_ret_7d"] = asset.pct_change(7)
        df[f"{name}_ret_30d"] = asset.pct_change(30)

        # 30d rolling correlation with BTC
        df[f"btc_{name}_corr_30d"] = close.rolling(30).corr(asset)

    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical calendar and halving cycle features."""
    df = df.copy()

    if "Date" not in df.columns:
        return df

    dates = pd.to_datetime(df["Date"])

    # Day of week (sin/cos encoded)
    dow = dates.dt.dayofweek.astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # Month of year (sin/cos encoded)
    month = dates.dt.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Quarter
    df["quarter"] = dates.dt.quarter.astype(float)

    # Bitcoin halving cycle position
    halving_pos = []
    for d in dates:
        past_halvings = [h for h in _HALVING_DATES if h <= d]
        future_halvings = [h for h in _HALVING_DATES if h > d]

        if past_halvings and future_halvings:
            last_halving = max(past_halvings)
            next_halving = min(future_halvings)
            cycle_length = (next_halving - last_halving).days
            days_since = (d - last_halving).days
            pos = days_since / cycle_length if cycle_length > 0 else 0.5
        elif past_halvings:
            last_halving = max(past_halvings)
            days_since = (d - last_halving).days
            pos = days_since / (4 * 365.25)
        else:
            pos = 0.0

        halving_pos.append(pos)

    df["halving_cycle_pos"] = halving_pos

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


def _select_features_mi(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    k: int = 40,
) -> List[str]:
    """
    Select top-K features using mutual information.
    """
    from sklearn.feature_selection import mutual_info_classif

    if len(feature_cols) <= k:
        logger.info("Feature count (%d) <= k (%d), skipping selection.", len(feature_cols), k)
        return feature_cols

    X_clean = X[feature_cols].copy()
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

    mi_scores = mutual_info_classif(
        X_clean.values, y.values, random_state=42, n_neighbors=5
    )
    mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
    selected = mi_series.head(k).index.tolist()

    logger.info(
        "Feature selection: %d → %d features (MI). Top 5: %s",
        len(feature_cols), len(selected),
        list(mi_series.head(5).index),
    )
    return selected


def build_features_and_target(
    df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Build feature matrix X and target series y from the merged BTC (+M2) frame.

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
    df = _add_cross_asset_features(df, config)
    df = _add_calendar_features(df)
    df = _add_target(df, config)

    horizon: int = config.get("btc_target_horizon", 7)
    target_col = f"up_{horizon}d"

    # Identify feature columns (exclude OHLCV, Date, target, and raw cross-asset close)
    exclude = {"Open", "High", "Low", "Close", "Volume", "Date"}
    cross_tickers = config.get("cross_asset_tickers", {})
    for name in cross_tickers:
        exclude.add(f"close_{name}")

    btc_feat_cols = [
        c for c in df.columns
        if c not in exclude and not c.startswith("up_")
    ]

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
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.dropna()

    X = df_feat[feature_cols].reset_index(drop=True)
    y = df_feat[target_col].reset_index(drop=True)
    dates = df_feat["Date"].reset_index(drop=True)

    # Feature selection via mutual information
    k = config.get("feature_selection_k", 40)
    if k > 0 and len(feature_cols) > k:
        selected = _select_features_mi(X, y, feature_cols, k=k)
        feature_cols = selected
        X = X[feature_cols]

    logger.info(
        "Feature matrix: %d rows × %d features (%d M2 columns)",
        len(X),
        len(feature_cols),
        len([c for c in feature_cols if c.startswith("M2_") or c.startswith("m2_")]),
    )
    return X, y, feature_cols
