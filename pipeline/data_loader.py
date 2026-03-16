"""
pipeline/data_loader.py
───────────────────────
Load BTC OHLCV data from yfinance, optionally merge with Global M2
liquidity features and cross-asset momentum data.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from pipeline.m2_liquidity import load_or_build_m2_series

logger = logging.getLogger(__name__)


def load_btc_data(config: dict) -> pd.DataFrame:
    """
    Download BTC OHLCV data from Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        Columns: Date, Open, High, Low, Close, Volume
        Sorted ascending by Date, no duplicate dates.
    """
    ticker: str = config.get("btc_ticker", "BTC-USD")
    start: str = config.get("btc_start_date", "2015-01-01")

    logger.info("Downloading BTC data: ticker=%s start=%s", ticker, start)
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False, timeout=100)

    if raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for ticker '{ticker}'. "
            "Check your internet connection or ticker symbol."
        )

    # Flatten MultiIndex columns if present (yfinance ≥0.2.x)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()

    # Normalise the date column name
    date_col = "Date" if "Date" in raw.columns else raw.columns[0]
    raw = raw.rename(columns={date_col: "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.normalize()

    # Keep standard OHLCV columns
    ohlcv = ["Open", "High", "Low", "Close", "Volume"]
    available = [c for c in ohlcv if c in raw.columns]
    df = raw[["Date"] + available].copy()
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

    logger.info(
        "BTC data loaded: %d rows (Date: %s → %s)",
        len(df),
        df["Date"].min().date(),
        df["Date"].max().date(),
    )
    return df


def load_cross_asset_data(config: dict, start_date: str) -> pd.DataFrame:
    """
    Download cross-asset price data (Gold, DXY, S&P500, NASDAQ) from yfinance.

    Fault-tolerant: if a ticker fails, it is skipped with a warning.

    Returns
    -------
    pd.DataFrame
        Columns: Date, close_gold, close_dxy, close_sp500, close_nasdaq (whichever succeed).
    """
    tickers = config.get("cross_asset_tickers", {})
    if not tickers or not config.get("fetch_cross_assets", True):
        return pd.DataFrame()

    frames = []
    for name, ticker in tickers.items():
        try:
            logger.info("Downloading cross-asset data: %s (%s)", name, ticker)
            raw = yf.download(ticker, start=start_date, auto_adjust=True, progress=False, timeout=60)
            if raw.empty:
                logger.warning("No data returned for cross-asset '%s' (%s), skipping.", name, ticker)
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw = raw.reset_index()
            date_col = "Date" if "Date" in raw.columns else raw.columns[0]
            raw = raw.rename(columns={date_col: "Date"})
            raw["Date"] = pd.to_datetime(raw["Date"]).dt.normalize()

            col_name = f"close_{name}"
            asset_df = raw[["Date"]].copy()
            asset_df[col_name] = raw["Close"].values
            asset_df = asset_df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
            frames.append(asset_df)
            logger.info("Cross-asset '%s': %d rows loaded.", name, len(asset_df))
        except Exception as e:
            logger.warning("Failed to download cross-asset '%s' (%s): %s", name, ticker, e)
            continue

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="Date", how="outer")

    merged = merged.sort_values("Date").reset_index(drop=True)
    # Forward-fill for different trading calendars
    for col in merged.columns:
        if col != "Date":
            merged[col] = merged[col].ffill()

    logger.info("Cross-asset data merged: %d rows, %d columns", len(merged), merged.shape[1])
    return merged


def load_data(config: dict) -> pd.DataFrame:
    """
    Main data-loading entry point for the pipeline.

    1. Downloads BTC OHLCV data.
    2. If ``config["use_m2_exog"]`` is True, builds/loads the Global M2
       liquidity feature table and inner-joins it to the BTC frame on Date.
    3. If ``config["fetch_cross_assets"]`` is True, downloads cross-asset
       data and left-joins it to the BTC frame on Date.

    Parameters
    ----------
    config : dict
        Pipeline configuration (see config.py).

    Returns
    -------
    pd.DataFrame
        Combined BTC (+M2 +cross-asset if enabled) DataFrame sorted by Date.
    """
    df_btc = load_btc_data(config)

    if config.get("use_m2_exog", True):
        start_date = df_btc["Date"].min().strftime("%Y-%m-%d")
        end_date = df_btc["Date"].max().strftime("%Y-%m-%d")

        df_m2 = load_or_build_m2_series(config, start_date=start_date, end_date=end_date)

        df_m2["Date"] = pd.to_datetime(df_m2["Date"]).dt.normalize()
        df_btc["Date"] = pd.to_datetime(df_btc["Date"]).dt.normalize()

        df_btc = df_btc.merge(df_m2, on="Date", how="inner")
        logger.info(
            "Merged BTC + M2 frame: %d rows (Date: %s → %s), %d columns",
            len(df_btc),
            df_btc["Date"].min().date(),
            df_btc["Date"].max().date(),
            df_btc.shape[1],
        )
    else:
        logger.info("use_m2_exog=False – skipping Global M2 features.")

    # Cross-asset data
    if config.get("fetch_cross_assets", True):
        start_date = df_btc["Date"].min().strftime("%Y-%m-%d")
        df_cross = load_cross_asset_data(config, start_date)
        if not df_cross.empty:
            df_cross["Date"] = pd.to_datetime(df_cross["Date"]).dt.normalize()
            df_btc = df_btc.merge(df_cross, on="Date", how="left")
            # Forward-fill cross-asset columns for weekends/holidays
            cross_cols = [c for c in df_cross.columns if c != "Date"]
            for c in cross_cols:
                df_btc[c] = df_btc[c].ffill()
            logger.info("Merged cross-asset data: %d columns added.", len(cross_cols))

    df_btc = df_btc.sort_values("Date").reset_index(drop=True)
    return df_btc
