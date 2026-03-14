"""
pipeline/data_loader.py
───────────────────────
Load BTC OHLCV data from yfinance and optionally merge with the Global M2
liquidity feature table produced by m2_liquidity.py.
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
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)

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


def load_data(config: dict) -> pd.DataFrame:
    """
    Main data-loading entry point for the pipeline.

    1. Downloads BTC OHLCV data.
    2. If ``config["use_m2_exog"]`` is True, builds/loads the Global M2
       liquidity feature table and inner-joins it to the BTC frame on Date.
       This automatically truncates early BTC days where M2 data is absent.

    Parameters
    ----------
    config : dict
        Pipeline configuration (see config.py).

    Returns
    -------
    pd.DataFrame
        Combined BTC (+M2 if enabled) DataFrame sorted by Date.
    """
    df_btc = load_btc_data(config)

    if not config.get("use_m2_exog", True):
        logger.info("use_m2_exog=False – skipping Global M2 features.")
        return df_btc

    start_date = df_btc["Date"].min().strftime("%Y-%m-%d")
    end_date = df_btc["Date"].max().strftime("%Y-%m-%d")

    df_m2 = load_or_build_m2_series(config, start_date=start_date, end_date=end_date)

    df_m2["Date"] = pd.to_datetime(df_m2["Date"]).dt.normalize()
    df_btc["Date"] = pd.to_datetime(df_btc["Date"]).dt.normalize()

    df_merged = df_btc.merge(df_m2, on="Date", how="inner")
    df_merged = df_merged.sort_values("Date").reset_index(drop=True)

    logger.info(
        "Merged BTC + M2 frame: %d rows (Date: %s → %s), %d columns",
        len(df_merged),
        df_merged["Date"].min().date(),
        df_merged["Date"].max().date(),
        df_merged.shape[1],
    )
    return df_merged
