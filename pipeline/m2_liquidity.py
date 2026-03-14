"""
pipeline/m2_liquidity.py
────────────────────────
Build (or load) a daily Global M2 liquidity series and generate lagged /
growth features ready for merging with BTC data.

Data sources
────────────
Primary  : FRED API  (US M2 + foreign M2 series + monthly FX rates)
Fallback : User-supplied CSV with at minimum columns [Date, M2_global_usd]

Country basket (configurable via config["m2_countries"])
─────────────────────────────────────────────────────────
US  → FRED M2SL          (Billions USD, monthly SA)
EA  → FRED MABMM301EZM189S (Millions EUR, monthly SA)  + FRED EXUSEU FX
CN  → FRED MYAGM2CNM189N  (Billions CNY, monthly NSA) + FRED EXCHUS FX
JP  → FRED MYAGM2JPM189N  (Billions JPY, monthly SA)  + FRED EXJPUS FX
GB  → FRED MABMM301GBM189S (Millions GBP, monthly SA) + FRED EXUSUK FX
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── FRED series metadata ───────────────────────────────────────────────────────
# Each entry: (m2_series_id, currency, unit_multiplier_to_billions, fx_series_id, fx_direction)
# fx_direction:
#   "multiply" → value_usd = local_value * fx_rate   (fx_rate = USD per 1 unit local)
#   "divide"   → value_usd = local_value / fx_rate   (fx_rate = local per 1 USD)
#   None       → already in USD, no FX needed

M2_FRED_SERIES: Dict[str, dict] = {
    "US": {
        "m2_id": "M2SL",
        "currency": "USD",
        "unit_to_billions": 1.0,      # already in billions USD
        "fx_id": None,
        "fx_direction": None,
    },
    "EA": {
        "m2_id": "MABMM301EZM189S",
        "currency": "EUR",
        "unit_to_billions": 1e-3,     # millions → billions
        "fx_id": "EXUSEU",            # USD per 1 EUR
        "fx_direction": "multiply",
    },
    "CN": {
        "m2_id": "MYAGM2CNM189N",
        "currency": "CNY",
        "unit_to_billions": 1.0,      # already in billions CNY
        "fx_id": "EXCHUS",            # CNY per 1 USD
        "fx_direction": "divide",
    },
    "JP": {
        "m2_id": "MYAGM2JPM189N",
        "currency": "JPY",
        "unit_to_billions": 1.0,      # already in billions JPY
        "fx_id": "EXJPUS",            # JPY per 1 USD
        "fx_direction": "divide",
    },
    "GB": {
        "m2_id": "MABMM301GBM189S",
        "currency": "GBP",
        "unit_to_billions": 1e-3,     # millions → billions
        "fx_id": "EXUSUK",            # USD per 1 GBP
        "fx_direction": "multiply",
    },
}

FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"


# ── Low-level FRED fetcher ─────────────────────────────────────────────────────

def _fetch_fred(series_id: str, api_key: str) -> pd.Series:
    """
    Download a FRED series and return it as a pd.Series with a DatetimeIndex.
    Returns an empty Series on failure (so the caller can handle gracefully).
    """
    if not api_key:
        raise ValueError(
            "FRED API key is required. Set the FRED_API_KEY environment variable "
            "or config['fred_api_key']."
        )
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": "2000-01-01",
    }
    try:
        resp = requests.get(FRED_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("observations", [])
        records = [
            (obs["date"], float(obs["value"]))
            for obs in data
            if obs["value"] != "."
        ]
        if not records:
            logger.warning("No data returned for FRED series %s", series_id)
            return pd.Series(dtype=float, name=series_id)
        dates, values = zip(*records)
        idx = pd.to_datetime(dates)
        return pd.Series(values, index=idx, name=series_id)
    except Exception as exc:
        logger.warning("Failed to fetch FRED series %s: %s", series_id, exc)
        return pd.Series(dtype=float, name=series_id)


# ── Country-level M2 in USD ────────────────────────────────────────────────────

def _country_m2_usd(country: str, api_key: str) -> pd.Series:
    """
    Return a monthly pd.Series of M2 (in billions USD) for one country.
    Returns an empty Series if data cannot be obtained.
    """
    meta = M2_FRED_SERIES.get(country)
    if meta is None:
        raise ValueError(f"Unknown country code '{country}'. "
                         f"Supported: {list(M2_FRED_SERIES)}")

    m2 = _fetch_fred(meta["m2_id"], api_key)
    if m2.empty:
        return pd.Series(dtype=float, name=f"m2_{country}_usd")

    # Convert to billions (local currency)
    m2_bn = m2 * meta["unit_to_billions"]

    if meta["fx_id"] is None:
        # Already USD
        result = m2_bn
    else:
        fx = _fetch_fred(meta["fx_id"], api_key)
        if fx.empty:
            logger.warning(
                "FX data unavailable for %s (%s); skipping country.", country, meta["fx_id"]
            )
            return pd.Series(dtype=float, name=f"m2_{country}_usd")

        # Resample FX to month-end to align with monthly M2
        fx_monthly = fx.resample("ME").last().ffill()

        # Align on common dates
        m2_bn, fx_monthly = m2_bn.align(fx_monthly, join="inner")

        if meta["fx_direction"] == "multiply":
            result = m2_bn * fx_monthly
        else:
            result = m2_bn / fx_monthly

    result.name = f"m2_{country}_usd"
    return result.dropna()


# ── Aggregate global M2 (monthly) ─────────────────────────────────────────────

def _build_global_m2_monthly(config: dict) -> pd.Series:
    """
    Fetch each country's M2, convert to billions USD, and sum to a single
    monthly global M2 series.
    """
    api_key: str = config.get("fred_api_key", "")
    countries: List[str] = config.get("m2_countries", list(M2_FRED_SERIES))

    series_list: List[pd.Series] = []
    failed: List[str] = []

    for country in countries:
        logger.info("Fetching M2 for country: %s", country)
        s = _country_m2_usd(country, api_key)
        if s.empty:
            logger.warning("No M2 data for %s; excluding from global sum.", country)
            failed.append(country)
        else:
            series_list.append(s)

    if not series_list:
        raise RuntimeError(
            "Could not obtain M2 data for any country. "
            "Check your FRED_API_KEY and network connectivity, or use m2_source='csv'."
        )

    if failed:
        logger.warning(
            "Global M2 computed without: %s. Results may underestimate global liquidity.",
            failed,
        )

    df = pd.concat(series_list, axis=1).sort_index()
    global_m2 = df.sum(axis=1, min_count=1).dropna()
    global_m2.name = "M2_global_usd"

    # Resample to month-end frequency (forward-fill gaps ≤ 2 months)
    global_m2 = global_m2.resample("ME").last().ffill(limit=2)
    return global_m2


# ── Resample monthly → daily ───────────────────────────────────────────────────

def _resample_to_daily(
    monthly: pd.Series,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Forward-fill a monthly series onto a complete daily date range.
    Liquidity conditions are treated as piecewise constant between
    reporting dates (standard macro-trading convention).
    """
    daily_idx = pd.date_range(start=start_date, end=end_date, freq="D")
    daily = monthly.reindex(daily_idx, method="ffill")
    daily.name = "M2_global_usd"
    return daily.to_frame().reset_index().rename(columns={"index": "Date"})


# ── Growth features ────────────────────────────────────────────────────────────

def _add_growth_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"m2_{w}d_chg"] = df["M2_global_usd"].pct_change(w)
    return df


# ── Lagged features ────────────────────────────────────────────────────────────

def _add_lagged_features(df: pd.DataFrame, lag_days: List[int]) -> pd.DataFrame:
    df = df.copy()
    base_cols = [c for c in df.columns if c != "Date"]
    for lag in lag_days:
        for col in base_cols:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


# ── CSV loader ─────────────────────────────────────────────────────────────────

def _load_from_csv(config: dict, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load a pre-computed Global M2 CSV.
    Expected columns: Date (YYYY-MM-DD), M2_global_usd

    The CSV may be at monthly or daily granularity; this function forward-fills
    to produce a complete daily series.
    """
    csv_path: str = config.get("m2_csv_path", os.path.join("data", "global_m2.csv"))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"M2 CSV not found at '{csv_path}'. "
            "Provide the file or set config['m2_source'] = 'bis_fred'."
        )
    raw = pd.read_csv(csv_path, parse_dates=["Date"])
    if "M2_global_usd" not in raw.columns:
        raise ValueError(
            "CSV must contain a 'M2_global_usd' column with USD-denominated M2 values."
        )
    raw = raw[["Date", "M2_global_usd"]].dropna().sort_values("Date")
    monthly = raw.set_index("Date")["M2_global_usd"]
    return _resample_to_daily(monthly, start_date, end_date)


# ── Public API ─────────────────────────────────────────────────────────────────

def load_or_build_m2_series(
    config: dict,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Build or load a daily Global M2 liquidity feature table.

    Parameters
    ----------
    config : dict
        Pipeline configuration (see config.py).
    start_date : str, optional
        Earliest date for the daily series (YYYY-MM-DD).
        Defaults to config['btc_start_date'].
    end_date : str, optional
        Latest date for the daily series (YYYY-MM-DD).
        Defaults to today.

    Returns
    -------
    pd.DataFrame
        Columns: Date, M2_global_usd, m2_*d_chg, M2_global_usd_lag_*, ...
        Rows with leading NaNs (from lagging) are **not** dropped here so
        the caller can decide how to handle them during the merge.
    """
    if start_date is None:
        start_date = config.get("btc_start_date", "2015-01-01")
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    source: str = config.get("m2_source", "bis_fred")

    if source == "csv":
        logger.info("Loading Global M2 from CSV: %s", config.get("m2_csv_path"))
        df = _load_from_csv(config, start_date, end_date)
    else:
        logger.info("Building Global M2 from FRED API (countries: %s)",
                    config.get("m2_countries"))
        monthly = _build_global_m2_monthly(config)
        df = _resample_to_daily(monthly, start_date, end_date)

    growth_windows: List[int] = config.get("m2_growth_windows", [7, 30, 90])
    df = _add_growth_features(df, growth_windows)

    lag_days: List[int] = config.get("m2_lag_days", [1, 7, 14, 21, 28])
    df = _add_lagged_features(df, lag_days)

    df["Date"] = pd.to_datetime(df["Date"])
    logger.info(
        "Global M2 feature table ready: %d rows, %d columns (Date: %s → %s)",
        len(df),
        df.shape[1],
        df["Date"].min().date(),
        df["Date"].max().date(),
    )
    return df
