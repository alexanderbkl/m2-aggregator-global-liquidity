"""
Live integration tests for real Yahoo Finance and FRED data paths.

These tests are intentionally opt-in and will be skipped unless:
    RUN_LIVE_DATA_TESTS=1

FRED-dependent tests also require:
    FRED_API_KEY=<your_key>
"""

from __future__ import annotations

import io
import logging
import os

import pandas as pd
import pytest

from pipeline.data_loader import load_btc_data, load_data
from pipeline.m2_liquidity import load_or_build_m2_series


logger = logging.getLogger(__name__)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_DATA_TESTS") != None,
    reason="Set RUN_LIVE_DATA_TESTS=1 to enable live Yahoo/FRED integration tests.",
)


def _log_frame_snapshot(name: str, df: pd.DataFrame, rows: int = 3) -> None:
    """Log DataFrame structure and first rows for live-debug visibility."""
    logger.info("[%s] shape=%s", name, df.shape)
    logger.info("[%s] columns=%s", name, list(df.columns))

    if "Date" in df.columns and not df.empty:
        dates = pd.to_datetime(df["Date"], errors="coerce")
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            logger.info(
                "[%s] date_range=%s -> %s (rows=%d, duplicates=%d)",
                name,
                valid_dates.min().date(),
                valid_dates.max().date(),
                len(df),
                int(dates.duplicated().sum()),
            )
        else:
            logger.info("[%s] date_range=unavailable (all Date values are NaT)", name)

    buf = io.StringIO()
    df.info(buf=buf)
    logger.info("[%s] info:\n%s", name, buf.getvalue().rstrip())
    logger.info("[%s] head(%d):\n%s", name, rows, df.head(rows).to_string(index=False))


def _base_config() -> dict:
    return {
        "btc_ticker": "BTC-USD",
        "btc_start_date": "2024-01-01",
        "use_m2_exog": True,
        "m2_source": "bis_fred",
        "fred_api_key": "aaf3121388bab2aba7ad45a91c0790a4",
        "m2_countries": ["US"],
        "m2_lag_days": [1, 7],
        "m2_growth_windows": [7, 30],
    }


def test_live_yahoo_download_returns_consistent_btc_frame():
    cfg = _base_config()

    df = load_btc_data(cfg)
    _log_frame_snapshot("BTC_YAHOO", df)

    assert not df.empty
    assert set(["Date", "Open", "High", "Low", "Close", "Volume"]).issubset(df.columns)
    assert df["Date"].is_monotonic_increasing
    assert not df["Date"].duplicated().any()


def test_live_fred_builds_m2_feature_table():
    cfg = _base_config()
    if not cfg["fred_api_key"]:
        pytest.skip("FRED_API_KEY not set; skipping live FRED integration test.")

    start_date = "2024-01-01"
    end_date = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    df_m2 = load_or_build_m2_series(cfg, start_date=start_date, end_date=end_date)
    _log_frame_snapshot("M2_FRED", df_m2)

    assert not df_m2.empty
    assert "Date" in df_m2.columns
    assert "M2_global_usd" in df_m2.columns
    assert "M2_global_usd_lag_1" in df_m2.columns
    assert "m2_7d_chg" in df_m2.columns


def test_live_btc_and_m2_are_mergeable():
    cfg = _base_config()
    if not cfg["fred_api_key"]:
        pytest.skip("FRED_API_KEY not set; skipping live merge integration test.")

    btc = load_btc_data(cfg)
    _log_frame_snapshot("BTC_FOR_MERGE", btc)

    m2 = load_or_build_m2_series(
        cfg,
        start_date=btc["Date"].min().strftime("%Y-%m-%d"),
        end_date=btc["Date"].max().strftime("%Y-%m-%d"),
    )
    _log_frame_snapshot("M2_FOR_MERGE", m2)

    merged = load_data(cfg)
    _log_frame_snapshot("MERGED_BTC_M2", merged)

    if "Date" in btc.columns and "Date" in m2.columns and "Date" in merged.columns:
        btc_dates = set(pd.to_datetime(btc["Date"]).dt.normalize())
        m2_dates = set(pd.to_datetime(m2["Date"]).dt.normalize())
        overlap = len(btc_dates & m2_dates)
        logger.info(
            "[MERGE_CHECK] btc_dates=%d m2_dates=%d overlap=%d merged_rows=%d",
            len(btc_dates),
            len(m2_dates),
            overlap,
            len(merged),
        )

    assert not merged.empty
    assert "Date" in merged.columns
    assert "Close" in merged.columns
    assert "M2_global_usd" in merged.columns
    assert merged["Date"].is_monotonic_increasing
    assert not merged["Date"].duplicated().any()
