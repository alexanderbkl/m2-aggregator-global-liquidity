"""
tests/test_data_loader.py
─────────────────────────
Unit tests for pipeline.data_loader.

These tests are fully offline. External data providers (Yahoo/FRED) are mocked
so we can validate data consistency and BTC+M2 mergeability deterministically.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from pipeline.data_loader import load_btc_data, load_data


class TestLoadBtcData:
    def _config(self) -> dict:
        return {
            "btc_ticker": "BTC-USD",
            "btc_start_date": "2020-01-01",
        }

    def test_downloaded_btc_is_sorted_unique_and_has_expected_columns(self):
        raw = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-01"]),
                "Open": [3.0, 1.0, 1.1],
                "High": [3.5, 1.5, 1.6],
                "Low": [2.5, 0.5, 0.6],
                "Close": [3.2, 1.2, 1.25],
                "Volume": [300, 100, 110],
            }
        )

        with patch("pipeline.data_loader.yf.download", return_value=raw):
            df = load_btc_data(self._config())

        assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
        assert df["Date"].is_monotonic_increasing
        assert not df["Date"].duplicated().any()
        assert len(df) == 2

    def test_flattens_yfinance_multiindex_columns(self):
        idx = pd.date_range("2020-01-01", periods=2, freq="D")
        cols = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["BTC-USD"]]
        )
        raw = pd.DataFrame(
            [
                [1.0, 1.5, 0.5, 1.2, 100],
                [2.0, 2.5, 1.5, 2.2, 200],
            ],
            index=idx,
            columns=cols,
        )

        with patch("pipeline.data_loader.yf.download", return_value=raw):
            df = load_btc_data(self._config())

        assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(set(df.columns))
        assert len(df) == 2

    def test_raises_on_empty_download(self):
        with patch("pipeline.data_loader.yf.download", return_value=pd.DataFrame()):
            with pytest.raises(RuntimeError, match="returned no data"):
                load_btc_data(self._config())


class TestLoadDataMergeability:
    def test_merges_btc_and_m2_on_date_inner_join(self):
        config = {
            "use_m2_exog": True,
        }

        df_btc = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                "Open": [1.0, 2.0, 3.0],
                "High": [1.5, 2.5, 3.5],
                "Low": [0.5, 1.5, 2.5],
                "Close": [1.2, 2.2, 3.2],
                "Volume": [100, 200, 300],
            }
        )
        df_m2 = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2020-01-02 08:30:00", "2020-01-03 21:00:00"]),
                "M2_global_usd": [1000.0, 1010.0],
                "M2_global_usd_lag_1": [995.0, 1005.0],
                "m2_7d_chg": [0.01, 0.015],
            }
        )

        with patch("pipeline.data_loader.load_btc_data", return_value=df_btc):
            with patch("pipeline.data_loader.load_or_build_m2_series", return_value=df_m2):
                merged = load_data(config)

        # Inner join should keep only overlapping dates (2020-01-02, 2020-01-03)
        assert len(merged) == 2
        assert merged["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-02", "2020-01-03"]
        assert "Close" in merged.columns
        assert "M2_global_usd" in merged.columns
        assert merged["Date"].is_monotonic_increasing

    def test_skips_m2_merge_when_flag_disabled(self):
        config = {
            "use_m2_exog": False,
        }
        df_btc = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "Open": [1.0, 2.0],
                "High": [1.5, 2.5],
                "Low": [0.5, 1.5],
                "Close": [1.2, 2.2],
                "Volume": [100, 200],
            }
        )

        with patch("pipeline.data_loader.load_btc_data", return_value=df_btc):
            with patch("pipeline.data_loader.load_or_build_m2_series") as mocked_m2:
                out = load_data(config)

        mocked_m2.assert_not_called()
        pd.testing.assert_frame_equal(out, df_btc)
