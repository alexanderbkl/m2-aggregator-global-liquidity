"""
tests/test_m2_liquidity.py
───────────────────────────
Unit tests for the Global M2 liquidity module.

All tests are designed to run **without** network access or a FRED API key
by using local CSV fixtures or patched HTTP calls.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pipeline.m2_liquidity import (
    _add_growth_features,
    _add_lagged_features,
    _fetch_fred,
    _country_m2_usd,
    _build_global_m2_monthly,
    _resample_to_daily,
    _load_from_csv,
    load_or_build_m2_series,
    M2_FRED_SERIES,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_monthly_series(start: str = "2020-01-31", periods: int = 24, value: float = 1000.0) -> pd.Series:
    """Return a monthly pd.Series with constant value (billions USD)."""
    idx = pd.date_range(start=start, periods=periods, freq="ME")
    return pd.Series(value, index=idx, name="M2_global_usd")


def _base_config() -> dict:
    return {
        "fred_api_key": "fake_key",
        "m2_countries": ["US", "EA"],
        "m2_source": "bis_fred",
        "m2_lag_days": [1, 7, 14],
        "m2_growth_windows": [7, 30],
        "btc_start_date": "2020-01-01",
        "m2_csv_path": "",
    }


# ── _resample_to_daily ─────────────────────────────────────────────────────────

class TestResampleToDaily:
    def test_output_shape(self):
        monthly = _make_monthly_series(start="2020-01-31", periods=12)
        df = _resample_to_daily(monthly, "2020-01-01", "2020-12-31")
        expected_days = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        assert len(df) == len(expected_days)

    def test_columns(self):
        monthly = _make_monthly_series()
        df = _resample_to_daily(monthly, "2020-01-01", "2020-06-30")
        assert "Date" in df.columns
        assert "M2_global_usd" in df.columns

    def test_forward_fill(self):
        """Values should be constant between reporting months (piecewise constant).

        With month-end observations on Jan 31 (100) and Feb 29 (200):
        - Jan 31 and all Feb days before Feb 29 carry value 100 (forward-filled from Jan 31).
        - Feb 29 and all March days carry value 200 (forward-filled from Feb 29).
        """
        idx = pd.to_datetime(["2020-01-31", "2020-02-29"])
        s = pd.Series([100.0, 200.0], index=idx, name="M2_global_usd")
        df = _resample_to_daily(s, "2020-01-01", "2020-03-10")
        # All March days should carry the Feb 29 value (200)
        mar_days = df[df["Date"].dt.month == 3]
        assert (mar_days["M2_global_usd"] == 200.0).all()
        # All Feb days before the 29th should still carry the Jan 31 value (100)
        feb_before_end = df[(df["Date"].dt.month == 2) & (df["Date"].dt.day < 29)]
        assert (feb_before_end["M2_global_usd"] == 100.0).all()

    def test_no_nans_after_first_observation(self):
        monthly = _make_monthly_series(start="2020-01-31", periods=6)
        df = _resample_to_daily(monthly, "2020-01-01", "2020-06-30")
        # The period before the first monthly observation (before 2020-01-31) may be NaN
        # but after the first observation everything should be filled
        after_first = df[df["Date"] >= pd.Timestamp("2020-01-31")]
        assert after_first["M2_global_usd"].isna().sum() == 0


# ── _add_growth_features ───────────────────────────────────────────────────────

class TestAddGrowthFeatures:
    def _base_df(self) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        vals = np.linspace(1000, 2000, 200)
        return pd.DataFrame({"Date": dates, "M2_global_usd": vals})

    def test_columns_added(self):
        df = _add_growth_features(self._base_df(), [7, 30])
        assert "m2_7d_chg" in df.columns
        assert "m2_30d_chg" in df.columns

    def test_does_not_mutate_input(self):
        df = self._base_df()
        df_copy = df.copy()
        _add_growth_features(df, [7])
        pd.testing.assert_frame_equal(df, df_copy)

    def test_growth_calculation(self):
        df = self._base_df()
        result = _add_growth_features(df, [7])
        # pct_change(7) of row 7 vs row 0
        expected = (df["M2_global_usd"].iloc[7] - df["M2_global_usd"].iloc[0]) / df["M2_global_usd"].iloc[0]
        assert abs(result["m2_7d_chg"].iloc[7] - expected) < 1e-10


# ── _add_lagged_features ───────────────────────────────────────────────────────

class TestAddLaggedFeatures:
    def _base_df(self) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=150, freq="D")
        vals = np.arange(150, dtype=float)
        return pd.DataFrame({"Date": dates, "M2_global_usd": vals})

    def test_lag_columns_added(self):
        df = _add_lagged_features(self._base_df(), [1, 7])
        assert "M2_global_usd_lag_1" in df.columns
        assert "M2_global_usd_lag_7" in df.columns

    def test_lag_values_correct(self):
        df = _add_lagged_features(self._base_df(), [1])
        # Row 10 lag_1 should equal row 9 original
        assert df["M2_global_usd_lag_1"].iloc[10] == df["M2_global_usd"].iloc[9]

    def test_first_lag_rows_are_nan(self):
        df = _add_lagged_features(self._base_df(), [7])
        assert df["M2_global_usd_lag_7"].iloc[:7].isna().all()


# ── _fetch_fred (mocked) ───────────────────────────────────────────────────────

class TestFetchFred:
    def _mock_response(self, observations: list) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"observations": observations}
        return resp

    def test_returns_series_with_correct_values(self):
        obs = [
            {"date": "2020-01-01", "value": "1000.5"},
            {"date": "2020-02-01", "value": "1010.0"},
        ]
        with patch("pipeline.m2_liquidity.requests.get", return_value=self._mock_response(obs)):
            s = _fetch_fred("M2SL", "fake_key")
        assert len(s) == 2
        assert s.iloc[0] == pytest.approx(1000.5)
        assert s.iloc[1] == pytest.approx(1010.0)

    def test_missing_value_dot_excluded(self):
        obs = [
            {"date": "2020-01-01", "value": "."},   # FRED missing
            {"date": "2020-02-01", "value": "900.0"},
        ]
        with patch("pipeline.m2_liquidity.requests.get", return_value=self._mock_response(obs)):
            s = _fetch_fred("M2SL", "fake_key")
        assert len(s) == 1

    def test_raises_without_api_key(self):
        with pytest.raises(ValueError, match="FRED API key"):
            _fetch_fred("M2SL", "")

    def test_returns_empty_series_on_network_error(self):
        with patch("pipeline.m2_liquidity.requests.get", side_effect=Exception("timeout")):
            s = _fetch_fred("M2SL", "fake_key")
        assert s.empty


# ── _country_m2_usd (mocked) ───────────────────────────────────────────────────

class TestCountryM2Usd:
    def _make_monthly_fred_series(self, value: float, n: int = 12) -> pd.Series:
        idx = pd.date_range("2020-01-31", periods=n, freq="ME")
        return pd.Series(value, index=idx)

    def test_us_no_fx(self):
        """US M2 should not require FX conversion."""
        with patch("pipeline.m2_liquidity._fetch_fred", return_value=self._make_monthly_fred_series(1000.0)):
            s = _country_m2_usd("US", "fake_key")
        assert not s.empty
        assert s.name == "m2_US_usd"

    def test_unknown_country_raises(self):
        with pytest.raises(ValueError, match="Unknown country"):
            _country_m2_usd("ZZ", "fake_key")

    def test_empty_m2_returns_empty(self):
        with patch("pipeline.m2_liquidity._fetch_fred", return_value=pd.Series(dtype=float)):
            s = _country_m2_usd("US", "fake_key")
        assert s.empty

    def test_ea_with_fx(self):
        """Euro-area M2 should be converted using EURUSD FX rate."""
        m2_eur = self._make_monthly_fred_series(1_000_000.0)   # millions EUR
        fx_eur = self._make_monthly_fred_series(1.1)            # USD per EUR

        call_map = {
            "MABMM301EZM189S": m2_eur,
            "EXUSEU": fx_eur,
        }

        def fake_fetch(series_id, api_key):
            return call_map.get(series_id, pd.Series(dtype=float))

        with patch("pipeline.m2_liquidity._fetch_fred", side_effect=fake_fetch):
            s = _country_m2_usd("EA", "fake_key")

        # Expected: (1_000_000 * 1e-3) * 1.1 = 1100 billions USD
        assert not s.empty
        assert s.iloc[0] == pytest.approx(1100.0)

    def test_date_alignment_first_of_month_m2_with_daily_fx(self):
        """Regression: FRED monthly M2 uses first-of-month dates; daily FX
        resampled to month-end must still align correctly (the core bug)."""
        # M2 series with first-of-month dates (as FRED actually returns)
        m2_first_of_month = pd.Series(
            [1_000_000.0] * 12,
            index=pd.date_range("2020-01-01", periods=12, freq="MS"),
        )
        # FX series with daily dates (as FRED actually returns for exchange rates)
        daily_idx = pd.date_range("2020-01-01", periods=366, freq="D")
        fx_daily = pd.Series(1.1, index=daily_idx)

        call_map = {
            "MABMM301EZM189S": m2_first_of_month,
            "EXUSEU": fx_daily,
        }

        def fake_fetch(series_id, api_key):
            return call_map.get(series_id, pd.Series(dtype=float))

        with patch("pipeline.m2_liquidity._fetch_fred", side_effect=fake_fetch):
            s = _country_m2_usd("EA", "fake_key")

        # Must NOT be empty – this was the bug (inner join on mismatched dates → 0 rows)
        assert not s.empty, (
            "Date-alignment bug: M2 first-of-month dates did not align with "
            "FX month-end dates after resampling"
        )
        assert s.iloc[0] == pytest.approx(1100.0)


# ── _load_from_csv ─────────────────────────────────────────────────────────────

class TestLoadFromCsv:
    def _write_csv(self, tmp_dir: str, content: str) -> str:
        path = os.path.join(tmp_dir, "global_m2.csv")
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_basic_load(self, tmp_path):
        content = "Date,M2_global_usd\n2020-01-31,1000\n2020-02-29,1010\n"
        path = self._write_csv(str(tmp_path), content)
        config = {"m2_csv_path": path}
        df = _load_from_csv(config, "2020-01-01", "2020-03-31")
        assert "M2_global_usd" in df.columns
        assert "Date" in df.columns
        assert len(df) > 0

    def test_missing_file_raises(self, tmp_path):
        config = {"m2_csv_path": str(tmp_path / "nonexistent.csv")}
        with pytest.raises(FileNotFoundError):
            _load_from_csv(config, "2020-01-01", "2020-12-31")

    def test_missing_column_raises(self, tmp_path):
        content = "Date,wrong_col\n2020-01-31,1000\n"
        path = self._write_csv(str(tmp_path), content)
        config = {"m2_csv_path": path}
        with pytest.raises(ValueError, match="M2_global_usd"):
            _load_from_csv(config, "2020-01-01", "2020-12-31")


# ── load_or_build_m2_series ────────────────────────────────────────────────────

class TestLoadOrBuildM2Series:
    def test_csv_path(self, tmp_path):
        content = "Date,M2_global_usd\n2020-01-31,5000\n2020-02-29,5100\n2020-03-31,5200\n"
        csv_path = tmp_path / "global_m2.csv"
        csv_path.write_text(content)

        config = {
            "m2_source": "csv",
            "m2_csv_path": str(csv_path),
            "m2_lag_days": [1, 7],
            "m2_growth_windows": [7],
            "btc_start_date": "2020-01-01",
        }
        df = load_or_build_m2_series(config, start_date="2020-01-01", end_date="2020-04-30")
        assert "Date" in df.columns
        assert "M2_global_usd" in df.columns
        assert "M2_global_usd_lag_1" in df.columns
        assert "M2_global_usd_lag_7" in df.columns
        assert "m2_7d_chg" in df.columns

    def test_bis_fred_path_mocked(self):
        """Test that the bis_fred path calls _build_global_m2_monthly correctly."""
        monthly = _make_monthly_series("2020-01-31", 24)

        config = {
            "m2_source": "bis_fred",
            "fred_api_key": "fake_key",
            "m2_countries": ["US"],
            "m2_lag_days": [1],
            "m2_growth_windows": [7],
            "btc_start_date": "2020-01-01",
        }

        with patch("pipeline.m2_liquidity._build_global_m2_monthly", return_value=monthly):
            df = load_or_build_m2_series(config, "2020-01-01", "2021-12-31")

        assert not df.empty
        assert "M2_global_usd_lag_1" in df.columns

    def test_all_lag_columns_present(self, tmp_path):
        content = "Date,M2_global_usd\n" + "\n".join(
            f"2020-{m:02d}-01,{1000 + m * 10}" for m in range(1, 13)
        )
        csv_path = tmp_path / "global_m2.csv"
        csv_path.write_text(content)

        lag_days = [1, 7, 14, 21, 28]
        config = {
            "m2_source": "csv",
            "m2_csv_path": str(csv_path),
            "m2_lag_days": lag_days,
            "m2_growth_windows": [7, 30],
            "btc_start_date": "2020-01-01",
        }
        df = load_or_build_m2_series(config, "2020-01-01", "2020-12-31")
        for lag in lag_days:
            assert f"M2_global_usd_lag_{lag}" in df.columns, f"Missing lag column: lag_{lag}"


# ── _build_global_m2_monthly ───────────────────────────────────────────────────

class TestBuildGlobalM2Monthly:
    def _mock_country(self, value: float = 1000.0) -> pd.Series:
        idx = pd.date_range("2020-01-31", periods=24, freq="ME")
        return pd.Series(value, index=idx)

    def test_sums_countries(self):
        config = {
            "fred_api_key": "fake_key",
            "m2_countries": ["US", "EA"],
        }
        with patch("pipeline.m2_liquidity._country_m2_usd", return_value=self._mock_country(1000.0)):
            result = _build_global_m2_monthly(config)
        # Result is now a DataFrame with M2_global_usd (sum) and per-country columns
        assert isinstance(result, pd.DataFrame)
        assert "M2_global_usd" in result.columns
        # 2 countries × 1000 each → global sum ≥ 1500
        assert (result["M2_global_usd"] >= 1500).all()

    def test_raises_when_all_countries_fail(self):
        config = {
            "fred_api_key": "fake_key",
            "m2_countries": ["US"],
        }
        with patch("pipeline.m2_liquidity._country_m2_usd", return_value=pd.Series(dtype=float)):
            with pytest.raises(RuntimeError, match="Could not obtain M2 data"):
                _build_global_m2_monthly(config)

    def test_per_country_columns_present(self):
        """DataFrame must include individual country columns alongside the global sum."""
        config = {
            "fred_api_key": "fake_key",
            "m2_countries": ["US", "EA"],
        }

        def side_effect(country, api_key):
            s = self._mock_country(1000.0)
            s.name = f"m2_{country}_usd"
            return s

        with patch("pipeline.m2_liquidity._country_m2_usd", side_effect=side_effect):
            result = _build_global_m2_monthly(config)

        assert "M2_global_usd" in result.columns
        assert "m2_US_usd" in result.columns
        assert "m2_EA_usd" in result.columns

    def test_partial_country_failure_is_logged(self, caplog):
        import logging
        config = {
            "fred_api_key": "fake_key",
            "m2_countries": ["US", "EA"],
        }
        call_count = {"n": 0}

        def side_effect(country, api_key):
            call_count["n"] += 1
            if country == "EA":
                return pd.Series(dtype=float)   # EA fails
            return self._mock_country(1000.0)

        with caplog.at_level(logging.WARNING, logger="pipeline.m2_liquidity"):
            with patch("pipeline.m2_liquidity._country_m2_usd", side_effect=side_effect):
                result = _build_global_m2_monthly(config)

        assert "EA" in caplog.text
        assert not result.empty
