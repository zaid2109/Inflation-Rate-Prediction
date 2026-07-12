"""Unit tests for feature engineering."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import build_feature_matrix, split, get_xy


def _make_raw(n: int = 420) -> dict:
    dates = pd.date_range("1990-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(42)
    return {
        "cpi":          pd.Series(250 + np.cumsum(rng.normal(0.3, 0.2, n)), index=dates, name="cpi"),
        "m2":           pd.Series(8000 + np.cumsum(rng.normal(50, 10, n)), index=dates, name="m2"),
        "unemployment": pd.Series(5 + rng.normal(0, 0.5, n), index=dates, name="unemployment"),
        "fed_funds":    pd.Series(2 + rng.normal(0, 0.3, n), index=dates, name="fed_funds"),
        "gdp_growth":   pd.Series(2.5 + rng.normal(0, 0.5, n), index=dates, name="gdp_growth"),
        "oil_wti":      pd.Series(60 + rng.normal(0, 5, n), index=dates, name="oil_wti"),
        "ppi":          pd.Series(200 + np.cumsum(rng.normal(0.2, 0.1, n)), index=dates, name="ppi"),
    }


def test_build_feature_matrix_shape():
    raw = _make_raw()
    df = build_feature_matrix(raw)
    assert len(df) > 0
    assert "inflation_rate" in df.columns
    assert df.isna().sum().sum() == 0


def test_lag_features_present():
    raw = _make_raw()
    df = build_feature_matrix(raw)
    for lag in [1, 3, 6, 12]:
        assert f"cpi_lag{lag}" in df.columns


def test_rolling_features_present():
    raw = _make_raw()
    df = build_feature_matrix(raw)
    for window in [3, 12]:
        assert f"cpi_roll_mean{window}" in df.columns
        assert f"cpi_roll_std{window}" in df.columns


def test_no_seasonal_dummies():
    """Month-of-year dummies were tried and dropped (train-only CV showed they
    added noise, not signal, for a 12-month-differenced YoY target — see README)."""
    raw = _make_raw()
    df = build_feature_matrix(raw)
    assert not any(c.startswith("month_") for c in df.columns)


def test_split_no_leakage():
    raw = _make_raw()
    df = build_feature_matrix(raw)
    train, test = split(df)
    assert train.index.max() < test.index.min()
    assert train.index.max().year <= 2018
    assert test.index.min().year >= 2019


def test_get_xy_target_not_in_features():
    raw = _make_raw()
    df = build_feature_matrix(raw)
    X, y = get_xy(df)
    assert "inflation_rate" not in X.columns
    assert y.name == "inflation_rate"
    assert len(X) == len(y)


def test_no_target_leakage_in_momentum_features():
    """infl_mom3/infl_mom6 must only use inflation_rate up to t-1, never t itself
    (i.e. never a value that is an additive function of the target being predicted)."""
    raw = _make_raw()
    df = build_feature_matrix(raw)
    # Reconstruct the pre-dropna inflation_rate series to check the exact formula.
    full = pd.DataFrame(raw).resample("MS").last().loc["1990-01-01":"2025-12-31"]
    full["inflation_rate"] = full["cpi"].pct_change(12, fill_method=None) * 100
    expected_mom3 = full["inflation_rate"].shift(1).diff(3)
    expected_mom6 = full["inflation_rate"].shift(1).diff(6)
    pd.testing.assert_series_equal(
        df["infl_mom3"], expected_mom3.reindex(df.index), check_names=False
    )
    pd.testing.assert_series_equal(
        df["infl_mom6"], expected_mom6.reindex(df.index), check_names=False
    )
    # infl_mom3(t) must equal inflation_rate(t-1) - inflation_rate(t-4), never
    # involving inflation_rate(t) — this is the leakage that was previously present.
    leaked_mom3 = full["inflation_rate"].diff(3).reindex(df.index)
    assert not df["infl_mom3"].equals(leaked_mom3)


def test_gdp_growth_raw_excluded_from_features():
    """Raw gdp_growth is a ffilled quarterly print that isn't known for the first
    1-2 months of its quarter — only the lagged version belongs in X."""
    raw = _make_raw()
    df = build_feature_matrix(raw)
    X, y = get_xy(df)
    assert "gdp_growth" not in X.columns
    assert "gdp_growth_lag1" in X.columns
