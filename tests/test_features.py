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


def test_seasonal_dummies():
    raw = _make_raw()
    df = build_feature_matrix(raw)
    for m in range(1, 13):
        assert f"month_{m}" in df.columns
    # Exactly one month dummy is 1 per row
    month_cols = [f"month_{m}" for m in range(1, 13)]
    assert (df[month_cols].sum(axis=1) == 1).all()


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
