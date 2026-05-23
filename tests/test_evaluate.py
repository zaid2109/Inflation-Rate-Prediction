"""Unit tests for evaluation metrics."""

import pytest
import numpy as np
import pandas as pd
from src.evaluate import compute_metrics, mape, directional_accuracy, compare_models, diebold_mariano


def test_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0, 2.0, 1.5])
    m = compute_metrics(y, y)
    assert m["MAE"]  == 0.0
    assert m["RMSE"] == 0.0
    assert m["R2"]   == 1.0


def test_mape_basic():
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([2.5, 4.0, 5.0])
    result = mape(y_true, y_pred)
    # |2.0-2.5|/2.0=0.25, |4.0-4.0|/4.0=0.0, |6.0-5.0|/6.0≈0.1667 → mean*100≈13.89
    expected = (0.5 / 2.0 + 0.0 / 4.0 + 1.0 / 6.0) / 3 * 100
    assert abs(result - expected) < 0.01


def test_directional_accuracy_perfect():
    y_true = np.array([1.0, 2.0, 3.0, 2.0])
    y_pred = np.array([0.5, 1.5, 2.5, 1.5])
    assert directional_accuracy(y_true, y_pred) == 100.0


def test_directional_accuracy_worst():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([3.0, 2.0, 1.0])
    assert directional_accuracy(y_true, y_pred) == 0.0


def test_metrics_keys():
    y = np.array([2.1, 3.3, 2.8, 3.5])
    p = np.array([2.2, 3.1, 3.0, 3.3])
    m = compute_metrics(y, p)
    assert set(m.keys()) == {"MAE", "RMSE", "MAPE", "R2", "Dir_Acc"}


def test_compare_models_returns_dataframe():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    results = {
        "ModelA": {"y_true": y, "y_pred": y + 0.1},
        "ModelB": {"y_true": y, "y_pred": y + 0.5},
    }
    df = compare_models(results)
    assert set(df.index) == {"ModelA", "ModelB"}
    assert "MAE" in df.columns
    assert "RMSE" in df.columns
    assert df.loc["ModelA", "MAE"] < df.loc["ModelB", "MAE"]


def test_compare_models_columns():
    y = np.array([2.0, 3.0, 4.0])
    results = {"M": {"y_true": y, "y_pred": y}}
    df = compare_models(results)
    assert list(df.columns) == ["MAE", "RMSE", "MAPE", "R2", "Dir_Acc"]


def test_diebold_mariano_identical():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.0, 1.0])
    result = diebold_mariano(y, y, y)
    assert result["dm_stat"] == 0.0
    assert result["p_value"] == 1.0


def test_diebold_mariano_clear_winner():
    rng = np.random.default_rng(42)
    y = rng.normal(3.0, 1.0, 60)
    y_good = y + rng.normal(0, 0.1, 60)
    y_bad  = y + rng.normal(0, 2.0, 60)
    result = diebold_mariano(y, y_good, y_bad)
    assert "dm_stat" in result
    assert "p_value" in result
    assert "conclusion" in result
    assert result["n"] == 60


def test_diebold_mariano_returns_dict_keys():
    y = np.linspace(1, 5, 40)
    res = diebold_mariano(y, y + 0.1, y + 0.5)
    assert set(res.keys()) == {"dm_stat", "p_value", "n", "conclusion"}


def test_mape_excludes_zeros():
    y_true = np.array([0.0, 2.0, 4.0])
    y_pred = np.array([1.0, 2.5, 4.0])
    result = mape(y_true, y_pred)
    # Zero in y_true must be excluded; only indices 1 and 2 count
    expected = (0.5 / 2.0 + 0.0 / 4.0) / 2 * 100
    assert abs(result - expected) < 0.01
