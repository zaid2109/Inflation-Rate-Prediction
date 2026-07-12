"""Unit tests for evaluation metrics."""

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot is imported anywhere

import pytest
import numpy as np
import pandas as pd
from src.evaluate import (
    compute_metrics, mape, directional_accuracy, compare_models, diebold_mariano,
    naive_baseline_metrics, bootstrap_confidence_interval, run_diebold_mariano_suite,
    plot_predictions, plot_metrics_heatmap,
)


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


def test_naive_baseline_metrics():
    y_true = np.array([2.0, 2.5, 2.2, 2.8, 3.0])
    m = naive_baseline_metrics(y_true)
    assert set(m.keys()) == {"MAE", "RMSE", "MAPE", "R2", "Dir_Acc"}
    # Persistence prediction for row i is y_true[i-1]; MAE should equal the
    # mean absolute first difference over the evaluated window.
    expected_mae = float(np.mean(np.abs(np.diff(y_true))))
    assert abs(m["MAE"] - expected_mae) < 1e-6


def test_bootstrap_confidence_interval_perfect_fit():
    y = np.array([2.0, 2.5, 2.2, 2.8, 3.0, 2.9, 2.4])
    ci = bootstrap_confidence_interval(y, y, n_boot=200)
    assert ci["ci_half_width"] == 0.0
    assert ci["residual_std"] == 0.0
    assert ci["ci_level"] == 0.90


def test_bootstrap_confidence_interval_keys_and_bounds():
    rng = np.random.default_rng(0)
    y_true = rng.normal(3.0, 1.0, 50)
    y_pred = y_true + rng.normal(0, 0.3, 50)
    ci = bootstrap_confidence_interval(y_true, y_pred, n_boot=200)
    assert set(ci.keys()) == {"ci_lower", "ci_upper", "ci_half_width", "residual_std", "ci_level"}
    assert ci["ci_lower"] <= ci["ci_upper"]
    assert ci["residual_std"] > 0


def test_run_diebold_mariano_suite():
    rng = np.random.default_rng(1)
    y = rng.normal(3.0, 1.0, 40)
    results = {
        "baseline": {"y_true": y, "y_pred": y + rng.normal(0, 1.0, 40)},
        "better":   {"y_true": y, "y_pred": y + rng.normal(0, 0.1, 40)},
    }
    df = run_diebold_mariano_suite(results, baseline="baseline")
    assert list(df.columns) == ["Model", "vs", "DM Stat", "p-value", "n", "Conclusion"]
    assert df.iloc[0]["Model"] == "better"


def test_run_diebold_mariano_suite_missing_baseline():
    with pytest.raises(ValueError):
        run_diebold_mariano_suite({"a": {"y_true": [1], "y_pred": [1]}}, baseline="missing")


def test_plot_predictions_smoke(tmp_path, monkeypatch):
    import src.evaluate as ev
    monkeypatch.setattr(ev, "FIGURES_DIR", tmp_path)
    dates = pd.date_range("2020-01-01", periods=5, freq="MS")
    y_true = np.array([2.0, 2.2, 2.5, 2.4, 2.6])
    preds = {"Model A": y_true + 0.1}
    plot_predictions(dates, y_true, preds, save=True)
    assert (tmp_path / "predictions.png").exists()


def test_plot_metrics_heatmap_smoke(tmp_path, monkeypatch):
    import src.evaluate as ev
    monkeypatch.setattr(ev, "FIGURES_DIR", tmp_path)
    df = pd.DataFrame({
        "MAE": [0.5, 0.8], "RMSE": [0.7, 1.1], "MAPE": [10.0, 15.0],
        "R2": [0.8, 0.6], "Dir_Acc": [70.0, 65.0],
    }, index=["ModelA", "ModelB"])
    plot_metrics_heatmap(df, save=True)
    assert (tmp_path / "metrics_heatmap.png").exists()
