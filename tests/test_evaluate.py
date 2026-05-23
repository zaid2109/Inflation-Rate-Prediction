"""Unit tests for evaluation metrics."""

import pytest
import numpy as np
from src.evaluate import compute_metrics, mape, directional_accuracy


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
