"""Walk-forward expanding-window validation.

Trains on all data up to month t, predicts t+1, advances one month.
Uses TimeSeriesSplit(n_splits=10) to produce out-of-sample predictions
that simulate real deployment conditions.
"""

import warnings
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
import xgboost as xgb

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent.parent / "models"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    nz   = y_true != 0
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = float(1 - ss_res / (ss_tot + 1e-12))
    da   = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4), "R2": round(r2, 4), "Dir_Acc": round(da, 2)}


def run_walkforward(X: pd.DataFrame, y: pd.Series, n_splits: int = 10) -> dict:
    """
    Expanding-window walk-forward validation using TimeSeriesSplit.

    At each fold the model is trained on all past data and evaluated on
    a future held-out window — exactly how a deployed system would work.

    Returns a dict with per-model out-of-sample predictions and metrics.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X_arr = X.values
    y_arr = y.values

    oof_ridge  = np.full(len(y), np.nan)
    oof_xgb    = np.full(len(y), np.nan)
    test_idx_all = []

    for train_idx, test_idx in tscv.split(X_arr):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr       = y_arr[train_idx]

        # Ridge
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
                        cv=TimeSeriesSplit(n_splits=3))
        ridge.fit(X_tr_s, y_tr)
        oof_ridge[test_idx] = ridge.predict(X_te_s)

        # XGBoost (use best hyperparams from full training to save time)
        best_params = {}
        p = MODELS_DIR / "xgboost.joblib"
        if p.exists():
            saved = joblib.load(p)
            best_params = saved.get("best_params", {})
        model_xgb = xgb.XGBRegressor(
            objective="reg:squarederror", random_state=42, verbosity=0,
            **best_params
        )
        model_xgb.fit(X_tr, y_tr)
        oof_xgb[test_idx] = model_xgb.predict(X_te)

        test_idx_all.extend(test_idx.tolist())

    # Only evaluate on rows that appeared in at least one test fold
    valid = ~np.isnan(oof_ridge)
    dates  = [str(d.date()) for d in y.index[valid]]
    actual = y_arr[valid]

    results = {
        "dates":  dates,
        "actual": actual.tolist(),
        "ridge": {
            "predictions": oof_ridge[valid].tolist(),
            "metrics": _metrics(actual, oof_ridge[valid]),
        },
        "xgboost": {
            "predictions": oof_xgb[valid].tolist(),
            "metrics": _metrics(actual, oof_xgb[valid]),
        },
    }

    # Save results for API/dashboard consumption
    out = MODELS_DIR / "walkforward_results.json"
    with open(out, "w") as f:
        json.dump(results, f)
    print(f"Walk-forward results saved → {out}")
    return results


def print_table(results: dict) -> None:
    print(f"\n{'Model':<12} {'MAE':>6} {'RMSE':>6} {'MAPE%':>7} {'R2':>7} {'DirAcc%':>8}")
    print("-" * 52)
    for name in ["ridge", "xgboost"]:
        m = results[name]["metrics"]
        print(f"{name:<12} {m['MAE']:>6.3f} {m['RMSE']:>6.3f} "
              f"{m['MAPE']:>7.2f} {m['R2']:>7.4f} {m['Dir_Acc']:>8.1f}")


if __name__ == "__main__":
    from src.features import load_processed, get_xy
    df = load_processed()
    X, y = get_xy(df)
    print(f"Running walk-forward validation on {len(df)} months …")
    results = run_walkforward(X, y, n_splits=10)
    print_table(results)
