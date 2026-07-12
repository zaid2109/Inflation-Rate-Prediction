"""Standalone/experimental ensemble stacking — Level-0: Ridge + XGBoost; Level-1: Ridge meta-model.

NOT part of the main pipeline. src/train.py::train_ensemble() is the canonical
implementation whose output (models/ensemble.joblib) is what api/main.py,
dashboard/app.py, and src/spf_comparison.py actually load. This file uses a
different bundle schema ({"ridge_pipe", "xgb_model", "meta_scaler", ...} instead
of {"linear_bundle", "xgb_bundle", ...}), so running it used to silently
overwrite models/ensemble.joblib with a bundle the rest of the app can't read
(KeyError at inference time). It now writes to a separate file so it can be
run for experimentation without breaking the API/dashboard.

Out-of-fold (OOF) predictions from the base models are used to train the
meta-model, so no test-set information leaks into the stacker.

Saves models/ensemble_standalone.joblib with the fitted meta-model and base bundles.
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
from sklearn.pipeline import make_pipeline
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


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    n_splits: int = 5,
) -> dict:
    """
    Build a 2-level stacking ensemble.

    Level-0 base models (Ridge, XGBoost) produce out-of-fold training
    predictions that form the meta-feature matrix for Level-1.
    The full training set is used to re-train base models for test inference.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # ── Level-0: OOF predictions on training set ──────────────────────────────
    print("Generating OOF predictions for ensemble …")

    # Ridge pipeline (scaler inside)
    ridge_pipe = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
                cv=TimeSeriesSplit(n_splits=3))
    )
    oof_ridge = cross_val_predict(ridge_pipe, X_train, y_train, cv=tscv)

    # XGBoost with best hyperparams from saved model
    best_params = {}
    p = MODELS_DIR / "xgboost.joblib"
    if p.exists():
        best_params = joblib.load(p).get("best_params", {})
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=42, verbosity=0,
        **best_params
    )
    oof_xgb = cross_val_predict(xgb_model, X_train, y_train, cv=tscv)

    # Meta-features: stack OOF columns
    meta_train = np.column_stack([oof_ridge, oof_xgb])

    # ── Level-1: Ridge meta-model ─────────────────────────────────────────────
    meta_scaler = StandardScaler()
    meta_X = meta_scaler.fit_transform(meta_train)
    meta_model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5)
    meta_model.fit(meta_X, y_train.values)
    print(f"  Meta-model alpha: {meta_model.alpha_:.4f}")

    # ── Re-train base models on full training set ─────────────────────────────
    ridge_pipe.fit(X_train, y_train)

    xgb_full = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=42, verbosity=0,
        **best_params
    )
    xgb_full.fit(X_train, y_train)

    # ── Test-set predictions ──────────────────────────────────────────────────
    test_ridge = ridge_pipe.predict(X_test)
    test_xgb   = xgb_full.predict(X_test)

    meta_test  = meta_scaler.transform(np.column_stack([test_ridge, test_xgb]))
    test_preds = meta_model.predict(meta_test)

    metrics = _metrics(y_test.values, test_preds)
    print(f"  Ensemble test metrics: {metrics}")

    # ── Save bundle ───────────────────────────────────────────────────────────
    bundle = {
        "meta_model":   meta_model,
        "meta_scaler":  meta_scaler,
        "ridge_pipe":   ridge_pipe,
        "xgb_model":    xgb_full,
        "metrics":      metrics,
    }
    joblib.dump(bundle, MODELS_DIR / "ensemble_standalone.joblib")
    print(f"Ensemble saved → {MODELS_DIR / 'ensemble_standalone.joblib'}")

    # Save metrics (separate file — NOT read by the API; see module docstring)
    with open(MODELS_DIR / "ensemble_standalone_metrics.json", "w") as f:
        json.dump(metrics, f)

    return {
        "bundle":  bundle,
        "y_pred":  test_preds,
        "metrics": metrics,
    }


def predict_ensemble(X: pd.DataFrame) -> np.ndarray:
    """Run inference with the saved standalone ensemble bundle (see module docstring)."""
    bundle = joblib.load(MODELS_DIR / "ensemble_standalone.joblib")
    r = bundle["ridge_pipe"].predict(X)
    x = bundle["xgb_model"].predict(X)
    meta = bundle["meta_scaler"].transform(np.column_stack([r, x]))
    return bundle["meta_model"].predict(meta)


if __name__ == "__main__":
    from src.features import load_processed, split, get_xy
    df = load_processed()
    train, test = split(df)
    X_train, y_train = get_xy(train)
    X_test,  y_test  = get_xy(test)
    result = train_ensemble(X_train, y_train, X_test, y_test)
    print("\nEnsemble vs single models on test set:")
    print(f"  Ensemble: MAE={result['metrics']['MAE']:.3f}  "
          f"RMSE={result['metrics']['RMSE']:.3f}  "
          f"DirAcc={result['metrics']['Dir_Acc']:.1f}%")
