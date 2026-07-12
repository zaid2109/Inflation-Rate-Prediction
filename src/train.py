"""Train all models (Ridge, ARIMA, XGBoost, LSTM, Ensemble) and log to MLflow."""

import warnings
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.meta_models import ConvexBlendMeta

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MLFLOW_URI = (Path(__file__).parent.parent / "mlruns").resolve().as_uri()
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("inflation-predictor")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_metrics(all_metrics: dict) -> None:
    """Persist per-model metrics to models/metrics.json for the dashboard."""
    path = MODELS_DIR / "metrics.json"
    path.write_text(json.dumps(all_metrics, indent=2))


# ---------------------------------------------------------------------------
# Linear Regression (Ridge with cross-validated alpha)
# ---------------------------------------------------------------------------

def train_linear(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = RidgeCV(
        alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
        cv=TimeSeriesSplit(n_splits=5),
    )
    model.fit(X_scaled, y_train)
    bundle = {"model": model, "scaler": scaler}
    joblib.dump(bundle, MODELS_DIR / "linear.joblib")
    return bundle


# ---------------------------------------------------------------------------
# ARIMA / SARIMA
# ---------------------------------------------------------------------------

def train_arima(y_train: pd.Series) -> dict:
    from pmdarima import auto_arima
    model = auto_arima(
        y_train,
        seasonal=True,
        m=12,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=4, max_q=4, max_P=2, max_Q=2,
    )
    bundle = {"model": model}
    joblib.dump(bundle, MODELS_DIR / "arima.joblib")
    return bundle


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    from sklearn.model_selection import RandomizedSearchCV
    tscv = TimeSeriesSplit(n_splits=5)
    # Widened from the original {n_estimators, max_depth, learning_rate, subsample,
    # colsample_bytree} grid to include explicit regularization (reg_alpha, reg_lambda,
    # min_child_weight) — a train-only CV comparison showed the extra regularization
    # knobs meaningfully reduce CV RMSE (~10%) given how few rows (~330) are available
    # relative to the feature count. RandomizedSearchCV is used because the full grid
    # (2*3*3*2*2*3*2*3 = 1296 combos) is too large to grid-search exhaustively at
    # 5-fold CV; 150 random draws covers the space well in reasonable time.
    param_dist = {
        "n_estimators":     [100, 200, 400],
        "max_depth":        [2, 3, 4],
        "learning_rate":    [0.03, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha":        [0, 0.1, 1.0],
        "reg_lambda":       [1.0, 5.0, 10.0],
        "min_child_weight": [1, 5, 10],
    }
    base = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, verbosity=0)
    search = RandomizedSearchCV(base, param_dist, n_iter=150, cv=tscv,
                                 scoring="neg_mean_squared_error",
                                 n_jobs=-1, verbose=0, random_state=42)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    bundle = {"model": model, "best_params": search.best_params_}
    joblib.dump(bundle, MODELS_DIR / "xgboost.joblib")
    return bundle


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def train_lstm(X_train: pd.DataFrame, y_train: pd.Series, lookback: int = 12) -> dict:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    Xs, ys = [], []
    for i in range(lookback, len(X_scaled)):
        Xs.append(X_scaled[i - lookback:i])
        ys.append(y_train.iloc[i])
    Xs, ys = np.array(Xs), np.array(ys)

    n_features = Xs.shape[2]
    # Simpler bidirectional LSTM — less prone to overfitting on small macro datasets
    model = Sequential([
        Input(shape=(lookback, n_features)),
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4))),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(32, return_sequences=False, kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss="mse")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.5, min_lr=1e-6),
    ]
    model.fit(
        Xs, ys,
        epochs=300,
        batch_size=16,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0,
    )
    model.save(MODELS_DIR / "lstm_model.keras")

    # Conformal prediction via split-conformal (MAPIE-style).
    # Use the last 20% of training sequences as a calibration set.
    cal_split = int(len(Xs) * 0.8)
    X_cal, y_cal = Xs[cal_split:], ys[cal_split:]
    preds_cal = model.predict(X_cal, verbose=0).flatten()
    residuals_cal = np.abs(y_cal - preds_cal)
    # 90% conformal quantile (alpha=0.10)
    q_90 = float(np.quantile(residuals_cal, 0.90))
    q_95 = float(np.quantile(residuals_cal, 0.95))

    bundle = {
        "scaler":   scaler,
        "lookback": lookback,
        "conformal_q90": round(q_90, 4),
        "conformal_q95": round(q_95, 4),
    }
    joblib.dump(bundle, MODELS_DIR / "lstm.joblib")
    print(f"  LSTM conformal intervals: ±{q_90:.3f} (90%) | ±{q_95:.3f} (95%)")
    return {"model": model, **bundle}


# ---------------------------------------------------------------------------
# Walk-forward (expanding-window) validation
# ---------------------------------------------------------------------------

def walk_forward_validation(
    df_full: pd.DataFrame,
    target: str = "inflation_rate",
    initial_train_end: str = "2010-12-31",
    test_start: str = "2011-01-01",
) -> dict:
    """
    Expanding-window walk-forward CV on XGBoost (fastest model for WF).
    Returns dict with dates, y_true, y_pred for the walk-forward period.
    """
    from src.features import get_xy

    drop_cols = {target, "cpi", "cpi_mom", "inflation_rate"}
    feat_cols = [c for c in df_full.columns if c not in drop_cols]

    test_df = df_full.loc[test_start:]
    wf_preds, wf_true, wf_dates = [], [], []

    print(f"Walk-forward validation: {len(test_df)} steps from {test_start}...")
    for i, date in enumerate(test_df.index):
        train_df = df_full.loc[:date].iloc[:-1]  # all data strictly before this date
        if len(train_df) < 60:
            continue
        X_tr = train_df[feat_cols]
        y_tr = train_df[target]
        X_te = test_df.loc[[date], feat_cols]
        y_te = float(test_df.loc[date, target])

        mdl = xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42, verbosity=0,
        )
        mdl.fit(X_tr, y_tr)
        pred = float(mdl.predict(X_te)[0])
        wf_preds.append(pred)
        wf_true.append(y_te)
        wf_dates.append(str(date.date()))

        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(test_df)} steps done")

    result = {"dates": wf_dates, "y_true": wf_true, "y_pred": wf_preds}
    path = MODELS_DIR / "walk_forward.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"Walk-forward saved -> {path}")
    return result


# ---------------------------------------------------------------------------
# Ensemble stacking (Level-0: Linear, XGBoost out-of-fold;
#                    Level-1: constrained convex-blend meta-learner)
# ---------------------------------------------------------------------------

def _select_blend_weight(oof_a: np.ndarray, oof_b: np.ndarray, y: np.ndarray,
                          n_grid: int = 101) -> tuple[float, float]:
    """Grid-search w in [0, 1] minimizing RMSE of w*a + (1-w)*b against y."""
    best_w, best_rmse = 0.5, np.inf
    for w in np.linspace(0, 1, n_grid):
        pred = w * oof_a + (1 - w) * oof_b
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w
    return best_w, best_rmse


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    linear_bundle: dict,
    xgb_bundle: dict,
    arima_bundle: dict,
    lstm_metrics: dict | None = None,
    linear_metrics: dict | None = None,
) -> dict:
    """
    Stacking ensemble (Level-0: Ridge + XGBoost; Level-1: constrained convex blend).
    LSTM is excluded from the ensemble — it uses a different input shape and
    historically underperforms the tabular models on this dataset.
    Meta-model weights are saved as ensemble weights for transparency.
    """
    tscv = TimeSeriesSplit(n_splits=5)

    scaler_lin = linear_bundle["scaler"]
    X_scaled   = scaler_lin.transform(X_train)
    alpha_best = float(linear_bundle["model"].alpha_)
    bp = xgb_bundle["best_params"]

    oof_linear = np.zeros(len(y_train))
    oof_xgb    = np.zeros(len(y_train))

    for tr_idx, val_idx in tscv.split(X_scaled):
        ridge_fold = Ridge(alpha=alpha_best)
        ridge_fold.fit(X_scaled[tr_idx], y_train.iloc[tr_idx])
        oof_linear[val_idx] = ridge_fold.predict(X_scaled[val_idx])

        xgb_fold = xgb.XGBRegressor(
            objective="reg:squarederror", random_state=42, verbosity=0, **bp,
        )
        xgb_fold.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_xgb[val_idx] = xgb_fold.predict(X_train.iloc[val_idx])

    # Meta-learner: constrained convex blend, not unconstrained Ridge. See
    # ConvexBlendMeta docstring — with r~0.8 correlated OOF predictions,
    # unconstrained regression produces unstable, sign-flipped coefficients
    # that don't generalize; a plain non-negative, sum-to-1 blend does.
    best_w, blend_rmse = _select_blend_weight(oof_linear, oof_xgb, y_train.values)
    meta_model = ConvexBlendMeta(best_w)

    weights = {
        "Linear Regression": round(best_w, 4),
        "XGBoost":            round(1 - best_w, 4),
        "LSTM Neural Net":    0.0,   # excluded from ensemble
        "oof_blend_rmse":     round(blend_rmse, 4),
    }
    (MODELS_DIR / "ensemble_weights.json").write_text(json.dumps(weights, indent=2))
    print(f"  Ensemble weights: {weights}")

    bundle = {
        "meta_model":    meta_model,
        "linear_bundle": linear_bundle,
        "xgb_bundle":    xgb_bundle,
        "weights":       weights,
    }
    joblib.dump(bundle, MODELS_DIR / "ensemble.joblib")
    return bundle


def predict_ensemble_from_bundle(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    scaler = bundle["linear_bundle"]["scaler"]
    pred_lin = bundle["linear_bundle"]["model"].predict(scaler.transform(X))
    pred_xgb = bundle["xgb_bundle"]["model"].predict(X)
    meta_X = np.column_stack([pred_lin, pred_xgb])
    return bundle["meta_model"].predict(meta_X)


# ---------------------------------------------------------------------------
# SHAP persistence
# ---------------------------------------------------------------------------

def save_shap(xgb_model, X_test: pd.DataFrame, test_index) -> None:
    import shap
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    np.save(MODELS_DIR / "shap_values.npy", shap_values)
    (MODELS_DIR / "shap_dates.json").write_text(
        json.dumps([str(d.date()) for d in test_index])
    )
    # Top-15 mean absolute SHAP per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    top = sorted(zip(X_test.columns, mean_shap.tolist()),
                 key=lambda x: x[1], reverse=True)[:15]
    (MODELS_DIR / "shap_summary.json").write_text(
        json.dumps([{"feature": f, "shap": round(v, 6)} for f, v in top])
    )
    print("SHAP saved.")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    run_walk_forward: bool = False,
    df_full: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from src.evaluate import compute_metrics, compare_models, naive_baseline_metrics, bootstrap_confidence_interval

    # Persist column order so the API can reindex inference rows
    feature_cols = list(X_train.columns)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    results = {}
    all_metrics = {}

    # --- Linear ---
    print("Training Linear Regression...")
    with mlflow.start_run(run_name="linear"):
        bundle_lin = train_linear(X_train, y_train)
        X_test_scaled = bundle_lin["scaler"].transform(X_test)
        y_pred_lin = bundle_lin["model"].predict(X_test_scaled)
        metrics = compute_metrics(y_test.values, y_pred_lin)
        mlflow.log_metrics(metrics)
        mlflow.log_param("alpha", bundle_lin["model"].alpha_)
        results["Linear"] = {"y_true": y_test.values, "y_pred": y_pred_lin, "metrics": metrics}
        all_metrics["linear"] = metrics
    print(f"  Linear: {metrics}")

    # --- ARIMA ---
    print("Training ARIMA...")
    with mlflow.start_run(run_name="arima"):
        bundle_arima = train_arima(y_train)
        y_pred_arima = bundle_arima["model"].predict(n_periods=len(y_test))
        metrics = compute_metrics(y_test.values, y_pred_arima)
        mlflow.log_metrics(metrics)
        mlflow.log_param("order", str(bundle_arima["model"].order))
        results["ARIMA"] = {"y_true": y_test.values, "y_pred": y_pred_arima, "metrics": metrics}
        all_metrics["arima"] = metrics
    print(f"  ARIMA: {metrics}")

    # --- XGBoost ---
    print("Training XGBoost (grid search)...")
    with mlflow.start_run(run_name="xgboost"):
        bundle_xgb = train_xgboost(X_train, y_train)
        y_pred_xgb = bundle_xgb["model"].predict(X_test)
        metrics = compute_metrics(y_test.values, y_pred_xgb)
        mlflow.log_metrics(metrics)
        mlflow.log_params(bundle_xgb["best_params"])
        results["XGBoost"] = {"y_true": y_test.values, "y_pred": y_pred_xgb, "metrics": metrics}
        all_metrics["xgboost"] = metrics
    print(f"  XGBoost: {metrics}")

    # Save SHAP values for test set
    save_shap(bundle_xgb["model"], X_test, y_test.index)

    # --- LSTM ---
    print("Training LSTM...")
    with mlflow.start_run(run_name="lstm"):
        bundle_lstm = train_lstm(X_train, y_train)
        scaler_lstm = bundle_lstm["scaler"]
        model_lstm  = bundle_lstm["model"]
        lookback    = bundle_lstm["lookback"]
        X_full_scaled = scaler_lstm.transform(pd.concat([X_train, X_test]))
        Xs_test = np.array([
            X_full_scaled[len(X_train) + j - lookback: len(X_train) + j]
            for j in range(len(X_test))
        ])
        y_pred_lstm = model_lstm.predict(Xs_test, verbose=0).flatten()
        metrics = compute_metrics(y_test.values[:len(y_pred_lstm)], y_pred_lstm)
        mlflow.log_metrics(metrics)
        mlflow.log_param("lookback", lookback)
        results["LSTM"] = {
            "y_true": y_test.values[:len(y_pred_lstm)],
            "y_pred": y_pred_lstm,
            "metrics": metrics,
        }
        all_metrics["lstm"] = metrics
    print(f"  LSTM: {metrics}")

    # --- Ensemble stacking ---
    print("Training Ensemble (stacking)...")
    with mlflow.start_run(run_name="ensemble"):
        bundle_ens = train_ensemble(X_train, y_train, bundle_lin, bundle_xgb, bundle_arima)
        y_pred_ens = predict_ensemble_from_bundle(bundle_ens, X_test)
        metrics = compute_metrics(y_test.values, y_pred_ens)
        mlflow.log_metrics(metrics)
        results["Ensemble"] = {"y_true": y_test.values, "y_pred": y_pred_ens, "metrics": metrics}
        all_metrics["ensemble"] = metrics
    print(f"  Ensemble: {metrics}")

    # --- Naive baseline (persistence: predict = last month's actual) ---
    naive_metrics = naive_baseline_metrics(y_test.values)
    all_metrics["naive"] = naive_metrics
    print(f"  Naive baseline: {naive_metrics}")

    # --- Bootstrap 90% confidence intervals for best models ---
    ci_data = {}
    for key, res in [("linear", results["Linear"]),
                     ("xgboost", results["XGBoost"]),
                     ("ensemble", results["Ensemble"])]:
        ci = bootstrap_confidence_interval(res["y_true"], res["y_pred"])
        ci_data[key] = ci
        print(f"  {key} CI ±{ci['ci_half_width']} (90%), residual std={ci['residual_std']}")
    (MODELS_DIR / "confidence_intervals.json").write_text(json.dumps(ci_data, indent=2))
    print("Bootstrap CIs saved.")

    # Persist all metrics
    _save_metrics(all_metrics)

    # Pick best model by RMSE (exclude ARIMA — it cannot do feature-based single predictions)
    candidates = {k: v for k, v in results.items() if k != "ARIMA"}
    best_name = min(candidates, key=lambda k: candidates[k]["metrics"]["RMSE"])
    (MODELS_DIR / "best_model.txt").write_text(best_name.lower())
    print(f"\nBest model: {best_name}")

    # Optional walk-forward validation
    if run_walk_forward and df_full is not None:
        print("\nRunning walk-forward validation (this takes a while)...")
        walk_forward_validation(df_full)

    return compare_models({k: v for k, v in results.items()})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-forward", action="store_true",
                        help="Also run walk-forward validation after training")
    args = parser.parse_args()

    from src.features import load_processed, split, get_xy
    df = load_processed()
    train, test = split(df)
    X_train, y_train = get_xy(train)
    X_test,  y_test  = get_xy(test)
    comparison = train_all(
        X_train, y_train, X_test, y_test,
        run_walk_forward=args.walk_forward,
        df_full=df,
    )
    print("\n=== Model Comparison ===")
    print(comparison.to_string())
