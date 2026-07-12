"""FastAPI service — inflation rate prediction + dashboard."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from api.schemas import PredictRequest, PredictResponse, ModelInfo, HistoryItem
from typing import List

MODELS_DIR    = Path(__file__).parent.parent / "models"
PROC_DIR      = Path(__file__).parent.parent / "data" / "processed"
DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"

FEATURE_LABELS = {
    "cpi_lag1": "CPI — 1 month ago", "cpi_lag3": "CPI — 3 months ago",
    "cpi_lag6": "CPI — 6 months ago", "cpi_lag12": "CPI — 1 year ago",
    "cpi_roll_mean3": "3-Month CPI Average", "cpi_roll_mean12": "12-Month CPI Average",
    "m2_yoy": "Money Supply Growth (YoY)", "fed_funds": "Fed Interest Rate",
    "fed_funds_yoy": "Interest Rate Change (YoY)", "unemployment": "Unemployment Rate",
    "oil_wti": "Oil Price (WTI)", "oil_wti_yoy": "Oil Price Change (YoY)",
    "ppi": "Producer Price Index", "ppi_yoy": "Producer Price Change (YoY)",
    "gdp_growth": "GDP Growth Rate", "m2_rate_interact": "Money × Rate Interaction",
    "post_covid": "Post-COVID Period", "rate_diff": "Rate Momentum",
    "m2": "Money Supply (M2)", "cpi_roll_std3": "CPI Volatility (3-mo)",
    "cpi_roll_std12": "CPI Volatility (12-mo)", "unemployment_yoy": "Unemployment Change (YoY)",
    "unemployment_mom": "Unemployment Change (MoM)", "m2_mom": "Money Supply Change (MoM)",
    "ppi_mom": "Producer Price Change (MoM)", "oil_wti_mom": "Oil Price Change (MoM)",
    "fed_funds_mom": "Interest Rate Change (MoM)", "gdp_growth_lag1": "GDP Growth (prev. quarter)",
}

_bundles: dict = {}
_feature_cols: list[str] = []


def _load_bundles():
    for name in ["linear", "xgboost", "lstm", "arima", "ensemble"]:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            _bundles[name] = joblib.load(path)
    cols_path = MODELS_DIR / "feature_columns.json"
    if cols_path.exists():
        _feature_cols.clear()
        _feature_cols.extend(json.loads(cols_path.read_text()))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_bundles()
    yield


app = FastAPI(title="Inflation API", version="1.1.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Dashboard SPA ─────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_dashboard():
    p = DASHBOARD_DIR / "index.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Dashboard not built.")
    return FileResponse(str(p))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _best_model_name() -> str:
    path = MODELS_DIR / "best_model.txt"
    if path.exists():
        return path.read_text().strip()
    return next((k for k in ["ensemble", "xgboost", "linear"] if k in _bundles), "linear")


def _run_model(model_name: str, X: pd.DataFrame) -> np.ndarray | None:
    if model_name == "linear" and "linear" in _bundles:
        b = _bundles["linear"]
        return b["model"].predict(b["scaler"].transform(X))
    if model_name == "xgboost" and "xgboost" in _bundles:
        return _bundles["xgboost"]["model"].predict(X)
    if model_name == "ensemble" and "ensemble" in _bundles:
        b = _bundles["ensemble"]
        pred_lin = b["linear_bundle"]["model"].predict(b["linear_bundle"]["scaler"].transform(X))
        pred_xgb = b["xgb_bundle"]["model"].predict(X)
        meta_X = np.column_stack([pred_lin, pred_xgb])
        return b["meta_model"].predict(meta_X)
    return None


def _build_feature_row(req: "PredictRequest") -> pd.DataFrame:
    """
    Build a single inference row for the "what-if" predictor.

    The six fields on PredictRequest are the only ones a user actually controls.
    Every other engineered feature (MoM/YoY changes, CPI lags/rolling stats,
    interaction terms, real_rate, infl_mom3/6, yield_curve_proxy...) is derived
    from the most recent real history in data/processed/features.csv, with the
    user's overrides substituted in place of "this month's" values. Previously
    these were hardcoded to 0.0 / stale constants, which silently zeroed out
    some of the model's most important features (e.g. ppi_yoy, real_rate) —
    sliders barely moved the prediction. Using real trailing history keeps the
    engineered features consistent with how they were computed during training.
    """
    hist_path = PROC_DIR / "features.csv"
    if not hist_path.exists():
        raise HTTPException(503, "Historical data not found — run the pipeline first.")
    hist = pd.read_csv(hist_path, index_col=0, parse_dates=True)

    latest = hist.iloc[-1]                              # most recent known month (t-1)
    prev   = hist.iloc[-2] if len(hist) >= 2 else latest # t-2, for MoM deltas
    yr_ago = hist.iloc[-13] if len(hist) >= 13 else latest  # t-13 ~ 12mo before latest

    def pct_chg(new: float, old: float) -> float:
        return ((new - old) / old * 100) if old else 0.0

    fed_funds_prev = float(prev["fed_funds"])
    m2_prev        = float(prev["m2"])

    row = {
        "m2": req.m2, "unemployment": req.unemployment, "fed_funds": req.fed_funds,
        "oil_wti": req.oil_wti, "ppi": req.ppi, "gdp_growth": req.gdp_growth,
        "gdp_growth_lag1": float(latest["gdp_growth"]),
        "m2_mom": pct_chg(req.m2, m2_prev),
        "m2_yoy": pct_chg(req.m2, float(yr_ago["m2"])),
        "unemployment_mom": pct_chg(req.unemployment, float(prev["unemployment"])),
        "unemployment_yoy": pct_chg(req.unemployment, float(yr_ago["unemployment"])),
        "fed_funds_mom": pct_chg(req.fed_funds, fed_funds_prev),
        "fed_funds_yoy": pct_chg(req.fed_funds, float(yr_ago["fed_funds"])),
        "oil_wti_mom": pct_chg(req.oil_wti, float(prev["oil_wti"])),
        "oil_wti_yoy": pct_chg(req.oil_wti, float(yr_ago["oil_wti"])),
        "ppi_mom": pct_chg(req.ppi, float(prev["ppi"])),
        "ppi_yoy": pct_chg(req.ppi, float(yr_ago["ppi"])),
        "cpi_lag1":  float(latest["cpi"]),
        "cpi_lag3":  float(hist["cpi"].iloc[-3])  if len(hist) >= 3  else float(latest["cpi"]),
        "cpi_lag6":  float(hist["cpi"].iloc[-6])  if len(hist) >= 6  else float(latest["cpi"]),
        "cpi_lag12": float(hist["cpi"].iloc[-12]) if len(hist) >= 12 else float(latest["cpi"]),
        "cpi_roll_mean3":  float(hist["cpi"].tail(3).mean()),
        "cpi_roll_std3":   float(hist["cpi"].tail(3).std(ddof=1)) if len(hist) >= 3 else 0.0,
        "cpi_roll_mean12": float(hist["cpi"].tail(12).mean()),
        "cpi_roll_std12":  float(hist["cpi"].tail(12).std(ddof=1)) if len(hist) >= 12 else 0.0,
        "rate_diff": req.fed_funds - fed_funds_prev,
        "m2_rate_interact": pct_chg(req.m2, m2_prev) * (req.fed_funds - fed_funds_prev),
        "post_covid": 1,
        "real_rate": req.fed_funds - float(latest["inflation_rate"]),
        "yield_curve_proxy": req.fed_funds - float(
            pd.concat([hist["fed_funds"].tail(11), pd.Series([req.fed_funds])]).mean()
        ),
        "infl_mom3": float(latest["inflation_rate"]) - float(hist["inflation_rate"].iloc[-4]) if len(hist) >= 4 else 0.0,
        "infl_mom6": float(latest["inflation_rate"]) - float(hist["inflation_rate"].iloc[-7]) if len(hist) >= 7 else 0.0,
    }
    # Note: req.month is accepted for API stability but is no longer a model
    # feature — month-of-year dummies were dropped (see src/features.py); a
    # 12-month YoY target already cancels out most within-year seasonality.
    df = pd.DataFrame([row])
    if _feature_cols:
        df = df.reindex(columns=_feature_cols, fill_value=0.0)
    return df


# ── New endpoints ─────────────────────────────────────────────────────────────

@app.get("/chart-data")
def chart_data(
    model: str = Query("xgboost"),
    year_from: int = Query(2000),
    year_to: int = Query(2025),
):
    p = PROC_DIR / "features.csv"
    if not p.exists():
        raise HTTPException(503, "Run the pipeline first.")
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df = df.loc[str(year_from):str(year_to)]

    feat_cols = [c for c in _feature_cols if c in df.columns] if _feature_cols \
                else [c for c in df.columns if c not in {"inflation_rate","cpi","cpi_mom"}]
    X = df[feat_cols]
    preds = _run_model(model, X)

    return {
        "dates":     [str(d.date()) for d in df.index],
        "actual":    [round(float(v), 3) for v in df["inflation_rate"]],
        "predicted": [round(float(v), 3) for v in preds] if preds is not None else [],
        "model":     model,
    }


@app.get("/summary")
def summary(model: str = Query("xgboost")):
    p = PROC_DIR / "features.csv"
    if not p.exists():
        raise HTTPException(503, "Run the pipeline first.")
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    feat_cols = [c for c in _feature_cols if c in df.columns] if _feature_cols \
                else [c for c in df.columns if c not in {"inflation_rate","cpi","cpi_mom"}]
    preds = _run_model(model, df[feat_cols])

    latest_actual = float(df["inflation_rate"].iloc[-1])
    latest_pred   = float(preds[-1]) if preds is not None else None
    mae = float(np.mean(np.abs(df["inflation_rate"].values - preds))) if preds is not None else None
    trend = None
    if preds is not None and len(preds) >= 6:
        trend = round(float(np.mean(preds[-3:])) - float(np.mean(preds[-6:-3])), 3)

    return {
        "latest_actual": round(latest_actual, 3),
        "latest_date":   str(df.index[-1].date()),
        "latest_pred":   round(latest_pred, 3) if latest_pred is not None else None,
        "mae":           round(mae, 3) if mae is not None else None,
        "trend":         trend,
        "model":         model,
    }


@app.get("/metrics")
def metrics():
    """Return saved per-model metrics from training."""
    p = MODELS_DIR / "metrics.json"
    if not p.exists():
        raise HTTPException(503, "Train models first (metrics.json not found).")
    return json.loads(p.read_text())


@app.get("/shap")
def shap_values(top: int = Query(15)):
    """Return SHAP summary (mean abs per feature) and per-date SHAP values."""
    summary_path = MODELS_DIR / "shap_summary.json"
    values_path  = MODELS_DIR / "shap_values.npy"
    dates_path   = MODELS_DIR / "shap_dates.json"
    if not summary_path.exists():
        raise HTTPException(503, "SHAP data not found. Train models first.")

    summary_data = json.loads(summary_path.read_text())[:top]
    result = {"summary": summary_data}

    if values_path.exists() and dates_path.exists():
        shap_arr = np.load(str(values_path))
        dates    = json.loads(dates_path.read_text())
        feat_cols = _feature_cols or [f"f{i}" for i in range(shap_arr.shape[1])]
        top_feats = [item["feature"] for item in summary_data]
        top_idxs  = [feat_cols.index(f) for f in top_feats if f in feat_cols]
        result["time_series"] = {
            "dates":    dates,
            "features": top_feats,
            "values":   [shap_arr[:, i].tolist() for i in top_idxs],
        }

    return result


@app.get("/walk-forward")
def walk_forward():
    """Return walk-forward validation results."""
    p = MODELS_DIR / "walk_forward.json"
    if not p.exists():
        raise HTTPException(503, "Walk-forward data not found. Run train.py --walk-forward")
    data = json.loads(p.read_text())
    y = np.array(data["y_true"])
    yp = np.array(data["y_pred"])
    mae  = round(float(np.mean(np.abs(y - yp))), 4)
    rmse = round(float(np.sqrt(np.mean((y - yp)**2))), 4)
    nz   = y != 0
    mape = round(float(np.mean(np.abs((y[nz]-yp[nz])/y[nz])))*100, 2)
    return {**data, "metrics": {"MAE": mae, "RMSE": rmse, "MAPE": mape}}


@app.get("/ablation")
def ablation():
    """Return ablation study results."""
    p = MODELS_DIR / "ablation_results.json"
    if not p.exists():
        raise HTTPException(503, "Ablation results not found. Run python -m src.ablation")
    return json.loads(p.read_text())


@app.get("/feature-importance")
def feature_importance(top: int = Query(10)):
    if "xgboost" not in _bundles:
        raise HTTPException(503, "XGBoost model not loaded.")
    imps  = _bundles["xgboost"]["model"].feature_importances_
    cols  = _feature_cols[:len(imps)] if _feature_cols else [f"f{i}" for i in range(len(imps))]
    pairs = sorted(zip(cols, imps.tolist()), key=lambda x: x[1], reverse=True)[:top]
    return [{"feature": f, "label": FEATURE_LABELS.get(f, f.replace("_"," ").title()),
             "importance": round(v, 4)} for f, v in pairs]


@app.get("/indicators")
def indicators():
    p = PROC_DIR / "features.csv"
    if not p.exists():
        raise HTTPException(503, "Run pipeline first.")
    df = pd.read_csv(p, index_col=0, parse_dates=True)

    def sparkline(col):
        if col not in df.columns: return []
        return [round(float(v), 3) for v in df[col].tail(12)]

    def change(col, periods=12):
        if col not in df.columns or len(df) < periods + 1: return 0.0
        return round(float(df[col].iloc[-1]) - float(df[col].iloc[-1 - periods]), 3)

    def latest(col):
        if col not in df.columns: return 0.0
        return round(float(df[col].iloc[-1]), 3)

    return [
        {"name":"GDP Growth (QoQ)",     "key":"gdp_growth",   "value":latest("gdp_growth"),  "unit":"%", "change":change("gdp_growth"),  "sparkline":sparkline("gdp_growth")},
        {"name":"M2 Money Supply (YoY)","key":"m2_yoy",        "value":latest("m2_yoy"),       "unit":"%", "change":change("m2_yoy"),       "sparkline":sparkline("m2_yoy")},
        {"name":"Crude Oil Price (WTI)","key":"oil_wti",       "value":latest("oil_wti"),      "unit":"$", "change":change("oil_wti"),      "sparkline":sparkline("oil_wti")},
        {"name":"Unemployment Rate",    "key":"unemployment",  "value":latest("unemployment"), "unit":"%", "change":change("unemployment"), "sparkline":sparkline("unemployment")},
        {"name":"Fed Funds Rate",       "key":"fed_funds",     "value":latest("fed_funds"),    "unit":"%", "change":change("fed_funds"),    "sparkline":sparkline("fed_funds")},
        {"name":"PPI (YoY)",            "key":"ppi_yoy",       "value":latest("ppi_yoy"),      "unit":"%", "change":change("ppi_yoy"),      "sparkline":sparkline("ppi_yoy")},
    ]


# ── Original endpoints ────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model_name = _best_model_name() if req.model == "best" else req.model
    if model_name not in _bundles:
        raise HTTPException(503, f"Model '{model_name}' not loaded.")
    features = _build_feature_row(req)
    try:
        if model_name == "linear":
            b = _bundles["linear"]
            pred = float(b["model"].predict(b["scaler"].transform(features))[0])
        elif model_name == "xgboost":
            pred = float(_bundles["xgboost"]["model"].predict(features)[0])
        elif model_name == "ensemble":
            pred = float(_run_model("ensemble", features)[0])
        elif model_name == "arima":
            pred = float(_bundles["arima"]["model"].predict(n_periods=1)[0])
        elif model_name == "lstm":
            # LSTM needs historical context; we approximate with a single-step forward pass
            # using the most recent 24 rows from the processed dataset
            p = PROC_DIR / "features.csv"
            if not p.exists():
                raise HTTPException(503, "Processed data not found for LSTM context.")
            df_hist = pd.read_csv(p, index_col=0, parse_dates=True)
            feat_cols = [c for c in _feature_cols if c in df_hist.columns]
            if len(df_hist) < 24:
                raise HTTPException(422, "Insufficient historical data for LSTM.")
            b = _bundles["lstm"]
            lookback = b["lookback"]
            # Build scaler from saved params — scaler is in the bundle
            import tensorflow as tf
            keras_path = MODELS_DIR / "lstm_model.keras"
            if not keras_path.exists():
                raise HTTPException(503, "LSTM model file not found.")
            lstm_model = tf.keras.models.load_model(str(keras_path))
            X_hist = df_hist[feat_cols].tail(lookback).values
            X_scaled = b["scaler"].transform(X_hist)
            seq = X_scaled.reshape(1, lookback, -1)
            pred = float(lstm_model.predict(seq, verbose=0)[0, 0])
        else:
            raise HTTPException(400, f"Unknown model: {model_name}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

    # Conformal prediction intervals
    lower_90 = upper_90 = conf_level = None
    if model_name == "lstm":
        q = _bundles["lstm"].get("conformal_q90")
        if q is not None:
            lower_90   = round(pred - q, 4)
            upper_90   = round(pred + q, 4)
            conf_level = 0.90
    elif model_name in ("ensemble", "xgboost", "linear"):
        # Use empirical residual std from metrics as a proxy interval
        metrics_path = MODELS_DIR / "metrics.json"
        if metrics_path.exists():
            saved = json.loads(metrics_path.read_text())
            rmse = saved.get(model_name, {}).get("RMSE")
            if rmse:
                lower_90   = round(pred - 1.645 * rmse, 4)
                upper_90   = round(pred + 1.645 * rmse, 4)
                conf_level = 0.90

    return PredictResponse(
        inflation_rate=round(pred, 4),
        model_used=model_name,
        lower_bound_90=lower_90,
        upper_bound_90=upper_90,
        confidence_level=conf_level,
    )


@app.get("/models", response_model=list[ModelInfo])
def list_models():
    best = _best_model_name()
    return [ModelInfo(name=n, is_best=(n == best), path=str(MODELS_DIR / f"{n}.joblib"))
            for n in ["linear", "arima", "xgboost", "lstm", "ensemble"]
            if (MODELS_DIR / f"{n}.joblib").exists()]


@app.get("/history", response_model=list[HistoryItem])
def history():
    p = PROC_DIR / "features.csv"
    if not p.exists():
        raise HTTPException(503, "Processed data not found.")
    df = pd.read_csv(p, index_col=0, parse_dates=True).tail(24)
    model_name = _best_model_name()
    actuals    = df["inflation_rate"].tolist()
    feat_cols  = [c for c in _feature_cols if c in df.columns] if _feature_cols \
                 else [c for c in df.columns if c not in {"inflation_rate","cpi","cpi_mom"}]
    preds_arr  = _run_model(model_name, df[feat_cols])
    preds      = preds_arr.tolist() if preds_arr is not None else [None] * len(df)
    return [HistoryItem(date=str(df.index[i].date()),
                        actual=round(actuals[i], 4),
                        predicted=round(preds[i], 4) if preds[i] is not None else None,
                        model=model_name) for i in range(len(df))]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_models": list(_bundles.keys()),
        "best_model": _best_model_name(),
    }


@app.post("/predict/batch", response_model=List[PredictResponse])
def predict_batch(requests_list: List[PredictRequest]):
    """Batch prediction endpoint — accepts an array of feature payloads."""
    results = []
    for req in requests_list:
        model_name = _best_model_name() if req.model == "best" else req.model
        if model_name not in _bundles:
            raise HTTPException(503, f"Model '{model_name}' not loaded.")
        features = _build_feature_row(req)
        try:
            if model_name == "linear":
                b = _bundles["linear"]
                pred = float(b["model"].predict(b["scaler"].transform(features))[0])
            elif model_name == "xgboost":
                pred = float(_bundles["xgboost"]["model"].predict(features)[0])
            elif model_name == "ensemble":
                pred = float(_run_model("ensemble", features)[0])
            elif model_name == "arima":
                pred = float(_bundles["arima"]["model"].predict(n_periods=1)[0])
            else:
                raise HTTPException(400, f"Batch not supported for model: {model_name}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, str(e))
        results.append(PredictResponse(inflation_rate=round(pred, 4), model_used=model_name))
    return results


@app.get("/features")
def features_endpoint(top: int = Query(10)):
    """Return SHAP feature importance for the most recent prediction (live, from test-set summary)."""
    summary_path = MODELS_DIR / "shap_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())[:top]
        return {
            "source": "shap",
            "features": [
                {
                    "feature": item["feature"],
                    "label": FEATURE_LABELS.get(item["feature"], item["feature"].replace("_", " ").title()),
                    "mean_abs_shap": round(item["shap"], 4),
                }
                for item in data
            ],
        }
    # Fallback to XGBoost gain importance
    if "xgboost" not in _bundles or not _feature_cols:
        raise HTTPException(503, "Feature importance not available. Train models first.")
    imps = _bundles["xgboost"]["model"].feature_importances_
    cols = _feature_cols[:len(imps)]
    pairs = sorted(zip(cols, imps.tolist()), key=lambda x: x[1], reverse=True)[:top]
    return {
        "source": "gain",
        "features": [
            {
                "feature": f,
                "label": FEATURE_LABELS.get(f, f.replace("_", " ").title()),
                "importance": round(v, 4),
            }
            for f, v in pairs
        ],
    }


@app.get("/dashboard-data")
def dashboard_data(model: str = Query("linear")):
    """Single endpoint that returns everything the dashboard needs."""
    p = PROC_DIR / "features.csv"
    if not p.exists():
        raise HTTPException(503, "Run the pipeline first.")

    df = pd.read_csv(p, index_col=0, parse_dates=True)
    feat_cols = [c for c in _feature_cols if c in df.columns] if _feature_cols \
                else [c for c in df.columns if c not in {"inflation_rate","cpi","cpi_mom"}]

    preds_all   = _run_model(model, df[feat_cols])

    cur  = float(df["inflation_rate"].iloc[-1])
    prev = float(df["inflation_rate"].iloc[-2])
    ten_yr_avg   = float(df["inflation_rate"].last("120ME").mean())
    ten_yr_start = str(df.index[-120].year) if len(df) >= 120 else "2015"
    ten_yr_end   = str(df.index[-1].year)

    fc3, fc12, ci_lower_f, ci_upper_f, future_dates = None, None, [], [], []
    if "arima" in _bundles:
        try:
            fc, ci = _bundles["arima"]["model"].predict(n_periods=12, return_conf_int=True)
            fc3  = round(float(np.mean(fc[:3])), 2)
            fc12 = round(float(fc[-1]), 2)
            last_date    = df.index[-1]
            future_dates = [str((last_date + pd.DateOffset(months=i+1)).date()) for i in range(12)]
            ci_lower_f   = [round(float(v), 2) for v in ci[:,0]]
            ci_upper_f   = [round(float(v), 2) for v in ci[:,1]]
        except Exception:
            pass

    chart_df     = df.loc["2020-01-01":]
    preds_chart  = _run_model(model, chart_df[feat_cols])
    res_std = 0.5
    if preds_all is not None:
        residuals = df["inflation_rate"].values - preds_all
        res_std   = float(np.std(residuals[-36:]))

    test_df    = df.loc["2019-01-01":]
    preds_test = _run_model(model, test_df[feat_cols])
    mae, rmse, mape_v, dir_acc = None, None, None, None
    if preds_test is not None:
        y      = test_df["inflation_rate"].values
        mae    = round(float(np.mean(np.abs(y - preds_test))), 2)
        rmse   = round(float(np.sqrt(np.mean((y - preds_test)**2))), 2)
        nz     = y != 0
        mape_v = round(float(np.mean(np.abs((y[nz]-preds_test[nz])/y[nz])))*100, 1)
        dir_acc = round(float(np.mean(np.sign(np.diff(y))==np.sign(np.diff(preds_test))))*100, 1)

    drivers = []
    if "xgboost" in _bundles and feat_cols:
        imps = _bundles["xgboost"]["model"].feature_importances_
        cols = feat_cols[:len(imps)]
        top  = sorted(zip(cols, imps.tolist()), key=lambda x: x[1], reverse=True)[:6]
        for rank, (feat, imp) in enumerate(top):
            impact = "High" if rank < 2 else "Medium" if rank < 4 else "Low"
            chg    = 0.0
            if feat in df.columns and len(df) >= 2:
                last_v, prev_v = float(df[feat].iloc[-1]), float(df[feat].iloc[-2])
                chg = round(last_v - prev_v, 2) if abs(prev_v) > 0.01 else 0.0
            drivers.append({
                "name": FEATURE_LABELS.get(feat, feat.replace("_"," ").title()),
                "impact": impact, "change": chg, "importance": round(imp*100, 1),
            })

    recent = []
    if preds_all is not None:
        for i in range(-6, 0):
            pred = round(float(preds_all[i]), 1)
            recent.append({
                "period":    df.index[i].strftime("%b %Y"),
                "predicted": pred,
                "lower":     round(pred - 1.96*res_std, 1),
                "upper":     round(pred + 1.96*res_std, 1),
                "actual":    round(float(df["inflation_rate"].iloc[i]), 1),
            })

    # Load real per-model metrics if available
    metrics_path = MODELS_DIR / "metrics.json"
    saved_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    return {
        "kpis": {
            "current_inflation":  round(cur, 1),
            "current_date":       df.index[-1].strftime("%b %Y"),
            "vs_prev_month":      round(cur - prev, 2),
            "forecast_3m":        fc3,
            "forecast_3m_date":   future_dates[2] if future_dates else None,
            "forecast_12m":       fc12,
            "forecast_12m_date":  future_dates[-1] if future_dates else None,
            "historical_avg_10y": round(ten_yr_avg, 1),
            "hist_range":         f"{ten_yr_start}–{ten_yr_end}",
        },
        "chart": {
            "dates":           [str(d.date()) for d in chart_df.index],
            "actual":          [round(float(v),2) for v in chart_df["inflation_rate"]],
            "predicted":       [round(float(v),2) for v in preds_chart] if preds_chart is not None else [],
            "ci_lower":        [round(float(v)-1.96*res_std,2) for v in preds_chart] if preds_chart is not None else [],
            "ci_upper":        [round(float(v)+1.96*res_std,2) for v in preds_chart] if preds_chart is not None else [],
            "future_dates":    future_dates,
            "future_predicted":[round(float(v),2) for v in
                                 _bundles["arima"]["model"].predict(n_periods=12)] if "arima" in _bundles else [],
            "future_ci_lower": ci_lower_f,
            "future_ci_upper": ci_upper_f,
        },
        "drivers":             drivers,
        "recent_predictions":  recent,
        "metrics": {
            "mae": mae, "rmse": rmse, "mape": mape_v, "dir_accuracy": dir_acc,
            "model": model,
        },
        "all_model_metrics":   saved_metrics,
        "feature_importance": [
            {"name": FEATURE_LABELS.get(f, f.replace("_"," ").title()),
             "importance": round(v*100,1)}
            for f,v in sorted(
                zip(feat_cols[:len(_bundles["xgboost"]["model"].feature_importances_)],
                    _bundles["xgboost"]["model"].feature_importances_.tolist()),
                key=lambda x:x[1], reverse=True)[:9]
        ] if "xgboost" in _bundles and feat_cols else [],
    }
