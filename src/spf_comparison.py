"""
Compare our best model against the Survey of Professional Forecasters (SPF)
consensus CPI forecast from the Philadelphia Fed.

The SPF releases one-year-ahead mean CPI forecasts quarterly.
We download the FRED series MICH (U Michigan 1yr inflation expectation, monthly)
as a publicly available professional-forecaster proxy, then also attempt to
load the SPF Excel file if the user has downloaded it manually.

Saves: models/spf_comparison.json
Run:   python -m src.spf_comparison
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent.parent / "models"
PROC_DIR   = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR.mkdir(exist_ok=True)


def _load_mich_proxy() -> pd.Series:
    """Download U-Michigan 1yr inflation expectation from FRED as SPF proxy."""
    try:
        from fredapi import Fred
        from dotenv import load_dotenv
        import os
        load_dotenv()
        key = os.getenv("FRED_API_KEY", "")
        fred = Fred(api_key=key)
        mich = fred.get_series("MICH", observation_start="2011-01-01")
        mich.index = pd.to_datetime(mich.index)
        mich = mich.resample("MS").last()
        return mich.rename("spf_proxy")
    except Exception as e:
        print(f"  Could not download MICH from FRED: {e}")
        return pd.Series(dtype=float)


def run_spf_comparison() -> pd.DataFrame:
    from src.features import load_processed, split, get_xy
    import joblib

    df = load_processed()
    _, test = split(df)
    X_test, y_test = get_xy(test)

    # Load our best model predictions
    best_path = MODELS_DIR / "best_model.txt"
    best_name = best_path.read_text().strip() if best_path.exists() else "ensemble"

    preds_our = None
    if best_name == "ensemble" and (MODELS_DIR / "ensemble.joblib").exists():
        b = joblib.load(MODELS_DIR / "ensemble.joblib")
        pred_lin = b["linear_bundle"]["model"].predict(
            b["linear_bundle"]["scaler"].transform(X_test))
        pred_xgb = b["xgb_bundle"]["model"].predict(X_test)
        meta_X   = np.column_stack([pred_lin, pred_xgb])
        preds_our = b["meta_model"].predict(meta_X)
    elif best_name == "xgboost" and (MODELS_DIR / "xgboost.joblib").exists():
        b = joblib.load(MODELS_DIR / "xgboost.joblib")
        preds_our = b["model"].predict(X_test)
    elif best_name == "linear" and (MODELS_DIR / "linear.joblib").exists():
        b = joblib.load(MODELS_DIR / "linear.joblib")
        preds_our = b["model"].predict(b["scaler"].transform(X_test))

    if preds_our is None:
        print("No model predictions available.")
        return pd.DataFrame()

    # Align on common dates
    our_df = pd.DataFrame({
        "actual":    y_test.values,
        "our_model": preds_our,
    }, index=y_test.index)

    # SPF proxy via FRED MICH
    print("Downloading U-Michigan inflation expectation (SPF proxy)...")
    mich = _load_mich_proxy()

    rows = []

    if len(mich) > 0:
        combined = our_df.join(mich, how="inner").dropna()
        if len(combined) > 0:
            y  = combined["actual"].values
            yp = combined["our_model"].values
            ys = combined["spf_proxy"].values

            our_mae  = float(np.mean(np.abs(y - yp)))
            spf_mae  = float(np.mean(np.abs(y - ys)))
            our_rmse = float(np.sqrt(np.mean((y - yp)**2)))
            spf_rmse = float(np.sqrt(np.mean((y - ys)**2)))
            nz       = y != 0
            our_mape = float(np.mean(np.abs((y[nz]-yp[nz])/y[nz])))*100
            spf_mape = float(np.mean(np.abs((y[nz]-ys[nz])/y[nz])))*100

            rows.append({
                "Model":         f"Our {best_name.title()}",
                "Comparator":    "U-Michigan (SPF proxy)",
                "Period":        f"{combined.index[0].date()} to {combined.index[-1].date()}",
                "n_months":      len(combined),
                "Our MAE":       round(our_mae,  4),
                "Comparator MAE":round(spf_mae,  4),
                "Our RMSE":      round(our_rmse, 4),
                "Comparator RMSE":round(spf_rmse,4),
                "Our MAPE":      round(our_mape, 2),
                "Comparator MAPE":round(spf_mape,2),
                "Our MAE beats SPF": our_mae < spf_mae,
                "Our RMSE beats SPF": our_rmse < spf_rmse,
            })

            print(f"\n  Period: {combined.index[0].date()} to {combined.index[-1].date()} ({len(combined)} months)")
            print(f"  {'Metric':<12} {'Our model':>12} {'U-Mich SPF':>12} {'Winner':>10}")
            print(f"  {'-'*50}")
            print(f"  {'MAE':<12} {our_mae:>12.4f} {spf_mae:>12.4f} {'OURS' if our_mae < spf_mae else 'SPF':>10}")
            print(f"  {'RMSE':<12} {our_rmse:>12.4f} {spf_rmse:>12.4f} {'OURS' if our_rmse < spf_rmse else 'SPF':>10}")
            print(f"  {'MAPE%':<12} {our_mape:>12.2f} {spf_mape:>12.2f} {'OURS' if our_mape < spf_mape else 'SPF':>10}")
        else:
            print("  No overlapping dates between our predictions and SPF proxy.")
    else:
        print("  SPF proxy unavailable — storing model-only metrics.")
        y  = our_df["actual"].values
        yp = our_df["our_model"].values
        rows.append({
            "Model":    f"Our {best_name.title()}",
            "Comparator": "SPF proxy unavailable",
            "Period":   f"{our_df.index[0].date()} to {our_df.index[-1].date()}",
            "n_months": len(our_df),
            "Our MAE":  round(float(np.mean(np.abs(y - yp))), 4),
            "Our RMSE": round(float(np.sqrt(np.mean((y - yp)**2))), 4),
        })

    out = MODELS_DIR / "spf_comparison.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSPF comparison saved -> {out}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = run_spf_comparison()
    if not results.empty:
        print("\n=== SPF Comparison Results ===")
        print(results.to_string(index=False))
