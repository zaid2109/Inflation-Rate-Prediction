"""SHAP explainability for the XGBoost model.

Computes SHAP values on the full feature matrix, saves:
  - models/shap_summary.json  — top-N features by mean |SHAP|
  - models/shap_values.npy    — full SHAP matrix (rows = samples, cols = features)
  - models/shap_dates.json    — ISO date strings aligned with shap_values rows
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent.parent / "models"


def compute_shap(X: pd.DataFrame, top_n: int = 15) -> dict:
    """Compute SHAP values using the saved XGBoost model."""
    bundle = joblib.load(MODELS_DIR / "xgboost.joblib")
    model  = bundle["model"]

    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(X)          # shape (n_samples, n_features)

    mean_abs    = np.mean(np.abs(shap_vals), axis=0) # shape (n_features,)
    feat_names  = list(X.columns)

    ranked = sorted(zip(feat_names, mean_abs.tolist()),
                    key=lambda x: x[1], reverse=True)[:top_n]

    summary = [{"feature": f, "shap": round(v, 6)} for f, v in ranked]

    # Persist
    with open(MODELS_DIR / "shap_summary.json", "w") as fh:
        json.dump(summary, fh)

    np.save(MODELS_DIR / "shap_values.npy", shap_vals.astype(np.float32))

    dates = [str(d.date()) for d in X.index]
    with open(MODELS_DIR / "shap_dates.json", "w") as fh:
        json.dump(dates, fh)

    print(f"SHAP values saved -> {MODELS_DIR / 'shap_summary.json'}")
    print(f"\nTop {top_n} features by mean |SHAP|:")
    for item in summary:
        bar = "|" * max(1, int(item["shap"] / ranked[0][1] * 20))
        print(f"  {item['feature']:<28} {bar}  {item['shap']:.4f}")

    return {"summary": summary, "shap_values": shap_vals,
            "feature_names": feat_names, "dates": dates}


if __name__ == "__main__":
    from src.features import load_processed, get_xy
    df = load_processed()
    X, _ = get_xy(df)
    compute_shap(X)
