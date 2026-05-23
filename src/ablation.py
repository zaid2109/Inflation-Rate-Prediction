"""
Ablation study: remove one feature group at a time, train XGBoost, measure RMSE.

Run:
    python -m src.ablation
"""

import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def _quick_xgb_rmse(X_train, y_train, X_test, y_test) -> float:
    mdl = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, verbosity=0,
    )
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    return float(np.sqrt(np.mean((y_test.values - preds) ** 2)))


def run_ablation(df: pd.DataFrame, target: str = "inflation_rate") -> pd.DataFrame:
    from src.features import split, get_xy

    train, test = split(df)
    X_train_full, y_train = get_xy(train, target)
    X_test_full,  y_test  = get_xy(test,  target)

    baseline_rmse = _quick_xgb_rmse(X_train_full, y_train, X_test_full, y_test)
    print(f"Baseline RMSE (full feature set): {baseline_rmse:.4f}")

    # Define feature groups to ablate
    feature_groups = {
        "No lag features":       [c for c in X_train_full.columns if "cpi_lag" in c],
        "No alternative data":   [c for c in X_train_full.columns if c in
                                   {"oil_wti", "oil_wti_mom", "oil_wti_yoy",
                                    "ppi", "ppi_mom", "ppi_yoy"}],
        "No interaction terms":  [c for c in X_train_full.columns if
                                   c in {"rate_diff", "m2_rate_interact"}],
        "No regime feature":     [c for c in X_train_full.columns if "post_covid" in c],
        "No rolling statistics": [c for c in X_train_full.columns if "roll" in c],
        "Univariate only":       [c for c in X_train_full.columns if c not in
                                   {"cpi_lag1", "cpi_lag3", "cpi_lag6", "cpi_lag12",
                                    "cpi_roll_mean3", "cpi_roll_std3",
                                    "cpi_roll_mean12", "cpi_roll_std12",
                                    "post_covid"} and not c.startswith("month_")],
    }

    rows = [{"Experiment": "Full model (baseline)", "RMSE": round(baseline_rmse, 4),
             "Delta vs Baseline": 0.0, "Delta %": "—", "Conclusion": "All features"}]

    for name, cols_to_drop in feature_groups.items():
        cols_to_drop = [c for c in cols_to_drop if c in X_train_full.columns]
        if not cols_to_drop:
            print(f"  Skipping '{name}' — no matching columns.")
            continue
        X_tr = X_train_full.drop(columns=cols_to_drop)
        X_te = X_test_full.drop(columns=cols_to_drop)
        rmse = _quick_xgb_rmse(X_tr, y_train, X_te, y_test)
        delta = rmse - baseline_rmse
        pct = delta / baseline_rmse * 100
        rows.append({
            "Experiment": name,
            "RMSE": round(rmse, 4),
            "Delta vs Baseline": round(delta, 4),
            "Delta %": f"+{pct:.0f}%" if pct >= 0 else f"{pct:.0f}%",
            "Conclusion": f"Dropped {len(cols_to_drop)} features",
        })
        print(f"  {name}: RMSE={rmse:.4f}  d={delta:+.4f}  ({pct:+.1f}%)")

    results_df = pd.DataFrame(rows)

    # Save
    out_path = MODELS_DIR / "ablation_results.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nAblation results saved -> {out_path}")
    return results_df


if __name__ == "__main__":
    from src.features import load_processed
    df = load_processed()
    results = run_ablation(df)
    print("\n=== Ablation Study Results ===")
    print(results.to_string(index=False))
