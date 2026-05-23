"""Metrics and model comparison utilities."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
from scipy import stats

FIGURES_DIR = Path(__file__).parent.parent / "figures"
MODELS_DIR  = Path(__file__).parent.parent / "models"
FIGURES_DIR.mkdir(exist_ok=True)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    actual_dir = np.diff(y_true)
    pred_dir   = np.diff(y_pred)
    return np.mean(np.sign(actual_dir) == np.sign(pred_dir)) * 100


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":      round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE":     round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAPE":     round(mape(y_true, y_pred), 4),
        "R2":       round(r2_score(y_true, y_pred), 4),
        "Dir_Acc":  round(directional_accuracy(y_true, y_pred), 2),
    }


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    results: {model_name: {"y_true": ..., "y_pred": ...}}
    Returns a DataFrame with one row per model.
    """
    rows = []
    for name, res in results.items():
        metrics = compute_metrics(
            np.array(res["y_true"]),
            np.array(res["y_pred"]),
        )
        metrics["Model"] = name
        rows.append(metrics)
    df = pd.DataFrame(rows).set_index("Model")
    return df[["MAE", "RMSE", "MAPE", "R2", "Dir_Acc"]]


def plot_predictions(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    save: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, y_true, label="Actual", color="black", linewidth=2)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (name, y_pred), color in zip(predictions.items(), colors):
        ax.plot(dates, y_pred, label=name, linestyle="--", color=color, linewidth=1.5)
    ax.set_title("Inflation Rate: Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Inflation Rate (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "predictions.png", dpi=150)
    plt.show()


def diebold_mariano(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    h: int = 1,
) -> dict:
    """
    Diebold-Mariano test (Harvey, Leybourne & Newbold 1997 small-sample correction).
    H0: the two models have equal predictive accuracy.
    Returns stat, p-value, and conclusion.
    h = forecast horizon (1 for one-step-ahead).
    """
    e_a = y_true - y_pred_a
    e_b = y_true - y_pred_b
    d   = e_a ** 2 - e_b ** 2          # loss differential (MSE-based)
    n   = len(d)

    d_bar = np.mean(d)
    # Newey-West variance with h-1 lags
    gamma0 = np.var(d, ddof=1)
    gamma  = sum(
        (1 - k / h) * np.cov(d[k:], d[:-k], ddof=1)[0, 1]
        for k in range(1, h)
    ) if h > 1 else 0.0
    var_d  = (gamma0 + 2 * gamma) / n

    # HLN correction factor
    hlncf  = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_stat = hlncf * d_bar / np.sqrt(var_d) if var_d > 0 else 0.0

    p_value = 2 * stats.t.sf(abs(dm_stat), df=n - 1)

    if p_value < 0.05:
        better = "A" if d_bar < 0 else "B"
        conclusion = f"Model {better} is significantly better (p={p_value:.4f})"
    else:
        conclusion = f"No significant difference (p={p_value:.4f})"

    return {
        "dm_stat":   round(float(dm_stat), 4),
        "p_value":   round(float(p_value), 4),
        "n":         int(n),
        "conclusion": conclusion,
    }


def run_diebold_mariano_suite(results: dict, baseline: str = "arima") -> pd.DataFrame:
    """
    Run DM test comparing each model against a baseline.
    results: {model_name: {"y_true": ..., "y_pred": ...}}
    """
    if baseline not in results:
        raise ValueError(f"Baseline '{baseline}' not in results dict.")
    y_true   = np.array(results[baseline]["y_true"])
    y_base   = np.array(results[baseline]["y_pred"])
    rows = []
    for name, res in results.items():
        if name == baseline:
            continue
        y_pred = np.array(res["y_pred"])
        n = min(len(y_true), len(y_pred))
        dm = diebold_mariano(y_true[:n], y_base[:n], y_pred[:n])
        rows.append({
            "Model":      name,
            "vs":         baseline,
            "DM Stat":    dm["dm_stat"],
            "p-value":    dm["p_value"],
            "n":          dm["n"],
            "Conclusion": dm["conclusion"],
        })
    df = pd.DataFrame(rows)

    out = MODELS_DIR / "diebold_mariano.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"Diebold-Mariano results saved -> {out}")
    return df


def plot_metrics_heatmap(metrics_df: pd.DataFrame, save: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    norm = metrics_df.copy()
    for col in ["MAE", "RMSE", "MAPE"]:
        norm[col] = 1 - (norm[col] - norm[col].min()) / (norm[col].max() - norm[col].min() + 1e-9)
    for col in ["R2", "Dir_Acc"]:
        norm[col] = (norm[col] - norm[col].min()) / (norm[col].max() - norm[col].min() + 1e-9)
    sns.heatmap(norm, annot=metrics_df.round(3), fmt="", cmap="RdYlGn", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Relative performance"})
    ax.set_title("Model Comparison (green = better)")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "metrics_heatmap.png", dpi=150)
    plt.show()
