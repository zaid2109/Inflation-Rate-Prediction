"""HMM regime detection — identifies high vs. low inflation regimes.

Fits a 2-state Gaussian HMM on the CPI inflation-rate series.
State 0 = low-inflation regime, State 1 = high-inflation regime
(states are re-labelled after fitting based on mean emission values).

Saves:
  models/regime_labels.json  — {date: 0|1} mapping for every month
  models/regime_stats.json   — per-state mean / std
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent.parent / "models"


def fit_regime(y: pd.Series, n_states: int = 2) -> dict:
    """Fit HMM and return per-month regime labels."""
    from hmmlearn.hmm import GaussianHMM

    series = y.dropna().values.reshape(-1, 1)
    dates  = [str(d.date()) for d in y.dropna().index]

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(series)
    raw_states = model.predict(series)

    # Re-label: state with lower mean emission = 0 (low-inflation)
    means = model.means_.flatten()
    if means[0] > means[1]:
        raw_states = 1 - raw_states

    labels = {d: int(s) for d, s in zip(dates, raw_states)}

    # Per-state statistics
    stats = {}
    for s in range(n_states):
        vals = y.dropna().values[raw_states == s]
        stats[str(s)] = {
            "label": "Low Inflation" if s == 0 else "High Inflation",
            "mean":  round(float(np.mean(vals)), 3),
            "std":   round(float(np.std(vals)), 3),
            "count": int(len(vals)),
        }

    with open(MODELS_DIR / "regime_labels.json", "w") as f:
        json.dump(labels, f)
    with open(MODELS_DIR / "regime_stats.json", "w") as f:
        json.dump(stats, f)

    print(f"Regime detection complete -> {MODELS_DIR / 'regime_labels.json'}")
    for s, info in stats.items():
        print(f"  State {s} ({info['label']}): "
              f"mean={info['mean']:.2f}%  std={info['std']:.2f}  n={info['count']}")

    # Identify contiguous regime periods for shaded chart bands
    periods = _contiguous_periods(dates, raw_states)
    with open(MODELS_DIR / "regime_periods.json", "w") as f:
        json.dump(periods, f)

    return {"labels": labels, "stats": stats, "periods": periods}


def _contiguous_periods(dates: list[str], states: np.ndarray) -> list[dict]:
    """Convert label array into list of {start, end, regime} dicts."""
    periods = []
    if len(dates) == 0:
        return periods
    cur_state = states[0]
    cur_start = dates[0]
    for i in range(1, len(dates)):
        if states[i] != cur_state:
            periods.append({"start": cur_start, "end": dates[i - 1],
                            "regime": int(cur_state)})
            cur_state = states[i]
            cur_start = dates[i]
    periods.append({"start": cur_start, "end": dates[-1], "regime": int(cur_state)})
    return periods


if __name__ == "__main__":
    from src.features import load_processed
    df = load_processed()
    result = fit_regime(df["inflation_rate"])
    high_periods = [p for p in result["periods"] if p["regime"] == 1]
    print(f"\nHigh-inflation periods detected: {len(high_periods)}")
    for p in high_periods:
        print(f"  {p['start']}  ->  {p['end']}")
