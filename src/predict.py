"""Inference utilities — load saved models and generate predictions."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

_cache: dict = {}


def _load(name: str):
    if name not in _cache:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}. Train the models first.")
        _cache[name] = joblib.load(path)
    return _cache[name]


def list_models() -> list[str]:
    return [p.stem for p in MODELS_DIR.glob("*.joblib")]


def predict_linear(features: pd.DataFrame) -> np.ndarray:
    bundle = _load("linear")
    X_scaled = bundle["scaler"].transform(features)
    return bundle["model"].predict(X_scaled)


def predict_xgboost(features: pd.DataFrame) -> np.ndarray:
    bundle = _load("xgboost")
    return bundle["model"].predict(features)


def predict_lstm(features: pd.DataFrame, lookback: int = 24) -> np.ndarray:
    import tensorflow as tf
    bundle = _load("lstm")
    scaler = bundle["scaler"]
    model  = bundle["model"]
    X_scaled = scaler.transform(features)
    sequences = np.array([X_scaled[i - lookback:i] for i in range(lookback, len(X_scaled) + 1)])
    if len(sequences) == 0:
        raise ValueError(f"Need at least {lookback} rows for LSTM inference.")
    preds = model.predict(sequences, verbose=0).flatten()
    # Pad the first `lookback` positions with NaN to align with input length
    padding = np.full(lookback - 1, np.nan) if len(features) > 1 else np.array([])
    return np.concatenate([padding, preds]) if len(padding) > 0 else preds


def predict_arima(steps: int = 12) -> np.ndarray:
    bundle = _load("arima")
    model  = bundle["model"]
    return model.predict(n_periods=steps)


def predict_best(features: pd.DataFrame) -> np.ndarray:
    """Use the model flagged as best during training."""
    meta_path = MODELS_DIR / "best_model.txt"
    if not meta_path.exists():
        raise FileNotFoundError("best_model.txt not found. Run train.py first.")
    best = meta_path.read_text().strip()
    dispatch = {
        "linear":  predict_linear,
        "xgboost": predict_xgboost,
        "lstm":    predict_lstm,
    }
    if best not in dispatch:
        raise ValueError(f"Unknown best model: {best}")
    return dispatch[best](features)
