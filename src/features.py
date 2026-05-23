"""Feature engineering pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END  = "2018-12-31"
TEST_START = "2019-01-01"


def build_feature_matrix(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    df = df.resample("MS").last()
    df = df.loc["1990-01-01":"2025-12-31"]

    # GDP is quarterly; forward-fill to monthly
    df["gdp_growth"] = df["gdp_growth"].ffill()

    # Year-over-year inflation rate (target)
    df["inflation_rate"] = df["cpi"].pct_change(12, fill_method=None) * 100
    df["cpi_mom"] = df["cpi"].pct_change(1, fill_method=None) * 100

    # Lag features for CPI
    for lag in [1, 3, 6, 12]:
        df[f"cpi_lag{lag}"] = df["cpi"].shift(lag)

    # Rolling statistics
    for window in [3, 12]:
        df[f"cpi_roll_mean{window}"] = df["cpi"].shift(1).rolling(window).mean()
        df[f"cpi_roll_std{window}"]  = df["cpi"].shift(1).rolling(window).std()

    # MoM and YoY changes for all macro indicators
    for col in ["m2", "unemployment", "fed_funds", "oil_wti", "ppi"]:
        df[f"{col}_mom"] = df[col].pct_change(1, fill_method=None) * 100
        df[f"{col}_yoy"] = df[col].pct_change(12, fill_method=None) * 100

    # GDP is already a growth rate — just lag it
    df["gdp_growth_lag1"] = df["gdp_growth"].shift(1)

    # Interaction: M2 growth × interest rate differential (fed_funds - lagged fed_funds)
    df["rate_diff"] = df["fed_funds"].diff(1)
    df["m2_rate_interact"] = df["m2_mom"] * df["rate_diff"]

    # Seasonal dummies (month of year)
    for m in range(1, 13):
        df[f"month_{m}"] = (df.index.month == m).astype(int)

    # COVID regime dummy
    df["post_covid"] = (df.index >= "2020-01-01").astype(int)

    df = df.dropna()
    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc[:TRAIN_END]
    test  = df.loc[TEST_START:]
    return train, test


def get_xy(df: pd.DataFrame, target: str = "inflation_rate") -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [target, "cpi", "cpi_mom", "inflation_rate"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df[target]
    return X, y


def save_processed(df: pd.DataFrame, filename: str = "features.csv") -> None:
    path = PROCESSED_DIR / filename
    df.to_csv(path)
    print(f"Saved {len(df)} rows to {path}")


def load_processed(filename: str = "features.csv") -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the feature pipeline first.")
    return pd.read_csv(path, index_col=0, parse_dates=True)


if __name__ == "__main__":
    from src.ingest import load_raw
    raw = load_raw()
    df = build_feature_matrix(raw)
    train, test = split(df)
    save_processed(df)
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    print(f"Features: {df.shape[1]} columns")
