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

    # Inflation momentum: 3-month and 6-month rate of change in inflation_rate itself.
    # Shifted by 1 first — using diff() on the unshifted series would leak inflation_rate(t)
    # (the target) directly into a feature for row t.
    df["infl_mom3"] = df["inflation_rate"].shift(1).diff(3)
    df["infl_mom6"] = df["inflation_rate"].shift(1).diff(6)

    # Yield curve proxy: fed_funds spread over its own 12-month moving average
    # (steepening = growth expectations rising = leading inflation signal)
    df["yield_curve_proxy"] = df["fed_funds"] - df["fed_funds"].rolling(12).mean()

    # Real interest rate proxy: fed_funds minus trailing 12-month inflation
    df["real_rate"] = df["fed_funds"] - df["inflation_rate"].shift(1)

    # Month-of-year dummies were tried and dropped: the target is a 12-month
    # (YoY) difference, which already cancels out most within-year seasonality
    # by construction, so the dummies mainly added noise/variance without a
    # real signal (confirmed via train-only TimeSeriesSplit CV — see README).

    # COVID regime dummy
    df["post_covid"] = (df.index >= "2020-01-01").astype(int)

    df = df.dropna()
    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc[:TRAIN_END]
    test  = df.loc[TEST_START:]
    return train, test


def get_xy(df: pd.DataFrame, target: str = "inflation_rate") -> tuple[pd.DataFrame, pd.Series]:
    # "gdp_growth" is the raw quarterly print forward-filled onto every month in the
    # quarter — for the first 1-2 months of a quarter that value isn't published yet
    # (BEA releases the advance estimate ~1 month after quarter-end), so using it
    # contemporaneously leaks future information. Only the properly lagged
    # "gdp_growth_lag1" is safe to use as a feature; the raw column is kept in the
    # dataframe for display purposes only.
    drop_cols = [target, "cpi", "cpi_mom", "inflation_rate", "gdp_growth"]
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
