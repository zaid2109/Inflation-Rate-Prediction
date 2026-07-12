"""Data ingestion from the FRED API."""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

FRED_SERIES = {
    "cpi":          "CPIAUCSL",   # CPI All Urban Consumers, monthly
    "m2":           "M2SL",       # M2 Money Supply, monthly
    "unemployment": "UNRATE",     # Unemployment Rate, monthly
    "fed_funds":    "FEDFUNDS",   # Federal Funds Rate, monthly
    "gdp_growth":   "A191RL1Q225SBEA",  # Real GDP growth, quarterly
    "oil_wti":      "DCOILWTICO", # WTI Crude Oil Price, daily -> resample monthly
    "ppi":          "PPIACO",     # Producer Price Index, monthly
}

START_DATE = "1990-01-01"
END_DATE   = "2025-12-31"


def get_fred_client() -> Fred:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Copy .env.example to .env and add your key.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=api_key)


def download_fred_series(fred: Fred, series_id: str, name: str) -> pd.Series:
    data = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
    data.name = name
    return data


def download_all(force: bool = False) -> dict[str, pd.Series]:
    fred = get_fred_client()
    results = {}

    for name, series_id in FRED_SERIES.items():
        out_path = RAW_DIR / f"{name}.csv"
        if out_path.exists() and not force:
            series = pd.read_csv(out_path, index_col=0, parse_dates=True).squeeze()
            series.name = name
        else:
            print(f"Downloading {name} ({series_id})...")
            series = download_fred_series(fred, series_id, name)
            if name == "oil_wti":
                series = series.resample("MS").mean()
            elif name == "gdp_growth":
                series = series.resample("MS").ffill()
            series.to_csv(out_path)
        results[name] = series

    return results


def load_raw() -> dict[str, pd.Series]:
    """Load previously downloaded raw CSVs without hitting the API."""
    results = {}
    for name in FRED_SERIES:
        path = RAW_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run download_all() first."
            )
        series = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
        series.name = name
        results[name] = series
    return results


if __name__ == "__main__":
    data = download_all()
    print("Downloaded series:")
    for name, s in data.items():
        print(f"  {name}: {len(s)} observations ({s.index[0].date()} – {s.index[-1].date()})")
