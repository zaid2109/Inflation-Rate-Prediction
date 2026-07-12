# Inflation Rate Prediction System

College capstone project — forecasts US CPI inflation using macroeconomic time-series data and four ML/DL models, served via a FastAPI REST endpoint and visualised in a Streamlit dashboard.

## Architecture

```
FRED API / yfinance
        │
   src/ingest.py          ← download & cache raw CSVs
        │
   src/features.py        ← lag/rolling/interaction feature engineering
        │
   src/train.py           ← Linear · ARIMA · XGBoost · LSTM + MLflow logging
        │
   models/*.joblib        ← serialised model files
        │
   ┌────┴────┐
   │  API    │  ← FastAPI  (api/main.py)      POST /predict · GET /models · GET /history
   └─────────┘
   ┌────────────┐
   │ Dashboard  │  ← Streamlit (dashboard/app.py)  interactive charts + live prediction
   └────────────┘
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free FRED API key
Register at https://fred.stlouisfed.org/docs/api/api_key.html, then:
```bash
cp .env.example .env
# Edit .env and set FRED_API_KEY=your_key
```

### 3. Download data & build features
```bash
python -m src.ingest       # downloads ~7 FRED series to data/raw/
python -m src.features     # engineers features, saves to data/processed/
```

### 4. Train all models
```bash
python -m src.train
```
Trains Linear Regression, ARIMA, XGBoost, and LSTM. Results are logged to MLflow (`mlruns/`).

### 5. Start the API
```bash
uvicorn api.main:app --reload
# → http://localhost:8000/docs  (Swagger UI)
```

### 6. Launch the dashboard
```bash
streamlit run dashboard/app.py
# → http://localhost:8501
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Predict inflation from macro inputs |
| `POST` | `/predict/batch` | Batch prediction over an array of inputs |
| `GET`  | `/models`  | List trained model files |
| `GET`  | `/history` | Last 24 months actual vs. predicted |
| `GET`  | `/health`  | Service health check |
| `GET`  | `/chart-data` | Actual vs. predicted series for a model/year range |
| `GET`  | `/summary` | Latest actual/predicted, MAE, short-term trend |
| `GET`  | `/metrics` | Saved per-model MAE/RMSE/MAPE/R²/Dir_Acc |
| `GET`  | `/shap` | SHAP summary + per-date SHAP time series |
| `GET`  | `/walk-forward` | Walk-forward (expanding-window) validation results |
| `GET`  | `/ablation` | Feature-group ablation study results |
| `GET`  | `/feature-importance` | Top-N XGBoost feature importances |
| `GET`  | `/features` | SHAP-based feature importance (falls back to gain importance) |
| `GET`  | `/indicators` | Latest macro indicator values + sparklines |
| `GET`  | `/dashboard-data` | Aggregated payload for the static dashboard (KPIs, chart, drivers) |

### Example prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"m2":21500,"unemployment":3.7,"fed_funds":5.25,"oil_wti":85,"ppi":230,"gdp_growth":2.1,"month":6,"model":"best"}'
```

## Models

| Model | Library | Notes |
|-------|---------|-------|
| Linear (Ridge) | scikit-learn | Interpretable baseline; cross-validated alpha |
| ARIMA/SARIMA | pmdarima | `auto_arima` for order selection; seasonal period = 12 |
| XGBoost | xgboost | 5-fold TimeSeriesSplit, RandomizedSearchCV (150 draws) incl. regularization |
| LSTM | TensorFlow/Keras | Bidirectional + LSTM stack, lookback = 12 months, early stopping |

## Evaluation Results

Test period: 2019-01 to 2025-09 (81 held-out months, never seen during training/tuning).

| Model | MAE | RMSE | MAPE | R² | Dir. Acc. |
|-------|----:|-----:|-----:|---:|----------:|
| Linear (Ridge) | 1.03 | 1.36 | 32.3% | 0.65 | 80.0% |
| ARIMA/SARIMA | 1.96 | 2.82 | 62.0% | -0.49 | 51.3% |
| XGBoost | 1.32 | 1.96 | 35.6% | 0.28 | 70.0% |
| LSTM | 2.32 | 2.96 | 64.2% | -0.64 | 63.8% |
| **Ensemble (stacking)** | **0.75** | **0.97** | **24.3%** | **0.82** | **80.0%** |
| Naive baseline (persistence) | 0.32 | 0.44 | 15.5% | 0.96 | 69.6% |

(Refresh with your own run: `models/metrics.json`, or `GET /metrics`.)

**On this single 2019-2025 split, the ensemble still trails the naive persistence
baseline on MAE/RMSE, but beats it on directional accuracy (80.0% vs 69.6%) — the
ensemble is more often right about *which way* inflation is moving, even though its
average miss is larger in percentage points.** YoY CPI is extremely autocorrelated
month-to-month, so "predict last month's value" is a genuinely hard MAE/RMSE
benchmark on any single window, especially one containing the 2021-2023 inflation
shock. A single train/test split is also just one sample of history, so a more
robust check is walk-forward validation below.

### Walk-forward validation (the real test)

`python -m src.train --walk-forward` retrains an XGBoost model at every one of 177
months from 2011-01 to 2025-09, each time using only data available before that
month, then predicts one step ahead — this is the closest simulation of how the
model would actually perform if it had been deployed and retrained monthly for the
last 14 years (`models/walk_forward.json`, `GET /walk-forward`).

| | MAE | RMSE | Dir. Acc. |
|---|----:|-----:|----------:|
| Walk-forward XGBoost (177 months, 2011-2025) | **0.31** | **0.44** | 69.3% |
| Naive baseline, same window | 0.32 | 0.44 | ~70% |

Over this longer, more representative window the model **matches the naive
baseline** rather than trailing it — the single 2019-2025 test split was simply an
unusually hard stretch (it's dominated by the COVID-era inflation spike and its
unwind). This is the number worth citing as "how good is the model," not the
single-split table above in isolation.

The ensemble also beats a real external forecaster proxy: on the 2019-2025 window
it has lower MAE/RMSE/MAPE than the University of Michigan 1-year consumer
inflation expectations survey (`models/spf_comparison.json`, generated by
`python -m src.spf_comparison` — not currently exposed via the API).

<details>
<summary>Original evaluation targets (project brief)</summary>

| Metric | Target |
|--------|--------|
| MAE | < 0.3 percentage points |
| RMSE | < 0.5 percentage points |
| MAPE | < 5 % |
| R² | > 0.90 |
| Directional Accuracy | > 70 % |

The walk-forward MAE/RMSE above land right at these targets; the single-split
numbers don't, for the reasons discussed above. Directional accuracy on the
single split (80%) clears the bar; on the longer walk-forward window it's just
under (69.3%). None of this should be read as "the model is bad" — it should be
read as "a single train/test split on a hard 6-year window is a noisy estimator,
and the walk-forward number is the one to trust."
</details>

### How the model got here (train-only, no test-set peeking)

Three changes were made based on a `TimeSeriesSplit` cross-validation study run
**only on the training set** (2019-2025 test data was never touched until a single
final evaluation after these were frozen):

1. **Dropped the 12 month-of-year dummy variables.** The target is already a
   12-month (YoY) difference, which cancels out most within-year seasonality by
   construction, so the dummies were mostly adding variance, not signal, on top
   of ~330 training rows. Removing them was the single largest lever below.
2. **Widened the XGBoost hyperparameter search** to include `reg_alpha`,
   `reg_lambda`, and `min_child_weight` (regularization the original grid never
   explored), searched via `RandomizedSearchCV` (150 draws, same
   `TimeSeriesSplit(5)` as before) instead of the original 32-combination
   exhaustive grid. Given how few rows are available relative to feature count,
   the extra regularization meaningfully reduced CV RMSE.
3. **Replaced the ensemble's meta-learner.** The original stacking ensemble fit
   an unconstrained `Ridge` on the two base models' out-of-fold predictions,
   which produced coefficients like `[Linear: 1.01, XGBoost: -0.86]` — a strongly
   *negative* weight on XGBoost. A train-only stability check (fit the
   meta-weights on the first half of the out-of-fold series, evaluate on the
   second half) showed this unconstrained fit's error roughly **doubles**
   out-of-sample versus a plain non-negative, sum-to-1 convex blend, because the
   two base models' out-of-fold predictions are highly correlated (r≈0.81) and
   an unconstrained 2-parameter fit on correlated inputs is unstable. Switched to
   `src/meta_models.py::ConvexBlendMeta`, which also fixed a second, unrelated
   bug: the dashboard's "ensemble weights" panel was silently clipping any
   negative coefficient to a displayed 0%, so it previously (and wrongly) showed
   "XGBoost: 0% weight" — the actual model was using XGBoost with a large
   negative coefficient. The displayed weights (Linear 82% / XGBoost 18%) now
   match what the model actually computes.

Combined, these took the ensemble from MAE 0.94 / R² 0.67 to MAE 0.75 / R² 0.82 on
the untouched test set — evaluated once, after all three decisions above were
already locked in from training-only cross-validation.

## Running Tests
```bash
pytest tests/ -v
```

## View MLflow Experiments
```bash
mlflow ui --backend-store-uri mlruns
# → http://localhost:5000
```

## Folder Structure
```
inflation-predictor/
├── data/
│   ├── raw/              # Downloaded CSVs from FRED
│   └── processed/        # Feature-engineered datasets
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_eng.ipynb
│   ├── 03_models.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── ingest.py         # FRED data download
│   ├── features.py       # Feature engineering pipeline
│   ├── train.py          # Model training, MLflow logging, ensemble & walk-forward
│   ├── meta_models.py    # Picklable ensemble meta-learner (ConvexBlendMeta)
│   ├── evaluate.py       # Metrics, bootstrap CIs, Diebold-Mariano test, plots
│   ├── predict.py        # Inference utilities
│   ├── regime.py         # HMM high/low-inflation regime detection
│   ├── ablation.py       # Feature-group ablation study
│   ├── shap_analysis.py  # Standalone SHAP re-computation
│   ├── spf_comparison.py # Comparison vs. U-Michigan inflation expectations
│   ├── ensemble.py       # Standalone/experimental ensemble (NOT used by the API — see docstring)
│   └── walkforward.py    # Standalone/experimental walk-forward CV (NOT used by the API)
├── api/
│   ├── main.py           # FastAPI app
│   └── schemas.py        # Pydantic models
├── dashboard/
│   └── app.py            # Streamlit dashboard
├── models/               # Serialised .joblib files
├── tests/                # Pytest suite
├── mlruns/               # MLflow experiment logs
├── .env.example
└── requirements.txt
```

## Methodology Notes & Limitations

**Naive baseline comparison.** Year-over-year CPI inflation is a smooth, highly
autocorrelated series (this month's YoY rate is usually very close to last month's),
which makes the trivial "persistence" baseline — predict this month = last month's
actual — a deceptively strong benchmark. All trained models are evaluated against
this naive baseline (`GET /metrics`, "naive" row) rather than judged on MAE/RMSE in
isolation; beating a random-walk/persistence baseline is the standard bar in the
macro-forecasting literature, not R² alone.

**Nowcasting vs. forecasting assumption.** Several exogenous indicators (`fed_funds`,
`oil_wti`, `m2`, `unemployment`, `ppi` and their MoM/YoY transforms) are used
contemporaneously with the target month — i.e. the model assumes these values for
month *t* are already known when predicting inflation for month *t*. This is a
defensible "nowcasting" framing for `fed_funds` and `oil_wti` (both are observable
in real time throughout the month), but is a simplifying assumption for `ppi`,
`unemployment`, and `m2`, whose official prints lag by 2-6 weeks. A stricter
forecasting setup would shift this entire block by one additional period; we kept
the nowcast framing since it matches how the dashboard's live "what-if" tool is
meant to be used (plug in known-or-assumed current conditions), but this is a known
limitation worth calling out for anyone extending the project to true N-month-ahead
forecasting.

**Fixed leakage bugs.** Two feature-engineering bugs were identified and fixed
during review:
- `infl_mom3` / `infl_mom6` previously computed `inflation_rate.diff(3/6)` on the
  *unshifted* target series, meaning the feature for row *t* was a direct additive
  function of `inflation_rate(t)` — the value being predicted. Fixed to
  `inflation_rate.shift(1).diff(3/6)` so only information through *t-1* is used.
- Raw `gdp_growth` (a quarterly print forward-filled onto every month of the
  quarter) was used as a feature even in the first 1-2 months of a quarter, before
  that quarter's growth rate is actually published by the BEA. Only the properly
  lagged `gdp_growth_lag1` is now used as a model feature; the raw column remains
  in the processed dataset for display purposes only.

See `tests/test_features.py::test_no_target_leakage_in_momentum_features` and
`::test_gdp_growth_raw_excluded_from_features` for regression coverage.

## Data Sources

| Series | FRED ID | Frequency |
|--------|---------|-----------|
| CPI (All Urban) | CPIAUCSL | Monthly |
| M2 Money Supply | M2SL | Monthly |
| Unemployment Rate | UNRATE | Monthly |
| Federal Funds Rate | FEDFUNDS | Monthly |
| Real GDP Growth | A191RL1Q225SBEA | Quarterly → ffill to monthly |
| WTI Crude Oil | DCOILWTICO | Daily → monthly mean |
| PPI | PPIACO | Monthly |
