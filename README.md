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
| `GET`  | `/models`  | List trained model files |
| `GET`  | `/history` | Last 24 months actual vs. predicted |
| `GET`  | `/health`  | Service health check |

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
| XGBoost | xgboost | 5-fold TimeSeriesSplit grid search |
| LSTM | TensorFlow/Keras | 2-layer stacked, lookback = 24 months, early stopping |

## Evaluation Targets

| Metric | Target |
|--------|--------|
| MAE | < 0.3 percentage points |
| RMSE | < 0.5 percentage points |
| MAPE | < 5 % |
| R² | > 0.90 |
| Directional Accuracy | > 70 % |

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
│   ├── train.py          # Model training & MLflow logging
│   ├── evaluate.py       # Metrics & plots
│   └── predict.py        # Inference utilities
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
