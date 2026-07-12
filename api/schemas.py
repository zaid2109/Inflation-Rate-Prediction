"""Pydantic request/response models for the FastAPI service."""

from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    m2: float          = Field(..., description="M2 money supply (billions USD)")
    unemployment: float = Field(..., description="Unemployment rate (%)")
    fed_funds: float   = Field(..., description="Federal funds rate (%)")
    oil_wti: float     = Field(..., description="WTI crude oil price (USD/barrel)")
    ppi: float         = Field(..., description="Producer Price Index")
    gdp_growth: float  = Field(..., description="Real GDP growth rate (%)")
    month: int         = Field(..., ge=1, le=12, description="Month of forecast (1–12)")
    model: Literal["linear", "xgboost", "lstm", "ensemble", "arima", "best"] = Field(
        "best", description="Which model to use for prediction"
    )

    model_config = {"json_schema_extra": {"example": {
        "m2": 21500.0,
        "unemployment": 3.7,
        "fed_funds": 5.25,
        "oil_wti": 85.0,
        "ppi": 230.0,
        "gdp_growth": 2.1,
        "month": 6,
        "model": "best",
    }}}


class PredictResponse(BaseModel):
    inflation_rate: float = Field(..., description="Predicted inflation rate (%)")
    model_used: str
    unit: str = "percent year-over-year"
    lower_bound_90: float | None = Field(None, description="90% conformal prediction lower bound")
    upper_bound_90: float | None = Field(None, description="90% conformal prediction upper bound")
    confidence_level: float | None = Field(None, description="Confidence level for bounds")


class ModelInfo(BaseModel):
    name: str
    is_best: bool
    path: str


class HistoryItem(BaseModel):
    date: str
    actual: float | None
    predicted: float | None
    model: str
