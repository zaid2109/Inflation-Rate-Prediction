"""Integration tests for the FastAPI service (no model files required)."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_models_empty(client):
    # Without trained model files the list should be empty (or return what exists)
    resp = client.get("/models")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_predict_missing_model(client):
    payload = {
        "m2": 21000.0,
        "unemployment": 3.7,
        "fed_funds": 5.25,
        "oil_wti": 85.0,
        "ppi": 230.0,
        "gdp_growth": 2.1,
        "month": 6,
        "model": "linear",
    }
    resp = client.post("/predict", json=payload)
    # 503 when model file missing, 200 if file exists
    assert resp.status_code in (200, 503)


def test_predict_invalid_month(client):
    payload = {
        "m2": 21000.0,
        "unemployment": 3.7,
        "fed_funds": 5.25,
        "oil_wti": 85.0,
        "ppi": 230.0,
        "gdp_growth": 2.1,
        "month": 13,
        "model": "linear",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
