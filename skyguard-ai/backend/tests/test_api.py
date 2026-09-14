from fastapi.testclient import TestClient
import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.main import app
from app.core.database import Base, engine, get_db
from app.models import models

# Ensure tables are created in the test DB
Base.metadata.create_all(bind=engine)

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_stations_empty():
    response = client.get("/api/stations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_predict_endpoint():
    payload = {
        "model": "lstm",
        "temperature_c": 50.0,
        "pressure_hpa": 1000.0,
        "relative_humidity_pct": 50.0
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert data["is_anomaly"] == 1 # Based on our dummy logic (>45C)

def test_statistics_endpoint():
    response = client.get("/api/statistics")
    assert response.status_code == 200
    data = response.json()
    assert "total_stations" in data
    assert "total_observations" in data
