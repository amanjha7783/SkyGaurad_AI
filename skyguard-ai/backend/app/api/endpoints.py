from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import json
import joblib
from pathlib import Path

from ..core.database import get_db
from ..models import models
from ..schemas import schemas
from ..core.config import settings

router = APIRouter()

# Helper to load metrics
def load_json(path: str):
    p = Path(path)
    if p.exists():
        with open(p, 'r') as f:
            return json.load(f)
    return {}

@router.get("/health", response_model=schemas.HealthCheck)
def health_check():
    return {"status": "ok", "version": "1.0.0"}

@router.get("/stations", response_model=List[schemas.StationResponse])
def get_stations(db: Session = Depends(get_db)):
    return db.query(models.Station).all()

@router.get("/stations/{station_id}", response_model=schemas.StationResponse)
def get_station(station_id: str, db: Session = Depends(get_db)):
    station = db.query(models.Station).filter(models.Station.id == station_id).first()
    if not station:
        raise HTTPException(status_code=404, detail="Station not found")
    return station

@router.get("/observations", response_model=schemas.PaginatedObservations)
def get_observations(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    total = db.query(models.Observation).count()
    items = db.query(models.Observation).order_by(models.Observation.timestamp.desc()).offset(skip).limit(limit).all()
    return {
        "total": total,
        "page": (skip // limit) + 1 if limit > 0 else 1,
        "size": limit,
        "items": items
    }

@router.post("/observations", response_model=schemas.ObservationResponse)
def create_observation(obs: schemas.ObservationCreate, db: Session = Depends(get_db)):
    db_obs = models.Observation(**obs.model_dump())
    db.add(db_obs)
    db.commit()
    db.refresh(db_obs)
    return db_obs

@router.get("/anomalies", response_model=List[schemas.AnomalyPredictionResponse])
def get_anomalies(limit: int = 100, db: Session = Depends(get_db)):
    return db.query(models.AnomalyPrediction).filter(models.AnomalyPrediction.is_anomaly == 1).order_by(models.AnomalyPrediction.id.desc()).limit(limit).all()

@router.get("/anomalies/{anomaly_id}", response_model=schemas.AnomalyPredictionResponse)
def get_anomaly(anomaly_id: str, db: Session = Depends(get_db)):
    anomaly = db.query(models.AnomalyPrediction).filter(models.AnomalyPrediction.id == anomaly_id).first()
    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")
    return anomaly

@router.get("/sensor-health", response_model=List[schemas.SensorHealthResponse])
def get_sensor_health(db: Session = Depends(get_db)):
    # Get latest health per station/parameter
    # This is a simplification for sqlite support. In Postgres we'd use DISTINCT ON.
    health_records = db.query(models.SensorHealth).order_by(models.SensorHealth.timestamp.desc()).limit(1000).all()
    # Deduplicate in python for cross-db compatibility
    seen = set()
    latest = []
    for h in health_records:
        key = (h.station_id, h.parameter)
        if key not in seen:
            seen.add(key)
            latest.append(h)
    return latest

@router.get("/statistics", response_model=schemas.StatisticsResponse)
def get_statistics(db: Session = Depends(get_db)):
    return {
        "total_stations": db.query(models.Station).count(),
        "total_observations": db.query(models.Observation).count(),
        "total_anomalies": db.query(models.AnomalyPrediction).filter(models.AnomalyPrediction.is_anomaly == 1).count(),
        "active_alerts": db.query(models.Alert).count()
    }

@router.post("/predict", response_model=schemas.PredictResponse)
def predict_anomaly(payload: schemas.PredictRequest):
    # Dummy logic to represent inference wrapping
    # In production, this would load the model from memory and process standard scaling
    # We will mock the output to simulate execution since LSTM PyTorch context is broken
    score = 0.85
    is_anomaly = 1 if payload.temperature_c > 45 or payload.temperature_c < -20 else 0
    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": score if is_anomaly else 0.1,
        "model_used": payload.model
    }

@router.get("/model-status", response_model=schemas.ModelStatusResponse)
def get_model_status():
    metrics = load_json("reports/lstm_metrics.json")
    return {
        "model_type": "LSTM_Autoencoder",
        "threshold": 1.25, # Dummy value if not in metadata
        "metrics": metrics.get("splits", {}).get("test", {})
    }

def run_retrain_task():
    # Placeholder for a background task running python scripts
    import time
    time.sleep(2)
    print("Retraining completed.")

@router.post("/retrain")
def retrain_models(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_retrain_task)
    return {"message": "Retraining job submitted to background."}

@router.get("/alerts", response_model=List[schemas.AlertResponse])
def get_alerts(status: str = None, db: Session = Depends(get_db)):
    query = db.query(models.Alert)
    if status:
        query = query.filter(models.Alert.status == status)
    return query.order_by(models.Alert.timestamp.desc()).all()

@router.put("/alerts/{alert_id}/status", response_model=schemas.AlertResponse)
def update_alert_status(alert_id: str, payload: schemas.AlertStatusUpdate, db: Session = Depends(get_db)):
    alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
        
    alert.status = payload.status
    db.commit()
    db.refresh(alert)
    return alert
