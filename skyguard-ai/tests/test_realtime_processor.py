import pytest
import sys
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.app.models import models
from backend.app.services.realtime_processor import RealtimeProcessor

# Use an in-memory SQLite database for testing
engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
models.Base.metadata.create_all(bind=engine)

def get_test_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

def test_realtime_processor():
    db = next(get_test_db())
    
    # Create station in DB first
    test_station = models.Station(id="TEST_01", name="Test Station", latitude=40.0, longitude=-74.0)
    db.add(test_station)
    db.commit()

    processor = RealtimeProcessor(db)

    # 1. Normal observation
    obs1 = {
        "station_id": "TEST_01",
        "timestamp": "2023-01-01T12:00:00",
        "temperature_c": 20.0,
        "pressure_hpa": 1013.0,
        "relative_humidity_pct": 50.0
    }
    
    res1 = processor.process_observation(obs1)
    assert res1["prediction"] in [0, 1]
    assert res1["health_score"] == 100.0
    
    # 2. Frozen sensor observation (triggering persistence)
    obs2 = {
        "station_id": "TEST_01",
        "timestamp": "2023-01-01T13:00:00",
        "temperature_c": 20.0,
        "pressure_hpa": 1013.0,
        "relative_humidity_pct": 50.0
    }
    obs3 = {
        "station_id": "TEST_01",
        "timestamp": "2023-01-01T14:00:00",
        "temperature_c": 20.0,
        "pressure_hpa": 1013.0,
        "relative_humidity_pct": 50.0
    }
    obs4 = {
        "station_id": "TEST_01",
        "timestamp": "2023-01-01T15:00:00",
        "temperature_c": 20.0,
        "pressure_hpa": 1013.0,
        "relative_humidity_pct": 50.0
    }
    obs5 = {
        "station_id": "TEST_01",
        "timestamp": "2023-01-01T16:00:00",
        "temperature_c": 20.0,
        "pressure_hpa": 1013.0,
        "relative_humidity_pct": 50.0
    }
    obs6 = {
        "station_id": "TEST_01",
        "timestamp": "2023-01-01T17:00:00",
        "temperature_c": 20.0,
        "pressure_hpa": 1013.0,
        "relative_humidity_pct": 50.0
    }
    obs7 = {
        "station_id": "TEST_01",
        "timestamp": "2023-01-01T18:00:00",
        "temperature_c": 20.0,
        "pressure_hpa": 1013.0,
        "relative_humidity_pct": 50.0
    }
    obs8 = {
        "station_id": "TEST_01",
        "timestamp": "2023-01-01T19:00:00",
        "temperature_c": 20.0,
        "pressure_hpa": 1013.0,
        "relative_humidity_pct": 50.0
    }

    processor.process_observation(obs2)
    processor.process_observation(obs3)
    processor.process_observation(obs4)
    processor.process_observation(obs5)
    processor.process_observation(obs6)
    processor.process_observation(obs7)
    res8 = processor.process_observation(obs8)
    
    # Health should drop significantly due to frozen sensor
    assert res8["prediction"] == 1
    assert res8["health_score"] < 40.0
    assert "processing_time_ms" in res8
    
    # Verify database
    obs_count = db.query(models.Observation).count()
    assert obs_count == 8
    
    health_count = db.query(models.SensorHealth).count()
    assert health_count == 8
    
    alert_count = db.query(models.Alert).count()
    # At least one alert should be triggered because of health dropping < 40
    assert alert_count > 0
