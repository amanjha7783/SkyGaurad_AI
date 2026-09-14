import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.models.sensor_health import SensorHealthEngine

def test_sensor_health_engine():
    engine = SensorHealthEngine(window_hours=3)
    
    # Create mock data for one station
    timestamps = pd.date_range("2023-01-01 00:00:00", periods=5, freq="1h")
    df = pd.DataFrame({
        'timestamp': timestamps,
        'station_id': ['S1'] * 5,
        'temperature_c_is_missing': [0, 1, 0, 0, 0], # missing at t=1
        'temperature_persistence':  [0, 0, 1, 2, 3], # frozen at t=4
        'lstm_prediction':          [0, 0, 1, 1, 0], # anomalies at t=2, t=3
        'lstm_reconstruction_error':[0, 0, 2.0, 6.0, 0] # severe anomaly at t=3
    })
    
    # Process
    result = engine.evaluate_parameter_health(df, 'temperature_c')
    
    # Assertions
    # t=0: healthy
    assert result.loc[0, 'temperature_health_score'] == 100
    assert "healthy" in result.loc[0, 'temperature_health_reason']
    
    # t=1: 1 missing (deduct 10) -> score 90
    assert result.loc[1, 'temperature_health_score'] == 90
    assert "Missing data" in result.loc[1, 'temperature_health_reason']
    
    # t=2: 1 missing (in last 3h), 1 anomaly (deduct 15) -> deduct 25 -> score 75
    assert result.loc[2, 'temperature_health_score'] == 75
    
    # t=3: 1 missing (in last 3h), 2 anomalies (deduct 30). Error average in last 3h is (0+2+6)/3 = 2.6 < 5.0 so no severe penalty. Total deduct = 40 -> score 60
    assert result.loc[3, 'temperature_health_score'] == 60
    
    # t=4: 0 missing (in last 3h since t=1 is outside window of 3h: t=2,3,4). 
    # anomalies in last 3h (t=2,3,4) = 2 -> deduct 30
    # Error average in last 3h (t=2,3,4) = (2+6+0)/3 = 2.66 < 5.0 -> no severe penalty
    # frozen sensor: persistence is 3 >= threshold(3) -> deduct 40
    # Total deduct = 70 -> score 30
    assert result.loc[4, 'temperature_health_score'] == 30
    assert "Frozen sensor" in result.loc[4, 'temperature_health_reason']
    assert result.loc[4, 'temperature_maintenance_required'] == 1
    assert result.loc[4, 'temperature_predicted_failure_risk'] == 0.70
    
    print("All sensor health tests passed!")

if __name__ == "__main__":
    test_sensor_health_engine()
