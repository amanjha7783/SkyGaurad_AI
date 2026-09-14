import pytest
import pandas as pd
import json
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.anomaly.inject_anomalies import inject_anomalies

@pytest.fixture
def dummy_clean_data(tmp_path):
    # Need enough rows to inject anomalies successfully without colliding too much
    rows = []
    for i in range(100):
        rows.append(f"2023-01-01 {i%24:02d}:00:00,S1,20.0,1012.0,55.0")
    
    data = "timestamp,station_id,temperature_c,pressure_hpa,relative_humidity_pct\n" + "\n".join(rows)
    input_path = tmp_path / "clean_aws.csv"
    with open(input_path, "w") as f:
        f.write(data)
    return str(input_path)

@pytest.fixture
def dummy_config(tmp_path):
    config = {
        "random_seed": 42,
        "anomaly_percentage": 0.2, # 20 rows out of 100
        "anomaly_classes": {
            "SPIKE": {"duration_range": [1, 1], "magnitude_range": [10, 20], "affected_parameters": ["temperature_c"]},
            "REAL_WEATHER_EVENT": {"duration_range": [2, 2], "magnitude_range": [5, 5], "affected_parameters": ["temperature_c"]}
        }
    }
    config_path = tmp_path / "anomaly_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)

def test_anomaly_injection(dummy_clean_data, dummy_config, tmp_path):
    output_path = str(tmp_path / "anomalies.csv")
    report_path = str(tmp_path / "report.json")
    
    report = inject_anomalies(dummy_clean_data, dummy_config, output_path, report_path)
    
    # Check outputs
    assert report['total_rows'] == 100
    assert report['injected_rows'] > 0
    
    df = pd.read_csv(output_path)
    
    # Check columns
    expected_cols = [
        'is_anomaly', 'anomaly_type', 'anomaly_subtype', 'anomaly_severity', 
        'anomaly_start', 'anomaly_end', 'fault_parameter', 'fault_source', 'ground_truth_reason'
    ]
    for c in expected_cols:
        assert c in df.columns
        
    # Check logic
    anomalies = df[df['fault_source'] == 'synthetic_injection']
    assert len(anomalies) > 0
    
    # Check REAL_WEATHER_EVENT has is_anomaly = 0
    real_events = anomalies[anomalies['anomaly_type'] == 'REAL_WEATHER_EVENT']
    if not real_events.empty:
        assert (real_events['is_anomaly'] == 0).all()
        assert (real_events['anomaly_severity'] == 'LOW').all()
        
    # Check SPIKE has is_anomaly = 1
    spikes = anomalies[anomalies['anomaly_type'] == 'SPIKE']
    if not spikes.empty:
        assert (spikes['is_anomaly'] == 1).all()
        # SPIKE isn't in the specific list in get_severity, so it defaults to MEDIUM or something (we set high/med/etc)
        # We can just check it's 1
