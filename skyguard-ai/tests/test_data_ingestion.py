import pytest
import pandas as pd
import yaml
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.preprocessing.data_loader import DataLoader

@pytest.fixture
def dummy_config(tmp_path):
    config = {
        "schema": {
            "timestamp": {"aliases": ["time"], "type": "datetime"},
            "station_id": {"aliases": ["stn"], "type": "string"},
            "temperature_c": {"aliases": ["temp"], "type": "float"},
            "pressure_hpa": {"aliases": ["pres"], "type": "float"},
            "relative_humidity_pct": {"aliases": ["rh"], "type": "float"},
            "latitude": {"aliases": ["lat"], "type": "float"},
            "longitude": {"aliases": ["lon"], "type": "float"},
            "source": {"aliases": [], "type": "string"}
        },
        "validation": {
            "temperature_c": {"min": -50, "max": 60},
            "pressure_hpa": {"min": 800, "max": 1100},
            "relative_humidity_pct": {"min": 0, "max": 100},
            "latitude": {"min": -90, "max": 90},
            "longitude": {"min": -180, "max": 180}
        }
    }
    config_path = tmp_path / "data_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)

@pytest.fixture
def dummy_data(tmp_path):
    # Includes an invalid temperature (100) and duplicate timestamp-station
    data = """time,stn,temp,pres,rh,lat,lon
2023-01-01 00:00:00,S1,25.5,1013.2,50,45.0,-120.0
2023-01-01 01:00:00,S1,100.0,1015.0,60,45.0,-120.0
2023-01-01 01:00:00,S1,26.0,1015.0,60,45.0,-120.0
2023-01-01 02:00:00,S2,,1010.0,55,46.0,-121.0
"""
    data_path = tmp_path / "dummy_raw.csv"
    with open(data_path, "w") as f:
        f.write(data)
    return str(data_path)

def test_data_loader_mapping_and_validation(dummy_config, dummy_data):
    loader = DataLoader(dummy_config)
    df, metadata = loader.ingest(dummy_data, "test_source")
    
    # Check shape: 1 duplicate removed, so 3 rows left
    assert len(df) == 3
    
    # Check standard columns
    assert "timestamp" in df.columns
    assert "station_id" in df.columns
    assert "temperature_c" in df.columns
    
    # Check invalid temperature handling (100 is out of bounds, so it should be NA/NaN)
    # The second row in the original file was dropped as duplicate? 
    # Wait, the duplicate logic keeps the first row. The first '01:00:00' had temp=100.0, 
    # which is invalidated. So it becomes NaN.
    # The duplicate (row 3 with temp 26.0) is dropped.
    out_of_bounds_row = df[df['timestamp'] == '2023-01-01 01:00:00']
    assert len(out_of_bounds_row) == 1
    assert pd.isna(out_of_bounds_row.iloc[0]['temperature_c'])
    
    # Check metadata
    assert metadata["row_count"] == 3
    assert "S1" in metadata["stations"]
    assert "test_source" == metadata["source_name"]
    
    # Check validation errors were reported
    errors = metadata["validation_report"]["validation_errors"]
    assert any("duplicate timestamp-station" in e for e in errors)
    assert any("temperature_c" in e and "out-of-bound" in e for e in errors)
