import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.features.feature_utils import (
    calculate_time_features,
    apply_station_rolling,
    apply_station_lags,
    apply_station_deltas,
    calculate_persistence
)
from ml.features.build_features import build_features

@pytest.fixture
def dummy_anomalous_data(tmp_path):
    # Two stations to test grouping
    rows = []
    
    # Station 1
    rows.append("2023-01-01 00:00:00,S1,20.0,1012.0,50.0,0")
    rows.append("2023-01-01 01:00:00,S1,21.0,1011.0,51.0,0")
    rows.append("2023-01-01 02:00:00,S1,21.0,1011.0,51.0,1") # Duplicate / frozen
    rows.append("2023-01-01 03:00:00,S1,22.0,1010.0,52.0,0")
    
    # Station 2
    rows.append("2023-01-01 00:00:00,S2,10.0,1020.0,40.0,0")
    rows.append("2023-01-01 01:00:00,S2,15.0,1015.0,45.0,1") # Spike
    
    data = "timestamp,station_id,temperature_c,pressure_hpa,relative_humidity_pct,is_anomaly\n" + "\n".join(rows)
    input_path = tmp_path / "anomaly_injected_aws.csv"
    with open(input_path, "w") as f:
        f.write(data)
    return str(input_path)

def test_feature_building_and_leakage(dummy_anomalous_data, tmp_path):
    output_path = str(tmp_path / "features.csv")
    build_features(dummy_anomalous_data, output_path)
    
    df = pd.read_csv(output_path)
    
    # Test grouping (S1 has 4 rows, S2 has 2 rows)
    s1 = df[df['station_id'] == 'S1'].sort_values('timestamp').reset_index(drop=True)
    s2 = df[df['station_id'] == 'S2'].sort_values('timestamp').reset_index(drop=True)
    
    # 1. Delta
    assert pd.isna(s1.loc[0, 'temperature_delta']) # First row has no past
    assert s1.loc[1, 'temperature_delta'] == 1.0 # 21.0 - 20.0
    
    # 2. Lag
    assert pd.isna(s1.loc[0, 'temperature_c_lag_1'])
    assert s1.loc[1, 'temperature_c_lag_1'] == 20.0
    
    # 3. Persistence
    assert s1.loc[0, 'temperature_persistence'] == 0
    assert s1.loc[1, 'temperature_persistence'] == 0
    assert s1.loc[2, 'temperature_persistence'] == 1 # 21.0 repeated
    
    # 4. Leakage verification
    # If S2 lag 1 leaked from S1, it would be 22.0 (S1's last row). It MUST be NaN.
    assert pd.isna(s2.loc[0, 'temperature_c_lag_1'])
    
    # 5. Time cyclical
    assert 'hour_sin' in df.columns
    assert 'day_of_year_cos' in df.columns
    
    # 6. Check required columns exist
    required_cols = [
        'rolling_temperature_mean_1h', 'rolling_temperature_std_1h',
        'temperature_zscore', 'pressure_delta', 'humidity_delta',
        'neighbor_temperature_difference'
    ]
    for c in required_cols:
        assert c in df.columns
