import pytest
import pandas as pd
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.preprocessing.clean_data import run_cleaning_pipeline

@pytest.fixture
def dummy_standard_data(tmp_path):
    data = """timestamp,station_id,temperature_c,pressure_hpa,relative_humidity_pct
2023-01-01 00:00:00,S1,20.0,1012.0,55.0
2023-01-01 01:00:00,S1,150.0,1012.0,55.0
2023-01-01 02:00:00,S1,22.0,700.0,55.0
2023-01-01 03:00:00,S1,21.0,1012.0,150.0
2023-01-01 04:00:00,S1,20.5,1011.0,56.0
2023-01-01 04:00:00,S1,20.5,1011.0,56.0
2023-01-01 05:00:00,,20.0,1010.0,57.0
2023-01-01 06:00:00,S1,,1010.0,57.0
"""
    # 1. Clean row
    # 2. Invalid temp (>60)
    # 3. Invalid pressure (<800)
    # 4. Invalid RH (>100)
    # 5. Clean row
    # 6. Duplicate of 5
    # 7. Missing station
    # 8. Missing temp

    input_path = tmp_path / "baseline.csv"
    with open(input_path, "w") as f:
        f.write(data)
    return str(input_path)

def test_cleaning_pipeline(dummy_standard_data, tmp_path):
    clean_out = str(tmp_path / "clean.csv")
    quar_out = str(tmp_path / "quarantine.csv")
    rep_out = str(tmp_path / "report.csv")
    sum_out = str(tmp_path / "summary.json")
    
    summary = run_cleaning_pipeline(dummy_standard_data, clean_out, quar_out, rep_out, sum_out)
    
    # 8 input rows, 1 clean, 7 quarantined (actually 2 clean row indices but one is duplicate)
    # Wait, both duplicates are flagged as 'duplicate_record' because keep=False in pandas duplicated by default.
    # So BOTH rows 5 and 6 will be quarantined.
    # So 1 clean row (the first one). Wait, is that right? Let's check logic:
    # df.duplicated(keep=False) flags ALL duplicates. So both row 5 and 6 go to quarantine. This is fine since it requires manual resolution to decide which one is correct.
    
    assert summary['input_rows'] == 8
    assert summary['clean_rows'] == 1
    assert summary['quarantined_rows'] == 7
    
    assert summary['invalid_temp_flagged'] == 1
    assert summary['invalid_pressure_flagged'] == 1
    assert summary['invalid_humidity_flagged'] == 1
    assert summary['duplicates_flagged'] == 2
    
    quarantine_df = pd.read_csv(quar_out)
    assert 'quarantine_reasons' in quarantine_df.columns
    
    # Check that report has reasons
    report_df = pd.read_csv(rep_out)
    assert len(report_df) == 7
    assert 'quarantine_reasons' in report_df.columns
