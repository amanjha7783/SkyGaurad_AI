import pandas as pd
import numpy as np
import yaml
import json
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ml.anomaly.anomaly_generators import AnomalyGenerators

def get_severity(anomaly_type: str) -> str:
    high_sev = ['MISSING_DATA', 'COMMUNICATION_ERROR', 'MULTI_SENSOR_FAILURE', 'FROZEN_SENSOR']
    med_sev = ['DRIFT', 'OFFSET', 'MULTIVARIATE_INCONSISTENCY', 'INTERMITTENT_FAILURE', 'HUMIDITY_SENSOR_FAILURE']
    if anomaly_type == 'REAL_WEATHER_EVENT':
        return 'LOW'
    if anomaly_type in high_sev:
        return 'CRITICAL'
    if anomaly_type in med_sev:
        return 'HIGH'
    return 'MEDIUM'

def inject_anomalies(input_path: str, config_path: str, output_path: str, report_path: str):
    print(f"Loading clean data from {input_path}")
    df = pd.read_csv(input_path)
    if df.empty:
        print("Empty dataset.")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    np.random.seed(config.get('random_seed', 42))
    
    # Initialize ground truth columns
    df['is_anomaly'] = 0
    df['anomaly_type'] = None
    df['anomaly_subtype'] = None
    df['anomaly_severity'] = None
    df['anomaly_start'] = None
    df['anomaly_end'] = None
    df['fault_parameter'] = None
    df['fault_source'] = 'synthetic_injection'
    df['ground_truth_reason'] = None
    
    anomaly_classes = config.get('anomaly_classes', {})
    total_rows = len(df)
    target_anomalies_rows = int(total_rows * config.get('anomaly_percentage', 0.1))
    
    if target_anomalies_rows == 0:
        target_anomalies_rows = 1 # inject at least something if dataset is tiny
        
    injected_rows = 0
    stats = {k: 0 for k in anomaly_classes.keys()}
    
    # Simple strategy: randomly pick a start index and apply a random anomaly
    max_attempts = 1000
    attempts = 0
    
    while injected_rows < target_anomalies_rows and attempts < max_attempts:
        attempts += 1
        a_type = np.random.choice(list(anomaly_classes.keys()))
        a_config = anomaly_classes[a_type]
        
        # Pick random duration
        dur_range = a_config.get('duration_range', [1, 1])
        duration = np.random.randint(dur_range[0], dur_range[1] + 1)
        
        # Pick random start index
        start_idx = np.random.randint(0, max(1, total_rows - duration))
        indices = list(range(start_idx, min(total_rows, start_idx + duration)))
        
        # Check if already anomalous
        if df.loc[indices, 'is_anomaly'].sum() > 0 or (df.loc[indices, 'anomaly_type'] == 'REAL_WEATHER_EVENT').any():
            continue
            
        # Select parameter
        params = a_config.get('affected_parameters', ['temperature_c'])
        if 'ALL' in params:
            param = ['temperature_c', 'pressure_hpa', 'relative_humidity_pct']
        else:
            param = np.random.choice(params)
            
        magnitude = None
        if 'magnitude_range' in a_config:
            mag_range = a_config['magnitude_range']
            magnitude = np.random.uniform(mag_range[0], mag_range[1])
            
        # Apply generator
        if a_type == 'SPIKE':
            df = AnomalyGenerators.apply_spike(df, indices, param, magnitude)
        elif a_type == 'FROZEN_SENSOR':
            df = AnomalyGenerators.apply_frozen_sensor(df, indices, param)
        elif a_type == 'DRIFT':
            df = AnomalyGenerators.apply_drift(df, indices, param, magnitude)
        elif a_type == 'OFFSET':
            df = AnomalyGenerators.apply_offset(df, indices, param, magnitude)
        elif a_type == 'MISSING_DATA':
            df = AnomalyGenerators.apply_missing_data(df, indices, param)
        elif a_type == 'COMMUNICATION_ERROR':
            df = AnomalyGenerators.apply_communication_error(df, indices, param) # param is list here
        elif a_type == 'PRESSURE_ANOMALY':
            df = AnomalyGenerators.apply_pressure_anomaly(df, indices, magnitude)
            param = 'pressure_hpa'
        elif a_type == 'HUMIDITY_SENSOR_FAILURE':
            val = a_config.get('value', 0.0)
            df = AnomalyGenerators.apply_humidity_sensor_failure(df, indices, val)
            param = 'relative_humidity_pct'
        elif a_type == 'TEMPORAL_ANOMALY':
            df = AnomalyGenerators.apply_temporal_anomaly(df, indices, param)
        elif a_type == 'MULTIVARIATE_INCONSISTENCY':
            df = AnomalyGenerators.apply_multivariate_inconsistency(df, indices)
            param = "temperature_c|relative_humidity_pct"
        elif a_type == 'MULTI_SENSOR_FAILURE':
            df = AnomalyGenerators.apply_multi_sensor_failure(df, indices, param)
        elif a_type == 'INTERMITTENT_FAILURE':
            df = AnomalyGenerators.apply_intermittent_failure(df, indices, param, magnitude)
        elif a_type == 'REAL_WEATHER_EVENT':
            df = AnomalyGenerators.apply_real_weather_event(df, indices, param, magnitude)
            
        # Record metadata
        start_time = df.loc[indices[0], 'timestamp']
        end_time = df.loc[indices[-1], 'timestamp']
        
        is_anom = 1 if a_type != 'REAL_WEATHER_EVENT' else 0
        
        df.loc[indices, 'is_anomaly'] = is_anom
        df.loc[indices, 'anomaly_type'] = a_type
        df.loc[indices, 'anomaly_subtype'] = 'synthetic'
        df.loc[indices, 'anomaly_severity'] = get_severity(a_type)
        df.loc[indices, 'anomaly_start'] = start_time
        df.loc[indices, 'anomaly_end'] = end_time
        df.loc[indices, 'fault_parameter'] = str(param)
        df.loc[indices, 'ground_truth_reason'] = f"Injected {a_type}"
        
        injected_rows += len(indices)
        stats[a_type] += 1
        
    print(f"Injection complete. Inserted {injected_rows} anomalous/event rows.")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    report = {
        "total_rows": total_rows,
        "injected_rows": injected_rows,
        "events_injected": stats
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inject controlled anomalies.")
    parser.add_argument('--input', type=str, default='data/processed/clean_aws.csv')
    parser.add_argument('--config', type=str, default='config/anomaly_config.yaml')
    parser.add_argument('--output', type=str, default='data/anomaly/anomaly_injected_aws.csv')
    parser.add_argument('--report', type=str, default='reports/anomaly_statistics.json')
    args = parser.parse_args()
    
    inject_anomalies(args.input, args.config, args.output, args.report)
