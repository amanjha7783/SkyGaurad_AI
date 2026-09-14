import os
import sys
import time
import requests
import argparse
import pandas as pd
from datetime import datetime

# ANSI Escape Codes for Colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

SCENARIOS = {
    1: {'name': 'Normal station', 'target_type': 'NORMAL'},
    2: {'name': 'Temperature spike', 'target_type': 'SPIKE'},
    3: {'name': 'Frozen temperature sensor', 'target_type': 'FROZEN_SENSOR'},
    4: {'name': 'Temperature drift', 'target_type': 'TEMPORAL_ANOMALY'},
    5: {'name': 'Missing communication', 'target_type': 'MISSING_DATA'},
    6: {'name': 'Humidity stuck at 100%', 'target_type': 'HUMIDITY_SENSOR_FAILURE'},
    7: {'name': 'Pressure anomaly', 'target_type': 'PRESSURE_ANOMALY'},
    8: {'name': 'Multi-sensor failure', 'target_type': 'MULTI_SENSOR_FAILURE'},
    9: {'name': 'Intermittent sensor failure', 'target_type': 'INTERMITTENT_FAILURE'},
    10: {'name': 'Real weather event', 'target_type': 'REAL_WEATHER_EVENT'}
}

def load_scenario_window(scenario_id: int):
    print(f"{Colors.OKCYAN}Loading Dataset...{Colors.ENDC}")
    df = pd.read_csv('data/anomaly/anomaly_injected_aws.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    target_type = SCENARIOS[scenario_id]['target_type']
    
    if target_type == 'NORMAL':
        # Find a chunk of purely normal data for a random station
        normal_df = df[df['is_anomaly'] == 0]
        if normal_df.empty:
            raise ValueError("No normal data found in dataset.")
        station = normal_df['station_id'].iloc[0]
        st_df = df[df['station_id'] == station].sort_values('timestamp').reset_index()
        # Find 4 hours of pure normal
        return st_df.head(24) # 24 obs = 4 hours roughly
        
    else:
        # Find the first anomaly of this type
        target_df = df[df['anomaly_type'] == target_type]
        if target_df.empty:
            raise ValueError(f"No anomaly of type {target_type} found in dataset.")
            
        first_event = target_df.iloc[0]
        station = first_event['station_id']
        event_time = first_event['timestamp']
        
        # Get data for this station
        st_df = df[df['station_id'] == station].sort_values('timestamp')
        
        # Slice 24 hours before, 12 hours after
        start_time = event_time - pd.Timedelta(hours=24)
        end_time = event_time + pd.Timedelta(hours=12)
        
        window_df = st_df[(st_df['timestamp'] >= start_time) & (st_df['timestamp'] <= end_time)]
        return window_df

def run_demo(scenario_id: int, speed: float = 1.0):
    scenario = SCENARIOS[scenario_id]
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== SKYGUARD AI DEMONSTRATION ==={Colors.ENDC}")
    print(f"{Colors.HEADER}Scenario {scenario_id}: {scenario['name']}{Colors.ENDC}")
    print(f"{Colors.HEADER}Target Anomaly Type: {scenario['target_type']}{Colors.ENDC}\n")
    
    try:
        window_df = load_scenario_window(scenario_id)
    except Exception as e:
        print(f"{Colors.FAIL}Error loading scenario: {e}{Colors.ENDC}")
        return
        
    print(f"{Colors.OKGREEN}Extracted {len(window_df)} observations. Replaying at {speed}x speed...{Colors.ENDC}\n")
    
    base_delay = 10 * 60 # 10 real minutes between obs usually
    if speed <= 0:
        actual_delay = 0.5
    else:
        actual_delay = 2.0 / speed # Just a nice viewing speed
        
    for _, row in window_df.iterrows():
        # Formulate payload
        payload = {
            "station_id": row['station_id'],
            "timestamp": row['timestamp'].isoformat(),
            "temperature_c": row.get('temperature_c'),
            "pressure_hpa": row.get('pressure_hpa'),
            "relative_humidity_pct": row.get('relative_humidity_pct'),
            "source": "demo"
        }
        
        is_ground_truth_fault = row.get('is_anomaly', 0) == 1 and row.get('anomaly_type') != 'REAL_WEATHER_EVENT'
        is_real_event = row.get('is_anomaly', 0) == 1 and row.get('anomaly_type') == 'REAL_WEATHER_EVENT'
        
        # Print Input
        time_str = row['timestamp'].strftime('%H:%M:%S')
        input_str = f"[{time_str}] {row['station_id']} | T:{row.get('temperature_c', 'N/A')} P:{row.get('pressure_hpa', 'N/A')} H:{row.get('relative_humidity_pct', 'N/A')}"
        
        if is_ground_truth_fault:
            print(f"{Colors.FAIL}[CONTROLLED FAULT] {input_str}{Colors.ENDC} -> {Colors.WARNING}[GROUND TRUTH: {row.get('anomaly_type')}]{Colors.ENDC}")
        elif is_real_event:
            print(f"{Colors.OKCYAN}[REAL WEATHER EVENT] {input_str}{Colors.ENDC} -> {Colors.WARNING}[GROUND TRUTH: REAL_WEATHER_EVENT]{Colors.ENDC}")
        else:
            print(f"{Colors.OKBLUE}[REAL OBSERVATION] {input_str}{Colors.ENDC}")
            
        # Post to API
        try:
            res = requests.post("http://localhost:8000/api/observations", json=payload)
            if res.status_code == 200:
                data = res.json()
                
                # Fetch recent predictions and alerts for this station to show detection status
                # (Since our post endpoint doesn't return the full async engine results immediately, 
                # we can simulate the detection print by querying the anomaly endpoint)
                
                # We will wait a tiny bit to allow background processing
                time.sleep(0.1)
                
                # Fetch last anomaly for this station
                anom_res = requests.get("http://localhost:8000/api/anomalies?limit=5")
                health_res = requests.get("http://localhost:8000/api/sensor-health")
                
                detected = False
                if anom_res.status_code == 200:
                    anoms = anom_res.json()
                    for a in anoms:
                        # Simple check if it matches the current timestamp (approx)
                        if a['observation_id'] == data.get('id'):
                            detected = True
                            score = a.get('anomaly_score', 0)
                            spatial = a.get('spatial_score', 0)
                            print(f"    {Colors.WARNING}|- [MODEL PREDICTION] DETECTED ANOMALY | IF_Score: {score:.2f} | Spatial: {spatial:.2f}{Colors.ENDC}")
                            break
                            
                if not detected:
                    print(f"    {Colors.OKGREEN}|- [MODEL PREDICTION] Normal{Colors.ENDC}")
                    
                # Print Health if relevant
                if health_res.status_code == 200:
                    healths = health_res.json()
                    for h in healths:
                        if h['station_id'] == row['station_id'] and h['parameter'] == 'temperature_c':
                            if h['health_score'] < 100:
                                col = Colors.WARNING if h['health_score'] > 40 else Colors.FAIL
                                print(f"    {col}|- [SENSOR HEALTH] Temp Health: {h['health_score']:.1f}% | Risk: {h['predicted_failure_risk']:.2f} | {h['health_reason']}{Colors.ENDC}")
                            break
                            
            else:
                print(f"{Colors.FAIL}API Error: {res.status_code}{Colors.ENDC}")
        except requests.exceptions.ConnectionError:
            print(f"{Colors.FAIL}Could not connect to FastAPI. Is it running on port 8000?{Colors.ENDC}")
            return
            
        time.sleep(actual_delay)

    print(f"\n{Colors.OKGREEN}Demo Scenario Complete.{Colors.ENDC}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkyGuard AI Demo Scenarios")
    parser.add_argument("--scenario", type=int, choices=range(1, 11), required=True, help="Scenario number 1-10")
    parser.add_argument("--speed", type=float, default=2.0, help="Replay speed multiplier")
    args = parser.parse_args()
    
    run_demo(args.scenario, args.speed)
