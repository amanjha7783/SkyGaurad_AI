import pandas as pd
import time
import requests
from datetime import datetime

class ReplaySimulator:
    def __init__(self, data_path: str, api_url: str = "http://localhost:8000/api/observations"):
        self.data_path = data_path
        self.api_url = api_url
        
    def run(self, speed_factor: float = 1.0):
        print(f"Loading dataset for replay from {self.data_path}")
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Group by timestamp so we send all stations for a given hour at once
        grouped = df.groupby('timestamp')
        timestamps = list(grouped.groups.keys())
        
        print(f"Starting replay simulator at {speed_factor}x speed...")
        print(f"Total time steps: {len(timestamps)}")
        
        for i in range(len(timestamps)):
            ts = timestamps[i]
            group_df = grouped.get_group(ts)
            
            # Send observations
            for _, row in group_df.iterrows():
                payload = {
                    "station_id": row['station_id'],
                    "timestamp": row['timestamp'].isoformat(),
                    "temperature_c": row.get('temperature_c'),
                    "pressure_hpa": row.get('pressure_hpa'),
                    "relative_humidity_pct": row.get('relative_humidity_pct'),
                    "source": "simulated_replay"
                }
                
                try:
                    resp = requests.post(self.api_url, json=payload, timeout=2)
                    if resp.status_code != 200:
                        print(f"Failed to post observation: {resp.text}")
                except Exception as e:
                    print(f"Connection error: {e}")
                    
            if i < len(timestamps) - 1:
                next_ts = timestamps[i + 1]
                # Calculate real-world delta in seconds
                delta_sec = (next_ts - ts).total_seconds()
                
                # Wait based on speed factor
                wait_time = delta_sec / speed_factor
                
                print(f"[{ts}] Sent {len(group_df)} observations. Waiting {wait_time:.2f}s for next batch...")
                time.sleep(wait_time)
                
        print("Replay complete.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--speed', type=float, default=100.0, help='Replay speed multiplier (e.g. 10, 100)')
    parser.add_argument('--input', type=str, default='data/anomaly/anomaly_injected_aws.csv')
    args = parser.parse_args()
    
    simulator = ReplaySimulator(args.input)
    simulator.run(speed_factor=args.speed)
