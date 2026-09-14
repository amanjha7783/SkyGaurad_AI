import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from ml.models.sensor_health import SensorHealthEngine

class SensorHealthService:
    def __init__(self, window_hours=6):
        self.engine = SensorHealthEngine(window_hours=window_hours)
        
    def generate_health_report(self, predictions_path: str, output_path: str):
        print(f"Loading predictions from {predictions_path}...")
        df = pd.read_csv(predictions_path)
        
        if df.empty:
            print("Empty dataset. Cannot generate health report.")
            return
            
        parameters = ['temperature_c', 'pressure_hpa', 'relative_humidity_pct']
        
        print("Calculating sensor health scores across all stations...")
        health_df = self.engine.process_all_stations(df, parameters)
        
        # Merge back with original dataset to have a complete picture if needed,
        # but the prompt asks to output `sensor_health.csv` which should contain the scores.
        # We will save just the health metrics with timestamps and station IDs to keep it clean.
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        health_df.to_csv(output_path, index=False)
        print(f"Sensor health data saved to {output_path}")
        return health_df

if __name__ == '__main__':
    service = SensorHealthService()
    # In production, we would use lstm_predictions.csv
    # If the user doesn't have lstm_predictions, we fall back to isolation_forest_predictions
    lstm_path = 'data/processed/lstm_predictions.csv'
    iforest_path = 'data/processed/isolation_forest_predictions.csv'
    
    input_path = lstm_path if Path(lstm_path).exists() else iforest_path
    output_path = 'data/processed/sensor_health.csv'
    
    service.generate_health_report(input_path, output_path)
