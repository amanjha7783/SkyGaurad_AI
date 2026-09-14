import os
import sys
import joblib
import json

# Add project root to path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.anomaly_detection.weather_api import WeatherAPIClient
from src.anomaly_detection.feature_processor import FeatureProcessor
from src.data.preprocessing import DataPreprocessor

class RealTimeDetector:
    def __init__(self):
        self.api_client = WeatherAPIClient()
        self.feature_processor = FeatureProcessor()
        self.preprocessor = DataPreprocessor()
        
        # Load trained preprocessing artifacts
        self.preprocessor.imputer = joblib.load('models/saved/imputer.pkl')
        self.preprocessor.scaler = joblib.load('models/saved/scaler.pkl')
        
        # Load trained ML model
        self.model = joblib.load('models/saved/isolation_forest.pkl')

    def detect_current_weather(self, lat: float, lon: float) -> dict:
        print(f"Fetching current weather for coordinates ({lat}, {lon})...")
        try:
            api_data = self.api_client.get_current_weather(lat, lon)
            print(f"Received API Data: {api_data}")
        except Exception as e:
            print(f"API Error: {e}")
            return {"error": str(e)}

        print("Mapping API data to ML features...")
        raw_features_df = self.feature_processor.map_api_to_features(api_data)
        
        print("Applying training-fitted preprocessing (imputation & scaling)...")
        # We only use transform() - we never fit() on live data!
        processed_features = self.preprocessor.transform(raw_features_df)
        
        print("Running Isolation Forest anomaly detection...")
        prediction = self.model.predict(processed_features)
        
        is_anomaly = bool(prediction[0] == -1)
        
        result = {
            "timestamp": api_data.get("timestamp"),
            "temperature_c": api_data.get("temperature_c"),
            "pressure_hpa": api_data.get("pressure_hpa"),
            "relative_humidity_pct": api_data.get("relative_humidity_pct"),
            "is_anomaly": is_anomaly,
            "status": "Anomalous" if is_anomaly else "Normal"
        }
        
        print(f"Detection Result: {json.dumps(result, indent=2)}")
        return result

if __name__ == "__main__":
    detector = RealTimeDetector()
    lat = float(os.getenv("LATITUDE", 40.7128))
    lon = float(os.getenv("LONGITUDE", -74.0060))
    detector.detect_current_weather(lat, lon)
