import pandas as pd
import numpy as np

class FeatureProcessor:
    def __init__(self):
        # The exact 11 features expected by the trained ML pipeline
        self.expected_features = [
            'temperature_c', 'pressure_hpa', 'relative_humidity_pct',
            'temperature_rate_change', 'pressure_rate_change', 'humidity_rate_change',
            'distance_to_nearest_station_km', 'neighbor_temperature_difference',
            'neighbor_pressure_difference', 'neighbor_humidity_difference',
            'sensor_health_score'
        ]

    def map_api_to_features(self, api_data: dict) -> pd.DataFrame:
        """
        Maps a single current weather API response to the ML feature schema.
        Missing contextual features (deltas, spatial data) are explicitly set to NaN
        so the ML pipeline's trained SimpleImputer can securely handle them using 
        training distribution medians without fabricating data.
        """
        feature_dict = {
            'temperature_c': api_data.get('temperature_c', np.nan),
            'pressure_hpa': api_data.get('pressure_hpa', np.nan),
            'relative_humidity_pct': api_data.get('relative_humidity_pct', np.nan),
            
            # Explicitly handling features not provided by a single live API point
            'temperature_rate_change': np.nan,
            'pressure_rate_change': np.nan,
            'humidity_rate_change': np.nan,
            'distance_to_nearest_station_km': np.nan,
            'neighbor_temperature_difference': np.nan,
            'neighbor_pressure_difference': np.nan,
            'neighbor_humidity_difference': np.nan,
            'sensor_health_score': 1.0 # Assume healthy sensor as a neutral default
        }
        
        # Ensure exact column ordering as trained
        df = pd.DataFrame([feature_dict])
        df = df[self.expected_features]
        return df
