import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.numeric_features = [
            'temperature_c', 'pressure_hpa', 'relative_humidity_pct',
            'temperature_rate_change', 'pressure_rate_change', 'humidity_rate_change',
            'distance_to_nearest_station_km', 'neighbor_temperature_difference',
            'neighbor_pressure_difference', 'neighbor_humidity_difference',
            'sensor_health_score'
        ]
        
    def fit(self, train_df: pd.DataFrame):
        # Fit imputer and scaler ONLY on train data
        available_features = [f for f in self.numeric_features if f in train_df.columns]
        self.used_features = available_features
        
        # We need to impute first
        train_data = train_df[self.used_features].copy()
        
        # Fit imputer
        self.imputer.fit(train_data)
        imputed_data = self.imputer.transform(train_data)
        
        # Fit scaler
        self.scaler.fit(imputed_data)
        
        # Save components
        joblib.dump(self.imputer, 'models/saved/imputer.pkl')
        joblib.dump(self.scaler, 'models/saved/scaler.pkl')
        import json
        with open('models/saved/used_features.json', 'w') as f:
            json.dump(self.used_features, f)
        
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # Transform data using fitted components
        data = df[self.used_features].copy()
        
        # Some columns might have all NaNs if not present in test but present in train
        # The imputer handles this if fitted correctly.
        
        imputed_data = self.imputer.transform(data)
        scaled_data = self.scaler.transform(imputed_data)
        return scaled_data

    def load_components(self):
        self.imputer = joblib.load('models/saved/imputer.pkl')
        self.scaler = joblib.load('models/saved/scaler.pkl')
        import json
        with open('models/saved/used_features.json', 'r') as f:
            self.used_features = json.load(f)
