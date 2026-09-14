import pandas as pd
import numpy as np
from typing import Tuple, List

class AnomalyGenerators:
    @staticmethod
    def apply_spike(df: pd.DataFrame, indices: list, param: str, magnitude: float) -> pd.DataFrame:
        df = df.copy()
        df.loc[indices, param] += magnitude * np.random.choice([-1, 1])
        return df

    @staticmethod
    def apply_frozen_sensor(df: pd.DataFrame, indices: list, param: str) -> pd.DataFrame:
        df = df.copy()
        # Freeze at the value just before the interval, or the first value
        if len(indices) > 0:
            val = df.loc[indices[0], param]
            df.loc[indices, param] = val
        return df

    @staticmethod
    def apply_drift(df: pd.DataFrame, indices: list, param: str, drift_rate: float) -> pd.DataFrame:
        df = df.copy()
        drift_values = np.arange(1, len(indices) + 1) * drift_rate * np.random.choice([-1, 1])
        df.loc[indices, param] += drift_values
        return df

    @staticmethod
    def apply_offset(df: pd.DataFrame, indices: list, param: str, offset: float) -> pd.DataFrame:
        df = df.copy()
        df.loc[indices, param] += offset * np.random.choice([-1, 1])
        return df

    @staticmethod
    def apply_missing_data(df: pd.DataFrame, indices: list, param: str) -> pd.DataFrame:
        df = df.copy()
        df.loc[indices, param] = pd.NA
        return df

    @staticmethod
    def apply_communication_error(df: pd.DataFrame, indices: list, params: List[str]) -> pd.DataFrame:
        df = df.copy()
        for p in params:
            df.loc[indices, p] = pd.NA
        return df

    @staticmethod
    def apply_pressure_anomaly(df: pd.DataFrame, indices: list, magnitude: float) -> pd.DataFrame:
        return AnomalyGenerators.apply_spike(df, indices, 'pressure_hpa', magnitude)

    @staticmethod
    def apply_humidity_sensor_failure(df: pd.DataFrame, indices: list, stuck_value: float = 0.0) -> pd.DataFrame:
        df = df.copy()
        df.loc[indices, 'relative_humidity_pct'] = stuck_value
        return df

    @staticmethod
    def apply_temporal_anomaly(df: pd.DataFrame, indices: list, param: str) -> pd.DataFrame:
        df = df.copy()
        # Add random noise mimicking timing jitter
        noise = np.random.normal(0, 5.0, len(indices))
        df.loc[indices, param] += noise
        return df

    @staticmethod
    def apply_multivariate_inconsistency(df: pd.DataFrame, indices: list) -> pd.DataFrame:
        df = df.copy()
        # Artificial inverse relationship violation: drop temp and drop humidity strongly
        df.loc[indices, 'temperature_c'] -= 10.0
        df.loc[indices, 'relative_humidity_pct'] -= 30.0
        return df

    @staticmethod
    def apply_multi_sensor_failure(df: pd.DataFrame, indices: list, params: List[str]) -> pd.DataFrame:
        df = df.copy()
        # Freeze all
        for p in params:
            if len(indices) > 0:
                val = df.loc[indices[0], p]
                df.loc[indices, p] = val
        return df

    @staticmethod
    def apply_intermittent_failure(df: pd.DataFrame, indices: list, param: str, magnitude: float) -> pd.DataFrame:
        df = df.copy()
        # Spike every other row
        for i, idx in enumerate(indices):
            if i % 2 == 0:
                df.loc[idx, param] += magnitude * np.random.choice([-1, 1])
        return df

    @staticmethod
    def apply_real_weather_event(df: pd.DataFrame, indices: list, param: str, magnitude: float) -> pd.DataFrame:
        df = df.copy()
        # Mimic a realistic smooth curve (e.g. sudden front)
        curve = np.sin(np.linspace(0, np.pi, len(indices))) * magnitude * np.random.choice([-1, 1])
        df.loc[indices, param] += curve
        return df
