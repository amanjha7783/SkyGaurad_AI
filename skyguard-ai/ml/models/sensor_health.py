import pandas as pd
import numpy as np
from typing import Dict, Any

class SensorHealthEngine:
    def __init__(self, window_hours=6):
        self.window_hours = window_hours
        # Thresholds
        self.frozen_threshold = 3  # hours
        
    def evaluate_parameter_health(self, df_station: pd.DataFrame, param_name: str) -> pd.DataFrame:
        """
        Evaluates health for a specific parameter (e.g. 'temperature') over time for a single station.
        df_station must be sorted chronologically.
        """
        # Ensure index is time for rolling operations
        df = df_station.copy()
        df = df.set_index('timestamp')
        
        health_scores = []
        reasons = []
        
        # Columns needed: lstm_prediction, lstm_reconstruction_error, {param}_persistence, {param}_c_is_missing
        # Note: missing col is e.g. temperature_c_is_missing, but parameter might just be 'temperature'
        # We need to map param_name to the actual columns
        col_prefix = param_name.split('_')[0] # 'temperature' from 'temperature_c'
        
        missing_col = f'{param_name}_is_missing'
        pers_col = f'{col_prefix}_persistence'
        pred_col = 'lstm_prediction'
        err_col = 'lstm_reconstruction_error'
        
        # We use a rolling window to gather recent evidence
        # Rolling sum of missing values
        if missing_col in df.columns:
            recent_missing = df[missing_col].rolling(f'{self.window_hours}h').sum()
        else:
            recent_missing = pd.Series(0, index=df.index)
            
        # Rolling sum of anomalies
        if pred_col in df.columns:
            recent_anomalies = df[pred_col].rolling(f'{self.window_hours}h').sum()
            avg_err = df[err_col].rolling(f'{self.window_hours}h').mean()
        else:
            recent_anomalies = pd.Series(0, index=df.index)
            avg_err = pd.Series(0, index=df.index)
            
        # Persistence is already consecutive counts, just take current value
        if pers_col in df.columns:
            current_pers = df[pers_col]
        else:
            current_pers = pd.Series(0, index=df.index)

        # Iterate through timestamps to calculate score
        for ts in df.index:
            score = 100
            reason_list = []
            
            # Evidence: Missing data
            m_count = recent_missing.loc[ts]
            if m_count > 0:
                deduction = min(30, m_count * 10)
                score -= deduction
                reason_list.append(f"Missing data ({int(m_count)} times in {self.window_hours}h)")
                
            # Evidence: Frozen sensor (persistence)
            p_count = current_pers.loc[ts]
            if p_count >= self.frozen_threshold:
                score -= 40
                reason_list.append(f"Frozen sensor behavior detected ({int(p_count)} consecutive identical readings)")
                
            # Evidence: Repeated anomalies
            a_count = recent_anomalies.loc[ts]
            if a_count > 0:
                # Severity based on average reconstruction error (if extremely high, penalize more)
                # Baseline penalty 15 per anomaly
                penalty = a_count * 15
                err_val = avg_err.loc[ts]
                if err_val > 5.0: # arbitrary high error threshold indicating severe anomaly
                    penalty += 20
                    reason_list.append(f"Severe anomaly detected (Error: {err_val:.1f})")
                
                score -= penalty
                reason_list.append(f"Recent anomalies ({int(a_count)} in {self.window_hours}h)")
                
            # Bound score
            score = max(0, min(100, int(score)))
            
            if score == 100:
                reason_list.append("Sensor is healthy.")
                
            health_scores.append(score)
            reasons.append(" | ".join(reason_list))
            
        result = pd.DataFrame(index=df.index)
        result[f'{col_prefix}_health_score'] = health_scores
        result[f'{col_prefix}_health_reason'] = reasons
        result[f'{col_prefix}_maintenance_required'] = (result[f'{col_prefix}_health_score'] <= 50).astype(int)
        result[f'{col_prefix}_predicted_failure_risk'] = (100 - result[f'{col_prefix}_health_score']) / 100.0
        
        return result.reset_index()

    def process_all_stations(self, df: pd.DataFrame, parameters: list) -> pd.DataFrame:
        """Processes health scores for all stations and all parameters."""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        all_results = []
        
        for station_id, group in df.groupby('station_id'):
            group = group.sort_values('timestamp')
            
            station_res = group[['timestamp', 'station_id']].copy()
            for param in parameters:
                param_res = self.evaluate_parameter_health(group, param)
                # Merge back
                station_res = pd.merge(station_res, param_res, on='timestamp', how='left')
                
            all_results.append(station_res)
            
        if not all_results:
            return pd.DataFrame()
            
        return pd.concat(all_results, ignore_index=True)
