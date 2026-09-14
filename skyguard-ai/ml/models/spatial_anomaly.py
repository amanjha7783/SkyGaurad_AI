from typing import Dict, Any

class SpatialAnomalyDetector:
    def __init__(self):
        # Thresholds for standard deviation/differences where we start penalizing
        # e.g., if a station is 10 degrees C different from local neighbors, it's highly suspect
        self.temp_threshold_high = 10.0 # deg C
        self.press_threshold_high = 15.0 # hPa
        self.hum_threshold_high = 30.0 # %

    def calculate_inconsistency_score(self, deltas: Dict[str, float]) -> float:
        """
        Takes in spatial deltas and outputs a score 0.0 to 1.0.
        1.0 means extremely inconsistent with neighbors (likely local sensor fault).
        """
        if not deltas:
            return 0.0
            
        score = 0.0
        
        # Temp Score
        if 'neighbor_temperature_difference' in deltas:
            t_diff = abs(deltas['neighbor_temperature_difference'])
            t_score = min(t_diff / self.temp_threshold_high, 1.0)
            score = max(score, t_score)
            
        # Pressure Score
        if 'neighbor_pressure_difference' in deltas:
            p_diff = abs(deltas['neighbor_pressure_difference'])
            p_score = min(p_diff / self.press_threshold_high, 1.0)
            score = max(score, p_score)
            
        # Humidity Score
        if 'neighbor_relative_humidity_difference' in deltas:
            h_diff = abs(deltas['neighbor_relative_humidity_difference'])
            h_score = min(h_diff / self.hum_threshold_high, 1.0)
            score = max(score, h_score)
            
        return float(score)
