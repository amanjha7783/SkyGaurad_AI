from typing import Dict, Any, List
import pandas as pd

from ml.features.spatial_features import find_nearest_neighbors, calculate_spatial_deltas
from ml.models.spatial_anomaly import SpatialAnomalyDetector
from ..models import models

class SpatialDetectionService:
    def __init__(self, db_session, radius_km: float = 50.0):
        self.db = db_session
        self.radius_km = radius_km
        self.detector = SpatialAnomalyDetector()
        
        # In-memory caches
        self.station_metadata = {} # dict of station_id -> {lat, lon, etc.}
        self.latest_observations = {} # dict of station_id -> dict of latest values
        self.neighbors_cache = {} # dict of station_id -> list of neighbor station_ids
        
        self._initialize_station_cache()
        
    def _initialize_station_cache(self):
        stations = self.db.query(models.Station).all()
        st_list = []
        for st in stations:
            st_dict = {
                'id': st.id,
                'latitude': st.latitude,
                'longitude': st.longitude
            }
            self.station_metadata[st.id] = st_dict
            st_list.append(st_dict)
            
        # Precompute neighbors
        for st in st_list:
            neighbors = find_nearest_neighbors(
                target_lat=st['latitude'],
                target_lon=st['longitude'],
                stations=st_list,
                radius_km=self.radius_km
            )
            # Filter out self
            neighbors = [n for n in neighbors if n['id'] != st['id']]
            self.neighbors_cache[st['id']] = neighbors

    def evaluate_observation(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes the incoming observation, updates the cache, 
        compares it to neighbors, and returns spatial metrics.
        """
        station_id = obs_dict['station_id']
        
        # Update cache with latest values
        self.latest_observations[station_id] = {
            'temperature_c': obs_dict.get('temperature_c'),
            'pressure_hpa': obs_dict.get('pressure_hpa'),
            'relative_humidity_pct': obs_dict.get('relative_humidity_pct')
        }
        
        neighbors = self.neighbors_cache.get(station_id, [])
        if not neighbors:
            return {
                'spatial_inconsistency_score': 0.0,
                'distance_to_nearest_station_km': None,
                'deltas': {}
            }
            
        nearest_distance = neighbors[0]['distance_km']
        
        # Get latest values for all neighbors
        neighbor_values = []
        for n in neighbors:
            n_id = n['id']
            if n_id in self.latest_observations:
                neighbor_values.append(self.latest_observations[n_id])
                
        # Calculate Deltas
        deltas = calculate_spatial_deltas(
            target_values=self.latest_observations[station_id],
            neighbor_values=neighbor_values
        )
        
        # Detect Anomaly Score
        score = self.detector.calculate_inconsistency_score(deltas)
        
        return {
            'spatial_inconsistency_score': score,
            'distance_to_nearest_station_km': nearest_distance,
            'deltas': deltas
        }
