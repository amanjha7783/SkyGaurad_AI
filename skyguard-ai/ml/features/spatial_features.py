import math
from typing import Dict, List, Tuple

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def find_nearest_neighbors(
    target_lat: float, 
    target_lon: float, 
    stations: List[Dict], 
    radius_km: float = 50.0
) -> List[Dict]:
    """
    Find stations within the specified radius.
    stations is a list of dicts with 'id', 'latitude', 'longitude'.
    """
    neighbors = []
    for st in stations:
        dist = haversine_distance(target_lat, target_lon, st['latitude'], st['longitude'])
        if dist > 0 and dist <= radius_km:
            st_copy = st.copy()
            st_copy['distance_km'] = dist
            neighbors.append(st_copy)
            
    # Sort by distance
    neighbors.sort(key=lambda x: x['distance_km'])
    return neighbors

def calculate_spatial_deltas(target_values: Dict[str, float], neighbor_values: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculates the difference between the target station's readings and the mean of its neighbors.
    Returns e.g. neighbor_temperature_difference
    """
    if not neighbor_values:
        return {}
        
    deltas = {}
    for param in ['temperature_c', 'pressure_hpa', 'relative_humidity_pct']:
        if param in target_values and target_values[param] is not None:
            # Extract valid neighbor values for this parameter
            valid_vals = [nv[param] for nv in neighbor_values if param in nv and nv[param] is not None]
            if valid_vals:
                neighbor_mean = sum(valid_vals) / len(valid_vals)
                deltas[f'neighbor_{param.replace("_c", "").replace("_hpa", "").replace("_pct", "")}_difference'] = target_values[param] - neighbor_mean
                
    return deltas
