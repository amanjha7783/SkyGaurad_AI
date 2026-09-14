from ml.features.spatial_features import haversine_distance, find_nearest_neighbors, calculate_spatial_deltas
from ml.models.spatial_anomaly import SpatialAnomalyDetector

def test_haversine():
    # NYC to London
    dist = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
    assert 5500 < dist < 5600

def test_find_neighbors():
    stations = [
        {'id': 'A', 'latitude': 40.0, 'longitude': -74.0},
        {'id': 'B', 'latitude': 40.1, 'longitude': -74.0}, # close (~11km)
        {'id': 'C', 'latitude': 41.0, 'longitude': -74.0}, # far (~111km)
    ]
    neighbors = find_nearest_neighbors(40.0, -74.0, stations, radius_km=50.0)
    assert len(neighbors) == 1
    assert neighbors[0]['id'] == 'B'

def test_spatial_deltas():
    target = {'temperature_c': 35.0}
    neighbors = [
        {'temperature_c': 20.0},
        {'temperature_c': 22.0}
    ]
    deltas = calculate_spatial_deltas(target, neighbors)
    assert 'neighbor_temperature_difference' in deltas
    assert deltas['neighbor_temperature_difference'] == 14.0 # 35 - 21

def test_anomaly_score():
    detector = SpatialAnomalyDetector()
    score1 = detector.calculate_inconsistency_score({'neighbor_temperature_difference': 2.0})
    assert score1 == 0.2
    
    score2 = detector.calculate_inconsistency_score({'neighbor_temperature_difference': 15.0})
    assert score2 == 1.0 # Maxed out
