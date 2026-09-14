from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
from datetime import datetime

class HealthCheck(BaseModel):
    status: str
    version: str

class StationBase(BaseModel):
    name: str
    city: str
    state: str
    latitude: float
    longitude: float
    altitude_m: float

class StationCreate(StationBase):
    id: str

class StationResponse(StationBase):
    id: str
    model_config = ConfigDict(from_attributes=True)

class ObservationBase(BaseModel):
    station_id: str
    timestamp: datetime
    temperature_c: Optional[float] = None
    pressure_hpa: Optional[float] = None
    relative_humidity_pct: Optional[float] = None
    source: str = "real"

class ObservationCreate(ObservationBase):
    pass

class ObservationResponse(ObservationBase):
    id: str
    model_config = ConfigDict(from_attributes=True)

class AnomalyPredictionResponse(BaseModel):
    id: str
    observation_id: str
    model_name: str
    is_anomaly: int
    anomaly_score: Optional[float] = None
    spatial_score: Optional[float] = None
    confidence_pct: Optional[float] = None
    model_config = ConfigDict(from_attributes=True)

class SensorHealthResponse(BaseModel):
    id: str
    station_id: str
    timestamp: datetime
    parameter: str
    health_score: float
    maintenance_required: int
    predicted_failure_risk: float
    health_reason: str
    model_config = ConfigDict(from_attributes=True)

class PaginatedObservations(BaseModel):
    total: int
    page: int
    size: int
    items: List[ObservationResponse]

class StatisticsResponse(BaseModel):
    total_stations: int
    total_observations: int
    total_anomalies: int
    active_alerts: int
    
class AlertResponse(BaseModel):
    id: str
    station_id: str
    timestamp: datetime
    severity: str
    parameter: Optional[str] = None
    anomaly_type: Optional[str] = None
    anomaly_score: Optional[float] = None
    sensor_health: Optional[float] = None
    confidence_pct: Optional[float] = None
    description: str
    recommended_action: Optional[str] = None
    status: str
    model_config = ConfigDict(from_attributes=True)
    
class AlertStatusUpdate(BaseModel):
    status: str
    
class PredictRequest(BaseModel):
    model: str = Field(default="lstm", description="Model to use: 'lstm' or 'isolation_forest'")
    temperature_c: float
    pressure_hpa: float
    relative_humidity_pct: float

class PredictResponse(BaseModel):
    is_anomaly: int
    anomaly_score: float
    model_used: str

class ModelStatusResponse(BaseModel):
    model_type: str
    threshold: Optional[float] = None
    metrics: dict
