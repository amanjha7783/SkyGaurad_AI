from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from ..core.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class Station(Base):
    __tablename__ = "stations"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    city = Column(String)
    state = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    altitude_m = Column(Float)
    
    observations = relationship("Observation", back_populates="station", cascade="all, delete-orphan")
    sensor_health = relationship("SensorHealth", back_populates="station", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="station", cascade="all, delete-orphan")


class Observation(Base):
    __tablename__ = "observations"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    station_id = Column(String, ForeignKey("stations.id"))
    timestamp = Column(DateTime, index=True)
    
    temperature_c = Column(Float, nullable=True)
    pressure_hpa = Column(Float, nullable=True)
    relative_humidity_pct = Column(Float, nullable=True)
    source = Column(String)
    
    station = relationship("Station", back_populates="observations")
    anomaly_predictions = relationship("AnomalyPrediction", back_populates="observation", cascade="all, delete-orphan")


class AnomalyPrediction(Base):
    __tablename__ = "anomaly_predictions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    observation_id = Column(String, ForeignKey("observations.id"))
    
    model_name = Column(String) # 'isolation_forest', 'lstm'
    is_anomaly = Column(Integer)
    anomaly_score = Column(Float, nullable=True)
    spatial_score = Column(Float, nullable=True)
    confidence_pct = Column(Float, nullable=True)
    
    observation = relationship("Observation", back_populates="anomaly_predictions")


class SensorHealth(Base):
    __tablename__ = "sensor_health"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    station_id = Column(String, ForeignKey("stations.id"))
    timestamp = Column(DateTime, index=True)
    
    parameter = Column(String) # 'temperature_c', 'pressure_hpa'
    health_score = Column(Float)
    maintenance_required = Column(Integer)
    predicted_failure_risk = Column(Float)
    health_reason = Column(String)
    
    station = relationship("Station", back_populates="sensor_health")


class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    station_id = Column(String, ForeignKey("stations.id"))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    severity = Column(String) # 'INFO', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    parameter = Column(String, nullable=True)
    anomaly_type = Column(String, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    sensor_health = Column(Float, nullable=True)
    confidence_pct = Column(Float, nullable=True)
    description = Column(String)
    recommended_action = Column(String, nullable=True)
    status = Column(String, default="OPEN") # 'OPEN', 'ACKNOWLEDGED', 'RESOLVED'
    
    station = relationship("Station", back_populates="alerts")
