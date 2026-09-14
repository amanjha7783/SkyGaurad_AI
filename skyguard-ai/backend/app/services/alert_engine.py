from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

from ..models import models

class NotificationProvider:
    """Abstraction for sending notifications via external channels (Email/SMS/Slack)"""
    def send(self, alert: models.Alert) -> bool:
        # Dummy implementation as requested
        print(f"[NOTIFICATION SENT] {alert.severity} Alert for {alert.station_id}: {alert.description}")
        return True

class AlertEngine:
    def __init__(self, db_session):
        self.db = db_session
        self.notification_providers: List[NotificationProvider] = [NotificationProvider()]
        
    def notify(self, alert: models.Alert):
        for provider in self.notification_providers:
            provider.send(alert)
            
    def _create_or_update_alert(
        self,
        station_id: str,
        timestamp: datetime,
        severity: str,
        parameter: str,
        anomaly_type: str,
        anomaly_score: float,
        sensor_health: float,
        description: str,
        recommended_action: str,
        confidence_pct: float = 95.0
    ):
        # Deduplication: Check if there is an OPEN alert for this station & parameter
        existing_alert = self.db.query(models.Alert).filter(
            models.Alert.station_id == station_id,
            models.Alert.parameter == parameter,
            models.Alert.status == "OPEN"
        ).first()
        
        if existing_alert:
            # Update the existing alert
            # If the new severity is higher (e.g. HIGH -> CRITICAL), upgrade it
            severity_rank = {"INFO": 1, "LOW": 2, "MEDIUM": 3, "HIGH": 4, "CRITICAL": 5}
            if severity_rank.get(severity, 0) > severity_rank.get(existing_alert.severity, 0):
                existing_alert.severity = severity
                
            existing_alert.timestamp = timestamp
            existing_alert.anomaly_score = anomaly_score
            existing_alert.sensor_health = sensor_health
            existing_alert.description = description
            existing_alert.recommended_action = recommended_action
            existing_alert.confidence_pct = confidence_pct
            
            # Commit the update
            self.db.add(existing_alert)
            self.notify(existing_alert)
            return existing_alert
            
        else:
            # Create a new alert
            new_alert = models.Alert(
                station_id=station_id,
                timestamp=timestamp,
                severity=severity,
                parameter=parameter,
                anomaly_type=anomaly_type,
                anomaly_score=anomaly_score,
                sensor_health=sensor_health,
                description=description,
                recommended_action=recommended_action,
                confidence_pct=confidence_pct,
                status="OPEN"
            )
            self.db.add(new_alert)
            self.notify(new_alert)
            return new_alert

    def process_anomaly(self, buffer_df: pd.DataFrame, station_id: str, current_ts: datetime, health_res: pd.DataFrame, spatial_score: float = 0.0, spatial_deltas: dict = None):
        """
        Evaluate recent observations to trigger alerts.
        Called per observation by the RealtimeProcessor.
        """
        if len(buffer_df) == 0:
            return
            
        # 0. Spatial Consistency Trigger
        if spatial_score > 0.8:
            temp_delta = spatial_deltas.get('neighbor_temperature_difference', 0.0) if spatial_deltas else 0.0
            
            # Build dynamic explainability text
            current_temp = buffer_df['temperature_c'].iloc[-1]
            neighbor_mean = current_temp - temp_delta
            conf = min(99.9, 80.0 + (spatial_score * 20.0))
            
            desc = (f"WHY WAS THIS FLAGGED? Temperature is {current_temp:.1f}°C, which is a {temp_delta:+.1f}°C difference "
                    f"from the local 50km geographic cluster average of {neighbor_mean:.1f}°C. "
                    f"This extreme spatial disagreement strongly indicates a localized hardware fault rather than a real weather event.")
            
            self._create_or_update_alert(
                station_id=station_id,
                timestamp=current_ts,
                severity="HIGH",
                parameter="temperature_c",
                anomaly_type="SPATIAL_DISAGREEMENT",
                anomaly_score=spatial_score,
                sensor_health=100.0, # Default to 100 for spatial, or could use real health
                description=desc,
                recommended_action="Cross-reference with nearby stations and investigate hardware.",
                confidence_pct=conf
            )
            
        # 1. Anomaly Persistence Trigger (3+ consecutive anomalies on temperature)
        if 'lstm_prediction' in buffer_df.columns:
            recent_preds = buffer_df['lstm_prediction'].tail(3).values
            if len(recent_preds) == 3 and all(p == 1 for p in recent_preds):
                score = float(buffer_df['lstm_reconstruction_error'].iloc[-1])
                health = float(health_res['temperature_health_score'].iloc[-1]) if not health_res.empty else 100.0
                
                conf = min(99.0, 75.0 + score)
                
                # Dynamic text
                t_change = buffer_df['temperature_c'].iloc[-1] - buffer_df['temperature_c'].iloc[-3]
                desc = (f"WHY WAS THIS FLAGGED? Temperature changed by {t_change:+.1f}°C over the last 3 observations, "
                        f"triggering consistent multivariate and sequential anomalies in the deep learning models. "
                        f"The reading is inconsistent with recent temporal behavior.")

                self._create_or_update_alert(
                    station_id=station_id,
                    timestamp=current_ts,
                    severity="HIGH" if score > 8.0 else "MEDIUM",
                    parameter="temperature_c",
                    anomaly_type="PERSISTENT_ANOMALY",
                    anomaly_score=score,
                    sensor_health=health,
                    description=desc,
                    recommended_action="Inspect hardware for calibration drift.",
                    confidence_pct=conf
                )
                
        # 2. Sensor Health Trigger
        if not health_res.empty:
            latest_health = health_res.iloc[-1]
            health_score = float(latest_health['temperature_health_score'])
            if health_score < 40:
                conf = 99.9
                reason = latest_health.get('temperature_health_reason', '')
                desc = (f"WHY WAS THIS FLAGGED? Sensor health has dropped to {health_score:.1f}%. "
                        f"The deterministic health engine observed: {reason}. This level of degradation usually precedes complete failure.")
                
                self._create_or_update_alert(
                    station_id=station_id,
                    timestamp=current_ts,
                    severity="CRITICAL",
                    parameter="temperature_c",
                    anomaly_type="HARDWARE_DEGRADATION",
                    anomaly_score=0.0,
                    sensor_health=health_score,
                    description=desc,
                    recommended_action="Immediate dispatch required for sensor replacement.",
                    confidence_pct=conf
                )
