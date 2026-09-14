import React, { useEffect, useState } from 'react';
import { fetchSensorHealth } from '../api/client';
import { Cpu, Wrench, CheckCircle, AlertTriangle } from 'lucide-react';

const SensorHealth = () => {
  const [healthData, setHealthData] = useState([]);

  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await fetchSensorHealth();
        setHealthData(data);
      } catch (err) {
        console.error(err);
      }
    };
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <div>
        <h1 style={{ fontSize: '1.75rem', marginBottom: '8px' }}>Hardware Diagnostics</h1>
        <p style={{ color: 'var(--text-muted)' }}>Persistent ML evaluation of physical sensor integrity</p>
      </div>

      <div className="grid-cols-2">
        {healthData.map((health) => {
          const isCritical = health.health_score < 40;
          const isWarning = health.health_score >= 40 && health.health_score < 80;
          
          return (
            <div key={`${health.station_id}-${health.parameter}`} className={`glass-panel ${isCritical ? 'animate-pulse-critical' : ''}`} style={{ padding: '24px', border: isCritical ? '1px solid rgba(255, 23, 68, 0.5)' : '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{ padding: '10px', background: 'rgba(255,255,255,0.05)', borderRadius: '10px' }}>
                    <Cpu size={24} color={isCritical ? "var(--status-critical)" : (isWarning ? "var(--status-warning)" : "var(--status-healthy)")} />
                  </div>
                  <div>
                    <h3 style={{ fontSize: '1.1rem' }}>{health.station_id}</h3>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', textTransform: 'capitalize' }}>{health.parameter.replace('_', ' ')} Sensor</p>
                  </div>
                </div>
                <div>
                  <span style={{ fontSize: '1.5rem', fontWeight: 700, color: isCritical ? "var(--status-critical)" : (isWarning ? "var(--status-warning)" : "var(--status-healthy)") }}>
                    {health.health_score}%
                  </span>
                </div>
              </div>
              
              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '16px', borderRadius: '8px', marginBottom: '16px' }}>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '8px' }}>Diagnostic Reason:</p>
                <p style={{ fontSize: '0.95rem' }}>{health.health_reason || "Operating optimally with no detected drift or faults."}</p>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--border-subtle)', paddingTop: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {health.maintenance_required === 1 ? (
                    <><Tool size={16} color="var(--status-critical)" /><span style={{ fontSize: '0.9rem', color: 'var(--status-critical)', fontWeight: 500 }}>Maintenance Required</span></>
                  ) : (
                    <><CheckCircle size={16} color="var(--status-healthy)" /><span style={{ fontSize: '0.9rem', color: 'var(--status-healthy)' }}>Nominal</span></>
                  )}
                </div>
                <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                  Failure Risk: <span style={{ color: '#fff', fontWeight: 600 }}>{(health.predicted_failure_risk * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      
      {healthData.length === 0 && (
        <div className="glass-panel" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
          No health data populated yet. Run the real-time simulation engine to generate diagnostics.
        </div>
      )}
    </div>
  );
};

export default SensorHealth;
