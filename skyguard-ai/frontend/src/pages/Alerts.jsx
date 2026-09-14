import React, { useEffect, useState } from 'react';
import { fetchAlerts, updateAlertStatus } from '../api/client';
import { ShieldAlert, CheckCircle, Clock } from 'lucide-react';

const Alerts = () => {
  const [alerts, setAlerts] = useState([]);
  const [filter, setFilter] = useState('OPEN'); // OPEN, ACKNOWLEDGED, RESOLVED, ALL

  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await fetchAlerts(filter === 'ALL' ? null : filter);
        setAlerts(data);
      } catch (err) {
        console.error(err);
      }
    };
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, [filter]);

  const handleStatusChange = async (id, newStatus) => {
    try {
      await updateAlertStatus(id, newStatus);
      // Optimistically update UI
      setAlerts(alerts.map(a => a.id === id ? { ...a, status: newStatus } : a).filter(a => filter === 'ALL' || a.status === filter));
    } catch (err) {
      console.error("Failed to update status", err);
    }
  };

  const getSeverityBadge = (severity) => {
    switch (severity) {
      case 'CRITICAL': return <span className="badge badge-critical">CRITICAL</span>;
      case 'HIGH': return <span className="badge badge-anomaly">HIGH</span>;
      case 'MEDIUM': return <span className="badge badge-warning">MEDIUM</span>;
      case 'LOW': return <span className="badge" style={{ background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.3)', color: '#fff' }}>LOW</span>;
      default: return <span className="badge" style={{ background: 'rgba(0,102,255,0.15)', border: '1px solid rgba(0,102,255,0.3)', color: '#0066ff' }}>INFO</span>;
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 style={{ fontSize: '1.75rem', marginBottom: '8px' }}>Incident Alert Center</h1>
          <p style={{ color: 'var(--text-muted)' }}>Deduplicated real-time notifications from the Intelligence Engine</p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          {['OPEN', 'ACKNOWLEDGED', 'RESOLVED', 'ALL'].map(f => (
            <button 
              key={f}
              onClick={() => setFilter(f)}
              style={{
                background: filter === f ? 'rgba(0,240,255,0.2)' : 'transparent',
                border: filter === f ? '1px solid var(--accent-cyan)' : '1px solid var(--border-subtle)',
                color: filter === f ? '#fff' : 'var(--text-muted)',
                padding: '6px 12px',
                borderRadius: '6px',
                cursor: 'pointer'
              }}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {alerts.length === 0 ? (
          <div className="glass-panel" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
            No {filter !== 'ALL' ? filter.toLowerCase() : ''} alerts found.
          </div>
        ) : (
          alerts.map(alert => (
            <div key={alert.id} className="glass-panel" style={{ padding: '24px', borderLeft: alert.severity === 'CRITICAL' ? '4px solid var(--status-critical)' : '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                    {getSeverityBadge(alert.severity)}
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>{new Date(alert.timestamp).toLocaleString()}</span>
                    <span style={{ background: 'rgba(255,255,255,0.1)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.8rem' }}>Station: {alert.station_id}</span>
                  </div>
                  <h3 style={{ fontSize: '1.2rem', marginBottom: '4px' }}>{alert.anomaly_type.replace('_', ' ')}</h3>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>{alert.description}</p>
                </div>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', minWidth: '150px' }}>
                  {alert.status === 'OPEN' && (
                    <>
                      <button onClick={() => handleStatusChange(alert.id, 'ACKNOWLEDGED')} style={{ background: 'rgba(255,234,0,0.15)', border: '1px solid rgba(255,234,0,0.3)', color: 'var(--status-warning)', padding: '8px 12px', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                        <Clock size={16} /> Acknowledge
                      </button>
                      <button onClick={() => handleStatusChange(alert.id, 'RESOLVED')} style={{ background: 'rgba(0,230,118,0.15)', border: '1px solid rgba(0,230,118,0.3)', color: 'var(--status-healthy)', padding: '8px 12px', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                        <CheckCircle size={16} /> Resolve
                      </button>
                    </>
                  )}
                  {alert.status === 'ACKNOWLEDGED' && (
                    <button onClick={() => handleStatusChange(alert.id, 'RESOLVED')} style={{ background: 'rgba(0,230,118,0.15)', border: '1px solid rgba(0,230,118,0.3)', color: 'var(--status-healthy)', padding: '8px 12px', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                      <CheckCircle size={16} /> Resolve
                    </button>
                  )}
                  {alert.status === 'RESOLVED' && (
                    <div style={{ padding: '8px 12px', textAlign: 'center', color: 'var(--status-healthy)', background: 'rgba(0,230,118,0.05)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                      <CheckCircle size={16} /> Resolved
                    </div>
                  )}
                </div>
              </div>
              
              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '16px', borderRadius: '8px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>Diagnostic Data</p>
                  <p style={{ fontSize: '0.9rem' }}>Parameter: <span style={{ color: '#fff' }}>{alert.parameter}</span></p>
                  {alert.anomaly_score !== null && <p style={{ fontSize: '0.9rem' }}>Anomaly Score: <span style={{ color: '#fff' }}>{(alert.anomaly_score).toFixed(2)}</span></p>}
                  {alert.sensor_health !== null && <p style={{ fontSize: '0.9rem' }}>Sensor Health: <span style={{ color: '#fff' }}>{alert.sensor_health.toFixed(1)}%</span></p>}
                  {alert.confidence_pct !== null && <p style={{ fontSize: '0.9rem' }}>ML Confidence: <span style={{ color: 'var(--status-healthy)' }}>{alert.confidence_pct.toFixed(1)}%</span></p>}
                </div>
                <div>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>Recommended Action</p>
                  <p style={{ fontSize: '0.9rem', color: 'var(--accent-cyan)' }}>{alert.recommended_action || "Manual investigation required."}</p>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default Alerts;
