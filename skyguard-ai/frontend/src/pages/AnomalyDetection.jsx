import React, { useEffect, useState } from 'react';
import { fetchAnomalies } from '../api/client';
import { ShieldAlert, Zap } from 'lucide-react';

const AnomalyDetection = () => {
  const [anomalies, setAnomalies] = useState([]);

  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await fetchAnomalies(100);
        setAnomalies(data);
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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 style={{ fontSize: '1.75rem', marginBottom: '8px' }}>Anomaly Log</h1>
          <p style={{ color: 'var(--text-muted)' }}>Event signatures detected by Isolation Forest and LSTM</p>
        </div>
        <div style={{ textAlign: 'right' }}>
          <p style={{ color: 'var(--status-anomaly)', fontSize: '2rem', fontWeight: 700, margin: 0 }}>{anomalies.length}</p>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Recent Faults</p>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
          <div style={{ padding: '8px', background: 'rgba(255, 61, 0, 0.15)', borderRadius: '8px' }}>
            <Zap size={20} color="var(--status-anomaly)" />
          </div>
          <h3>Detection Stream</h3>
        </div>
        
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                <th style={{ padding: '12px' }}>Event ID</th>
                <th style={{ padding: '12px' }}>Model Engine</th>
                <th style={{ padding: '12px' }}>Confidence Score</th>
                <th style={{ padding: '12px' }}>Severity</th>
                <th style={{ padding: '12px' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {anomalies.length === 0 ? (
                <tr>
                  <td colSpan="5" style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    No recent anomalies detected. System operating nominally.
                  </td>
                </tr>
              ) : (
                anomalies.map(anom => (
                  <tr key={anom.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                    <td style={{ padding: '12px', fontFamily: 'monospace', fontSize: '0.9rem' }}>{anom.id.split('-')[0]}</td>
                    <td style={{ padding: '12px' }}>
                      <span style={{ padding: '4px 8px', background: 'rgba(0,102,255,0.2)', borderRadius: '4px', fontSize: '0.8rem' }}>
                        {anom.model_name}
                      </span>
                    </td>
                    <td style={{ padding: '12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div style={{ width: '100px', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${Math.min(100, anom.anomaly_score * 100)}%`, height: '100%', background: 'var(--status-anomaly)' }}></div>
                        </div>
                        <span>{(anom.anomaly_score * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td style={{ padding: '12px' }}>
                      {anom.anomaly_score > 0.8 ? (
                        <span className="badge badge-critical">CRITICAL</span>
                      ) : (
                        <span className="badge badge-warning">HIGH</span>
                      )}
                    </td>
                    <td style={{ padding: '12px' }}>
                      <button style={{ background: 'transparent', border: '1px solid var(--border-subtle)', color: '#fff', padding: '4px 12px', borderRadius: '4px', cursor: 'pointer', fontSize: '0.8rem' }}>
                        Investigate
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default AnomalyDetection;
