import React, { useEffect, useState } from 'react';
import { fetchObservations, fetchAnomalies } from '../api/client';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceDot } from 'recharts';

const LiveMonitoring = () => {
  const [data, setData] = useState([]);
  const [anomalies, setAnomalies] = useState([]);

  useEffect(() => {
    const loadData = async () => {
      try {
        const obs = await fetchObservations(50);
        const anoms = await fetchAnomalies(50);
        
        // Reverse because backend sends descending, charts need ascending
        setData(obs.items.reverse());
        setAnomalies(anoms.reverse());
      } catch (err) {
        console.error("Failed to fetch live data", err);
      }
    };
    
    loadData();
    const interval = setInterval(loadData, 3000);
    return () => clearInterval(interval);
  }, []);

  // Format data for chart
  const chartData = data.map(d => {
    const timeStr = new Date(d.timestamp).toLocaleTimeString();
    const isAnomaly = anomalies.find(a => a.observation_id === d.id);
    return {
      name: timeStr,
      temp: d.temperature_c,
      isAnomaly: !!isAnomaly
    };
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <div>
        <h1 style={{ fontSize: '1.75rem', marginBottom: '8px' }}>Live Monitoring</h1>
        <p style={{ color: 'var(--text-muted)' }}>Streaming telemetry sequence</p>
      </div>

      <div className="glass-panel" style={{ padding: '24px', height: '400px' }}>
        <h3 style={{ marginBottom: '24px' }}>Temperature Stream (°C)</h3>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
            <XAxis 
              dataKey="name" 
              stroke="var(--text-muted)" 
              tick={{ fill: 'var(--text-muted)', fontSize: 12 }} 
              tickMargin={10}
            />
            <YAxis 
              stroke="var(--text-muted)" 
              tick={{ fill: 'var(--text-muted)', fontSize: 12 }} 
              domain={['auto', 'auto']}
            />
            <Tooltip 
              contentStyle={{ background: 'var(--bg-panel)', border: '1px solid var(--border-subtle)', borderRadius: '8px', color: '#fff' }}
              itemStyle={{ color: '#00f0ff' }}
            />
            <Line 
              type="monotone" 
              dataKey="temp" 
              stroke="var(--accent-cyan)" 
              strokeWidth={2} 
              dot={false}
              activeDot={{ r: 8, fill: 'var(--accent-cyan)', stroke: '#fff' }}
            />
            {chartData.map((entry, index) => {
              if (entry.isAnomaly) {
                return (
                  <ReferenceDot 
                    key={`anomaly-${index}`} 
                    x={entry.name} 
                    y={entry.temp} 
                    r={6} 
                    fill="var(--status-anomaly)" 
                    stroke="none" 
                  />
                );
              }
              return null;
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ marginBottom: '16px' }}>Recent Observations Log</h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                <th style={{ padding: '12px' }}>Timestamp</th>
                <th style={{ padding: '12px' }}>Station</th>
                <th style={{ padding: '12px' }}>Temp (°C)</th>
                <th style={{ padding: '12px' }}>Pressure (hPa)</th>
                <th style={{ padding: '12px' }}>Humidity (%)</th>
                <th style={{ padding: '12px' }}>Spatial Score</th>
                <th style={{ padding: '12px' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {/* Reverse back to show newest first in table */}
              {[...data].reverse().slice(0, 10).map(obs => {
                const isAnom = anomalies.find(a => a.observation_id === obs.id);
                return (
                  <tr key={obs.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                    <td style={{ padding: '12px', fontFamily: 'monospace' }}>{new Date(obs.timestamp).toLocaleString()}</td>
                    <td style={{ padding: '12px' }}>{obs.station_id}</td>
                    <td style={{ padding: '12px' }}>{obs.temperature_c?.toFixed(1) || 'N/A'}</td>
                    <td style={{ padding: '12px' }}>{obs.pressure_hpa?.toFixed(1) || 'N/A'}</td>
                    <td style={{ padding: '12px' }}>{obs.relative_humidity_pct?.toFixed(1) || 'N/A'}</td>
                    <td style={{ padding: '12px' }}>
                      {isAnom && isAnom.spatial_score !== null ? (
                        <span style={{ color: isAnom.spatial_score > 0.8 ? 'var(--status-critical)' : 'var(--text-muted)' }}>
                          {(isAnom.spatial_score * 100).toFixed(0)}%
                        </span>
                      ) : '-'}
                    </td>
                    <td style={{ padding: '12px' }}>
                      {isAnom ? (
                        <span className="badge badge-anomaly">ANOMALY</span>
                      ) : (
                        <span className="badge badge-healthy">NORMAL</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default LiveMonitoring;
