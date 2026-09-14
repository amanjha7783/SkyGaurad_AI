import React, { useEffect, useState } from 'react';
import { Activity, ShieldAlert, Wifi, HardDrive, Cpu, AlertTriangle } from 'lucide-react';
import { fetchStatistics, fetchSensorHealth } from '../api/client';

const StatCard = ({ title, value, icon: Icon, color, subtitle }) => (
  <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
      <div>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '8px' }}>{title}</p>
        <h2 style={{ fontSize: '2rem', margin: 0, fontWeight: 700 }}>{value}</h2>
      </div>
      <div style={{ 
        padding: '12px', 
        borderRadius: '12px', 
        background: `rgba(${color}, 0.15)`,
        border: `1px solid rgba(${color}, 0.3)`
      }}>
        <Icon color={`rgb(${color})`} size={24} />
      </div>
    </div>
    {subtitle && <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{subtitle}</p>}
  </div>
);

const Overview = () => {
  const [stats, setStats] = useState(null);
  const [healthData, setHealthData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [statData, hData] = await Promise.all([
          fetchStatistics(),
          fetchSensorHealth()
        ]);
        setStats(statData);
        setHealthData(hData);
      } catch (err) {
        console.error("Failed to load overview data", err);
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <div style={{ padding: '40px' }} className="gradient-text">Initializing Telemetry...</div>;
  }

  // Derived metrics
  const avgHealth = healthData.length > 0 
    ? Math.round(healthData.reduce((acc, curr) => acc + curr.health_score, 0) / healthData.length)
    : 100;
    
  const criticalSensors = healthData.filter(h => h.health_score < 40).length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 style={{ fontSize: '1.75rem', marginBottom: '8px' }}>Global Overview</h1>
          <p style={{ color: 'var(--text-muted)' }}>Real-time telemetry and network status</p>
        </div>
        <div className="badge badge-healthy">
          Network Sync Active
        </div>
      </div>

      <div className="grid-cols-3">
        <StatCard 
          title="Online Stations" 
          value={stats?.total_stations || 0} 
          icon={Wifi} 
          color="0, 230, 118" 
          subtitle="All nodes reporting"
        />
        <StatCard 
          title="Observations Ingested" 
          value={(stats?.total_observations || 0).toLocaleString()} 
          icon={HardDrive} 
          color="0, 102, 255" 
          subtitle="Lifetime dataset volume"
        />
        <StatCard 
          title="Active Anomalies" 
          value={stats?.total_anomalies || 0} 
          icon={Activity} 
          color="255, 61, 0" 
          subtitle="Flagged by IsolationForest/LSTM"
        />
        <StatCard 
          title="Average Sensor Health" 
          value={`${avgHealth}%`} 
          icon={Cpu} 
          color={avgHealth > 80 ? "0, 230, 118" : (avgHealth > 50 ? "255, 234, 0" : "255, 23, 68")} 
          subtitle="Fleet-wide physical status"
        />
        <StatCard 
          title="Critical Sensors" 
          value={criticalSensors} 
          icon={AlertTriangle} 
          color="255, 23, 68" 
          subtitle="Requires immediate maintenance"
        />
        <StatCard 
          title="Open Alerts" 
          value={stats?.active_alerts || 0} 
          icon={ShieldAlert} 
          color="255, 234, 0" 
          subtitle="Unresolved system warnings"
        />
      </div>
      
      <div className="grid-cols-2">
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ marginBottom: '16px' }}>System Diagnostics</h3>
          <p style={{ color: 'var(--text-muted)' }}>Isolation Forest model online. LSTM Autoencoder ready.</p>
        </div>
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ marginBottom: '16px' }}>Network Activity</h3>
          <p style={{ color: 'var(--text-muted)' }}>Receiving standard telemetry broadcasts on port 8000.</p>
        </div>
      </div>
    </div>
  );
};

export default Overview;
