import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { Activity, Map, LayoutDashboard, AlertTriangle, ShieldCheck, Database, Zap } from 'lucide-react';
import './index.css';

// We will import pages as we build them. For now, placeholders.
import Overview from './pages/Overview';
import LiveMonitoring from './pages/LiveMonitoring';
import StationMap from './pages/StationMap';
import AnomalyDetection from './pages/AnomalyDetection';
import SensorHealth from './pages/SensorHealth';
import Alerts from './pages/Alerts';

const Sidebar = () => {
  const navItems = [
    { name: 'Overview', path: '/', icon: LayoutDashboard },
    { name: 'Live Monitoring', path: '/monitoring', icon: Activity },
    { name: 'Station Map', path: '/map', icon: Map },
    { name: 'Anomaly Detection', path: '/anomalies', icon: Zap },
    { name: 'Sensor Health', path: '/health', icon: ShieldCheck },
    { name: 'Historical Data', path: '/history', icon: Database },
    { name: 'Alerts', path: '/alerts', icon: AlertTriangle }
  ];

  return (
    <div className="glass-panel" style={{ width: '260px', height: '100%', border: 'none', borderRight: '1px solid rgba(255,255,255,0.08)', borderRadius: '0', display: 'flex', flexDirection: 'column', padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '40px' }}>
        <div style={{ width: '32px', height: '32px', background: 'linear-gradient(135deg, #00f0ff, #0066ff)', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 10px rgba(0,240,255,0.4)' }}>
          <ShieldCheck size={20} color="#fff" />
        </div>
        <h2 style={{ fontSize: '1.25rem', letterSpacing: '0.05em' }} className="gradient-text">SKYGUARD <span style={{ fontWeight: 300, color: '#fff' }}>AI</span></h2>
      </div>
      
      <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px', flex: 1 }}>
        {navItems.map((item) => (
          <NavLink 
            key={item.name} 
            to={item.path} 
            style={({ isActive }) => ({
              display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 16px', borderRadius: '8px',
              textDecoration: 'none', color: isActive ? '#fff' : 'var(--text-muted)',
              background: isActive ? 'rgba(0, 102, 255, 0.2)' : 'transparent',
              borderLeft: isActive ? '3px solid var(--accent-cyan)' : '3px solid transparent',
              transition: 'all 0.2s',
              fontWeight: isActive ? 600 : 400
            })}
          >
            <item.icon size={18} style={{ opacity: 0.8 }} />
            <span>{item.name}</span>
          </NavLink>
        ))}
      </nav>
      
      <div style={{ marginTop: 'auto', padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
        <p>System Status: <span style={{ color: 'var(--status-healthy)' }}>● Online</span></p>
        <p style={{ marginTop: '4px' }}>Latency: 12ms</p>
      </div>
    </div>
  );
};

const Topbar = () => (
  <header style={{ height: '70px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
    <div>
      <h3 style={{ fontSize: '1.1rem', fontWeight: 500, color: 'var(--text-main)' }}>Monitoring Console</h3>
      <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Demo Mode: Base weather data is real. Sensor anomalies are synthetically injected.</p>
    </div>
    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
      <button style={{ background: 'var(--bg-panel)', border: '1px solid var(--border-subtle)', color: '#fff', padding: '8px 16px', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Zap size={14} color="var(--accent-cyan)" />
        Run Demo Replay
      </button>
      <div style={{ width: '36px', height: '36px', borderRadius: '50%', background: 'linear-gradient(135deg, #444, #222)', border: '1px solid #555' }}></div>
    </div>
  </header>
);

const App = () => {
  return (
    <Router>
      <div className="app-container">
        <Sidebar />
        <div className="main-content" style={{ padding: 0 }}>
          <Topbar />
          <div style={{ padding: '24px', flex: 1, overflowY: 'auto' }}>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/monitoring" element={<LiveMonitoring />} />
              <Route path="/map" element={<StationMap />} />
              <Route path="/anomalies" element={<AnomalyDetection />} />
              <Route path="/health" element={<SensorHealth />} />
              <Route path="/alerts" element={<Alerts />} />
              {/* Fallbacks */}
              <Route path="*" element={<div style={{ padding: '40px', textAlign: 'center' }}><h2 className="gradient-text">Page Under Construction</h2></div>} />
            </Routes>
          </div>
        </div>
      </div>
    </Router>
  );
};

export default App;
