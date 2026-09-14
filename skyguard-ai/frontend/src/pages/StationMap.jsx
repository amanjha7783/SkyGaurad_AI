import React, { useEffect, useState } from 'react';
import { fetchStations, fetchSensorHealth } from '../api/client';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { ShieldAlert, ShieldCheck } from 'lucide-react';

// Fix Leaflet icon issue in React
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom colored icons
const createCustomIcon = (color) => {
  return L.divIcon({
    className: 'custom-div-icon',
    html: `<div style="background-color: ${color}; width: 16px; height: 16px; border-radius: 50%; border: 3px solid #fff; box-shadow: 0 0 10px ${color};"></div>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11]
  });
};

const iconHealthy = createCustomIcon('#00e676');
const iconWarning = createCustomIcon('#ffea00');
const iconCritical = createCustomIcon('#ff1744');

const StationMap = () => {
  const [stations, setStations] = useState([]);
  const [healthData, setHealthData] = useState([]);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [st, hd] = await Promise.all([fetchStations(), fetchSensorHealth()]);
        setStations(st);
        setHealthData(hd);
      } catch (err) {
        console.error(err);
      }
    };
    loadData();
    const interval = setInterval(loadData, 10000);
    return () => clearInterval(interval);
  }, []);

  const getStationHealth = (stationId) => {
    const h = healthData.find(d => d.station_id === stationId);
    return h ? h.health_score : 100;
  };

  const getIcon = (score) => {
    if (score > 80) return iconHealthy;
    if (score > 40) return iconWarning;
    return iconCritical;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', height: '100%' }}>
      <div>
        <h1 style={{ fontSize: '1.75rem', marginBottom: '8px' }}>Station Geolocation Map</h1>
        <p style={{ color: 'var(--text-muted)' }}>Live regional tracking and physical sensor status</p>
      </div>

      <div className="glass-panel" style={{ flex: 1, padding: '8px', overflow: 'hidden', minHeight: '600px' }}>
        <MapContainer 
          center={[39.8283, -98.5795]} 
          zoom={4} 
          style={{ height: '100%', width: '100%', borderRadius: '8px' }}
        >
          {/* Dark CartoDB Matter tiles */}
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          />
          
          {stations.map(station => {
            const score = getStationHealth(station.id);
            return (
              <Marker 
                key={station.id} 
                position={[station.latitude, station.longitude]}
                icon={getIcon(score)}
              >
                <Popup>
                  <div style={{ padding: '4px' }}>
                    <h4 style={{ color: '#fff', marginBottom: '4px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '4px' }}>{station.name}</h4>
                    <p style={{ margin: '4px 0', fontSize: '13px' }}>ID: {station.id}</p>
                    <p style={{ margin: '4px 0', fontSize: '13px' }}>Location: {station.city}, {station.state}</p>
                    
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '12px' }}>
                      {score > 80 ? <ShieldCheck size={16} color="#00e676" /> : <ShieldAlert size={16} color={score > 40 ? "#ffea00" : "#ff1744"} />}
                      <span style={{ fontSize: '14px', fontWeight: 600, color: score > 80 ? "#00e676" : (score > 40 ? "#ffea00" : "#ff1744") }}>
                        Health: {score}%
                      </span>
                    </div>
                  </div>
                </Popup>
              </Marker>
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
};

export default StationMap;
