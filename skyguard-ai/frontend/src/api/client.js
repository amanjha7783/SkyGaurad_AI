import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 5000,
});

export const fetchStatistics = async () => {
  const { data } = await apiClient.get('/statistics');
  return data;
};

export const fetchStations = async () => {
  const { data } = await apiClient.get('/stations');
  return data;
};

export const fetchObservations = async (limit = 100) => {
  const { data } = await apiClient.get(`/observations?limit=${limit}`);
  return data;
};

export const fetchAnomalies = async (limit = 100) => {
  const { data } = await apiClient.get(`/anomalies?limit=${limit}`);
  return data;
};

export const fetchSensorHealth = async () => {
  const { data } = await apiClient.get('/sensor-health');
  return data;
};

export const fetchAlerts = async (status = null) => {
  const url = status ? `/alerts?status=${status}` : '/alerts';
  const { data } = await apiClient.get(url);
  return data;
};

export const updateAlertStatus = async (alertId, status) => {
  const { data } = await apiClient.put(`/alerts/${alertId}/status`, { status });
  return data;
};
