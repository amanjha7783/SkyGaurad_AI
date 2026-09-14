# Data Card: SkyGuard AI

## Base Dataset (Real)
- **Source**: Open-Meteo Historical Weather API.
- **Geography**: 5 clustered stations around the US Northeast (NYC, Boston, Philadelphia, etc.).
- **Timeframe**: Jan 1, 2020 - Dec 31, 2023.
- **Variables**: Temperature (°C), Pressure (hPa), Relative Humidity (%).

## Anomaly Injection Framework (Synthetic)
To evaluate the ML pipeline without waiting years for hardware to fail naturally, we built a deterministic injection engine (`scripts/ingest_data.py`).

**Injected Fault Classes:**
- `SPIKE`: Sudden single-observation massive variance.
- `FROZEN_SENSOR`: Constant exact value for extended periods.
- `TEMPORAL_ANOMALY` (Drift): Slow continuous skewing of calibration.
- `MULTI_SENSOR_FAILURE`: Total station blackout across all parameters.
- `MISSING_DATA`: Network dropouts.
- `HUMIDITY_SENSOR_FAILURE`: Value stuck precisely at 100%.

**Important Disclaimer for Judges:** 
All base weather variance (diurnal cycles, storms, seasonal shifts) in this project is 100% REAL. The anomalies triggering the system are strictly controlled, synthesized faults designed to test the model's resilience. No false claims of "real hardware failure data" are made.
