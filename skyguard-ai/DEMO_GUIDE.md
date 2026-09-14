# SkyGuard AI: Demonstration Scenarios

To showcase the capabilities of SkyGuard AI's detection algorithms, we provide a dedicated Demo Runner (`scripts/demo.py`). This script extracts perfectly framed 4-hour historical windows from our synthetic ground-truth dataset and streams them into the live FastAPI backend.

## Prerequisites
1. Ensure the **FastAPI backend** is actively running (`uvicorn app.main:app --port 8000`).
2. Ensure the **React Frontend** is actively running (`npm run dev`) and open in your browser.

## Running a Scenario
Open a terminal in the root directory (ensure your virtual environment is activated) and run:

```bash
python scripts/demo.py --scenario <ID> --speed <SPEED>
```

- `--scenario`: A number from 1 to 10.
- `--speed`: Playback speed (default is 2.0). Setting this to 5.0 will rapidly play through the 4-hour window in under 30 seconds.

## Available Scenarios

1. **Normal station**: Proves the system remains quiet and healthy during standard meteorological variance.
2. **Temperature spike**: Simulates an extreme, sudden outlier (e.g. electrical short).
3. **Frozen temperature sensor**: Simulates a sensor reporting the exact same 15.1°C for hours despite diurnal changes.
4. **Temperature drift**: Simulates a slow, insidious failure where temperature climbs artificially by 0.5°C per hour.
5. **Missing communication**: Simulates network drops (Null values).
6. **Humidity stuck at 100%**: Severe sensor saturation fault.
7. **Pressure anomaly**: Drastic pressure jumps.
8. **Multi-sensor failure**: Simulates a catastrophic station failure where all three parameters break simultaneously.
9. **Intermittent sensor failure**: Rapid oscillation between normal and completely broken states.
10. **Real weather event**: Simulates a legitimate, severe regional storm. *The system should NOT flag this as a sensor fault, proving spatial and multivariate intelligence.*

## Terminal Output Legend

When the script runs, it provides a color-coded log:
- **[REAL OBSERVATION]**: Standard historical weather data.
- **[CONTROLLED FAULT]**: The exact moment our script injects a hardware failure.
- **[GROUND TRUTH]**: The true label for the current tick.
- **[MODEL PREDICTION]**: How SkyGuard AI reacted (Anomaly Score, Spatial Score, Sensor Health).

Watch the React Dashboard's **Live Monitoring** and **Alerts** pages immediately populate as the script executes!
