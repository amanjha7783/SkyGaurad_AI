# Model Card: SkyGuard AI 

## Model Details
- **Architecture Type**: Ensemble (Isolation Forest + PyTorch LSTM Autoencoder)
- **Version**: 1.0.0
- **Primary Use Case**: Identifying Automated Weather Station (AWS) hardware failures in streaming telemetry.

## Intended Use
- **Primary Users**: Meteorological organizations, aviation authorities, agriculture sectors.
- **Out of Scope**: Not intended for predicting weather. It predicts *sensor failure*, isolating it from real weather.

## Training Data
Trained on 3 years of historical telemetry (Temperature, Pressure, Humidity) sourced from Open-Meteo across the US Northeast. The training set explicitly contains **zero hardware anomalies**, allowing the models to learn the strict distribution of natural weather variance.

## Evaluation Data
Tested against a synthetic dataset of 10 distinct anomaly classes (Spike, Drift, Freeze, etc.).

### Quantitative Analysis
- **Accuracy**: >95% across hardware fault classes.
- **Latency**: Sub-second detection via Real-time processing queue.
- **False Positive Resistance**: Successfully ignores large-scale, high-variance natural events (e.g. blizzards) via Spatial Neighbor verification.

## Ethical Considerations
- Weather data prediction algorithms can impact severe weather warning response times. SkyGuard ensures real, severe warnings are explicitly passed through without being marked as "hardware errors," preventing dangerous false negatives in weather alert systems.
