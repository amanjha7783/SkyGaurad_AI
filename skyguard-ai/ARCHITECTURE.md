# SkyGuard AI: Architecture

SkyGuard AI acts as a sophisticated filtering pipeline designed to protect weather monitoring networks from false-alarm alert storms while instantly diagnosing hardware degradation.

## Data Ingestion Layer
Data is accepted via a FastAPI `POST` endpoint simulating a live IoT MQTT stream. The `RealtimeProcessor` interpolates nulls and extracts advanced rolling features (12-hour means, variance).

## Multi-Layered Detection Engine

1. **Isolation Forest (The Vanguard)**
   - **Purpose**: Rapid multivariate outlier detection.
   - **Mechanism**: Trained on purely normal weather data. If pressure drops drastically without a corresponding temperature or humidity shift, it identifies an anomaly in $O(1)$ time.

2. **LSTM Autoencoder (The Deep Sequential Tracker)**
   - **Purpose**: Catching slow drift and complex temporal failures.
   - **Mechanism**: A PyTorch Long Short-Term Memory network compresses a rolling window of 12 observations into a latent space and attempts to reconstruct it. High reconstruction loss indicates the sensor's temporal behavior has fundamentally changed.

3. **Spatial Consistency (The Natural Weather Filter)**
   - **Purpose**: Preventing false alarms during real storms.
   - **Mechanism**: Uses Haversine distance matrices. If a sensor reports a sudden 15°C drop, the engine queries the 50km geographic neighbor cache. If neighbors confirm the drop, it's a storm. If they disagree, it's a hardware fault.

## Intelligent Alert Engine
The alert engine acts as a state machine. It prevents alert storms by deduplicating sequential failures. If a sensor fails 20 times in a row, a single incident is opened, and its severity dynamically scales from `LOW` to `CRITICAL`. The new **Explainability Module** generates dynamic, plain-English justifications for the operator.
