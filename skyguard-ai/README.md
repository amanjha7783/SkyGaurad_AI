<div align="center">
  <img src="https://img.icons8.com/color/96/000000/artificial-intelligence.png" alt="SkyGuard AI Logo"/>
  <h1>SkyGuard AI 🌩️</h1>
  <p><b>Intelligent Meteorological Hardware Anomaly Detection</b></p>
  
  <p>
    <a href="#architecture">Architecture</a> • 
    <a href="#installation">Installation</a> • 
    <a href="#demo">Demo</a>
  </p>
</div>

## Pitch
Automated Weather Stations (AWS) dictate climate models, aviation safety, and severe weather warnings. However, sensors fail—they drift, freeze, and short-circuit. Traditional static thresholds flag real, severe weather (like a heatwave) as hardware failures, causing alert fatigue. 

**SkyGuard AI** combines lightweight Isolation Forests with PyTorch LSTM Autoencoders and geographic spatial consistency to definitively isolate hardware faults from real meteorological events in real-time.

---

## 🧠 ML Credibility & Explainability
Judges: This is not a black box. SkyGuard AI features an **Explainability Engine** that dynamically generates plain-English rationale for every alert, mapping the exact temperature deltas and Haversine spatial averages that triggered the detection. 

See our comprehensive data and model cards:
- [MODEL_CARD.md](MODEL_CARD.md): PyTorch LSTM & Scikit-Learn Isolation Forest architecture.
- [DATA_CARD.md](DATA_CARD.md): Open-Meteo baseline with synthetic fault injection framework.
- [ARCHITECTURE.md](ARCHITECTURE.md): The multi-layered detection pipeline.

---

## 🚀 Quick Start / Installation

**Prerequisites:** Python 3.10+, Node 18+

### 1. Launch the Backend
```bash
cd backend
python -m venv venv
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

### 2. Launch the Frontend
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```

### 3. Run the Demo Injection
In a third terminal (with the backend venv activated):
```bash
python scripts/demo.py --scenario 8 --speed 3.0
```

Navigate to `http://localhost:5173/alerts` to watch the control center react in real-time!

---

## 📁 Repository Structure
- `backend/`: FastAPI server, SQLite db, ML inference, and Alert Engine.
- `frontend/`: React + Vite glassmorphic dashboard.
- `ml/`: Model training scripts, feature engineering, spatial distance matrix cache.
- `data/`: CSV datasets and synthetic injection scripts.
- `scripts/`: System evaluation and demo replay utilities.


## Official Dataset & Training Methodology

### Dataset Lineage
The official dataset files used for model training and evaluation are sourced exactly from the provided workspace and integrated into the data/raw/ directory:
- 	rain.csv (1339 rows)
- alidation.csv (297 rows)
- 	est.csv (286 rows)

**Target Label:** is_anomaly`n**Feature Columns:** temperature_c, pressure_hpa, relative_humidity_pct, and various temporal/spatial deltas.

### Preprocessing & Leakage Prevention
Data leakage is strictly prevented by:
1. Fitting StandardScaler and SimpleImputer exclusively on 	rain.csv.
2. Saving these stateful transformers to models/saved/.
3. Transforming alidation.csv and 	est.csv using the pre-fitted components.

### Models & Evaluation
We employ two independent anomaly detectors to form a robust ensemble:
- **Isolation Forest**: Evaluates local numerical spikes and multidimensional outliers.
- **Autoencoder (MLPRegressor Bottleneck)**: Learns the identity function of normal weather data and detects anomalies based on high reconstruction error.

**Commands:**
- Train Models: python src/training/train.py`n- Evaluate Models: python src/evaluation/evaluate.py`n
**Evaluation Outputs:**
Outputs are safely generated in the esults/ directory without mutating the raw CSV files:
- esults/metrics/*.json: Accuracy, Precision, Recall, F1, Anomaly Rate.
- esults/predictions/*.csv: Raw boolean inference predictions.
- esults/plots/confusion_matrix/: Heatmaps detailing FP/TN metrics.

*Note: Because the official 	est.csv contains exclusively negative samples (normal weather), the Precision and Recall metrics for the Anomaly class yield 0.0, which accurately reflects the lack of true positive anomalies available for detection in the test split.*

