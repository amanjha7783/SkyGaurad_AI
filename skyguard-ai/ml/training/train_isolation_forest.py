import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ml.models.isolation_forest_model import BaselineAnomalyDetector

def train_pipeline(input_path: str, model_path: str, predictions_path: str, importance_path: str):
    print(f"Loading features from {input_path}")
    df = pd.read_csv(input_path)
    
    if df.empty:
        print("Empty dataset.")
        return

    # Ensure chronological order for strict split
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Columns to exclude from training (metadata + ground truth + leakage)
    exclude_cols = [
        'timestamp', 'station_id', 'station_name', 'city', 'state', 'source',
        'is_anomaly', 'anomaly_type', 'anomaly_subtype', 'anomaly_severity',
        'anomaly_start', 'anomaly_end', 'fault_parameter', 'fault_source', 'ground_truth_reason'
    ]

    features = [c for c in df.columns if c not in exclude_cols]
    
    # Calculate splits
    n = len(df)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)

    df['split'] = 'test'
    df.loc[:train_idx-1, 'split'] = 'train'
    df.loc[train_idx:val_idx-1, 'split'] = 'val'

    X_train = df[df['split'] == 'train'][features]

    print(f"Training Baseline Isolation Forest on {len(X_train)} samples with {len(features)} features...")
    
    # Estimate contamination based on actual training ground truth to give the baseline a fair shot
    # Since it's unsupervised, we just pass this as a hyperparameter
    contamination = max(0.01, min(0.5, df.loc[:train_idx-1, 'is_anomaly'].mean()))
    
    model = BaselineAnomalyDetector(random_state=42, contamination=contamination)
    model.fit(X_train)

    # Save model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Generate predictions across ALL splits
    print("Generating predictions...")
    df['iforest_prediction'] = model.predict(df[features])
    
    # Save predictions
    Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    # Calculate feature importance
    print("Calculating permutation feature importance...")
    # Use a subset of training data for speed
    X_sample = X_train.sample(n=min(5000, len(X_train)), random_state=42)
    importances = model.get_feature_importances(X_sample, features)
    
    Path(importance_path).parent.mkdir(parents=True, exist_ok=True)
    with open(importance_path, 'w') as f:
        json.dump(importances, f, indent=4)
    print(f"Feature importances saved to {importance_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Baseline Isolation Forest.")
    parser.add_argument('--input', type=str, default='data/processed/features.csv')
    parser.add_argument('--model-out', type=str, default='ml/models/saved/isolation_forest.joblib')
    parser.add_argument('--predictions-out', type=str, default='data/processed/isolation_forest_predictions.csv')
    parser.add_argument('--importance-out', type=str, default='reports/feature_importance.json')
    args = parser.parse_args()
    
    train_pipeline(args.input, args.model_out, args.predictions_out, args.importance_out)
