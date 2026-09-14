import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import joblib
import json
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ml.models.lstm_autoencoder import LSTMAnomalyDetector

def create_sequences(X, y, seq_len):
    """
    Creates sequences of length seq_len.
    Returns X_seq (N, seq_len, features) and y_seq (N).
    The label for the sequence is the label of the LAST time step.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len - 1])
    return np.array(X_seq), np.array(y_seq)

def generate_station_sequences(df, features, seq_len):
    """Generates sequences per station to avoid crossing station boundaries."""
    X_all, y_all, indices_all = [], [], []
    for station_id, group in df.groupby('station_id'):
        group = group.sort_values('timestamp')
        if len(group) < seq_len:
            continue
        
        # Get raw numpy arrays
        X_val = group[features].values
        y_val = group['is_anomaly'].values
        idx_val = group.index.values
        
        # Create sequences
        X_s, y_s = create_sequences(X_val, y_val, seq_len)
        
        # The indices correspond to the LAST step of the sequence
        idx_s = idx_val[seq_len - 1:]
        
        X_all.append(X_s)
        y_all.append(y_s)
        indices_all.append(idx_s)
        
    if len(X_all) == 0:
        return np.array([]), np.array([]), np.array([])
        
    return np.vstack(X_all), np.concatenate(y_all), np.concatenate(indices_all)

def train_lstm(input_path: str, model_dir: str, predictions_path: str):
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    exclude_cols = [
        'timestamp', 'station_id', 'station_name', 'city', 'state', 'source',
        'is_anomaly', 'anomaly_type', 'anomaly_subtype', 'anomaly_severity',
        'anomaly_start', 'anomaly_end', 'fault_parameter', 'fault_source', 'ground_truth_reason'
    ]
    features = [c for c in df.columns if c not in exclude_cols]
    
    # Chronological Split
    n = len(df)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)

    df['split'] = 'test'
    df.loc[:train_idx-1, 'split'] = 'train'
    df.loc[train_idx:val_idx-1, 'split'] = 'val'

    # Impute missing values with median from training set to satisfy scaler/pytorch
    for col in features:
        train_median = df.loc[:train_idx-1, col].median()
        # Fallback to 0 if all NaNs
        if pd.isna(train_median): train_median = 0
        df[col] = df[col].fillna(train_median)

    # Scaling
    scaler = StandardScaler()
    df.loc[:train_idx-1, features] = scaler.fit_transform(df.loc[:train_idx-1, features])
    df.loc[train_idx:val_idx-1, features] = scaler.transform(df.loc[train_idx:val_idx-1, features])
    df.loc[val_idx:, features] = scaler.transform(df.loc[val_idx:, features])

    # Configuration
    seq_len = 6
    batch_size = 64
    epochs = 50
    lr = 0.001
    
    torch.manual_seed(42)
    np.random.seed(42)

    print("Generating sequences...")
    X_seq, y_seq, idx_seq = generate_station_sequences(df, features, seq_len)
    
    # Get split information from the corresponding indices
    splits = df.loc[idx_seq, 'split'].values
    
    train_mask = (splits == 'train') & (y_seq == 0) # Train ONLY on normal data
    val_mask = (splits == 'val')
    test_mask = (splits == 'test')
    
    X_train = torch.tensor(X_seq[train_mask], dtype=torch.float32)
    X_val = torch.tensor(X_seq[val_mask], dtype=torch.float32)
    X_test = torch.tensor(X_seq[test_mask], dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False)
    
    detector = LSTMAnomalyDetector(input_dim=len(features), hidden_dim=32, num_layers=2, lr=lr)
    
    print("Training LSTM Autoencoder...")
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    model_path = Path(model_dir) / 'lstm_autoencoder.pth'
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        train_loss = detector.train_step(train_loader)
        val_loss = detector.evaluate(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            detector.save(str(model_path))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
    # Load best model
    detector.load(str(model_path))
    
    # Generate scores for Validation set to find threshold
    print("Finding optimal threshold on Validation set...")
    val_errors, _ = detector.calculate_reconstruction_error(X_val)
    y_val = y_seq[val_mask]
    
    best_threshold = 0
    best_f1 = -1
    
    # Search threshold grid based on percentiles of validation error
    percentiles = np.linspace(80, 99.9, 50)
    for p in percentiles:
        threshold = np.percentile(val_errors, p)
        preds = (val_errors > threshold).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    print(f"Optimal Threshold: {best_threshold:.4f} (Val F1: {best_f1:.4f})")
    
    # Save metadata
    meta = {
        'threshold': float(best_threshold),
        'seq_len': seq_len,
        'features': features
    }
    with open(Path(model_dir) / 'lstm_metadata.json', 'w') as f:
        json.dump(meta, f)
    joblib.dump(scaler, Path(model_dir) / 'lstm_scaler.joblib')

    # Generate predictions for all generated sequences
    print("Generating predictions...")
    all_errors, _ = detector.calculate_reconstruction_error(torch.tensor(X_seq, dtype=torch.float32))
    all_preds = (all_errors > best_threshold).astype(int)
    
    # Map back to dataframe
    # Rows that were dropped (due to seq_len) will be NaN
    df['lstm_reconstruction_error'] = np.nan
    df['lstm_prediction'] = np.nan
    
    df.loc[idx_seq, 'lstm_reconstruction_error'] = all_errors
    df.loc[idx_seq, 'lstm_prediction'] = all_preds
    
    Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/processed/features.csv')
    parser.add_argument('--model-dir', type=str, default='ml/models/saved')
    parser.add_argument('--predictions-out', type=str, default='data/processed/lstm_predictions.csv')
    args = parser.parse_args()
    
    train_lstm(args.input, args.model_dir, args.predictions_out)
