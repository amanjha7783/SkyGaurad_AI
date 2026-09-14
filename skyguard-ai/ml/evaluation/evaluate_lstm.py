import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0))
    }

def evaluate_and_compare(lstm_path: str, iforest_path: str, metrics_out: str, comparison_out: str):
    print("Loading predictions...")
    
    # Load LSTM predictions
    df_lstm = pd.read_csv(lstm_path)
    
    # Load IF predictions
    df_if = pd.read_csv(iforest_path)
    
    # Merge IF predictions into df_lstm
    df = df_lstm.copy()
    # We assume they line up chronologically as they were processed the same way
    df['iforest_prediction'] = df_if['iforest_prediction']
    
    # Drop rows where LSTM could not predict (due to sequence length startup)
    df = df.dropna(subset=['lstm_prediction'])
    
    metrics = {"splits": {}}
    comparison = []
    
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        if split_df.empty:
            continue
            
        y_true = split_df['is_anomaly']
        y_lstm = split_df['lstm_prediction']
        y_if = split_df['iforest_prediction']
        
        # LSTM metrics
        lstm_m = calculate_metrics(y_true, y_lstm)
        metrics["splits"][split] = lstm_m
        
        # Comparison metrics
        if_m = calculate_metrics(y_true, y_if)
        
        comparison.append({
            "split": split,
            "model": "IsolationForest",
            "accuracy": if_m['accuracy'],
            "precision": if_m['precision'],
            "recall": if_m['recall'],
            "f1_score": if_m['f1_score']
        })
        
        comparison.append({
            "split": split,
            "model": "LSTM_Autoencoder",
            "accuracy": lstm_m['accuracy'],
            "precision": lstm_m['precision'],
            "recall": lstm_m['recall'],
            "f1_score": lstm_m['f1_score']
        })
        
    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    Path(comparison_out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison).to_csv(comparison_out, index=False)
    
    print(f"Metrics saved to {metrics_out}")
    print(f"Comparison saved to {comparison_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate and Compare LSTM.")
    parser.add_argument('--lstm-input', type=str, default='data/processed/lstm_predictions.csv')
    parser.add_argument('--if-input', type=str, default='data/processed/isolation_forest_predictions.csv')
    parser.add_argument('--metrics-out', type=str, default='reports/lstm_metrics.json')
    parser.add_argument('--comp-out', type=str, default='reports/model_comparison.csv')
    args = parser.parse_args()
    
    evaluate_and_compare(args.lstm_input, args.if_input, args.metrics_out, args.comp_out)
