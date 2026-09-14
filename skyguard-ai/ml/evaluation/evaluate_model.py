import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {}
        
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr)
    }

def calculate_latency(df):
    """Calculate detection latency (in rows/hours) for true anomalies."""
    # Filter only actual anomalies
    anomalies = df[df['is_anomaly'] == 1].copy()
    if anomalies.empty:
        return {"mean_latency_rows": None}
        
    # Group by the anomaly event (using anomaly_start and station_id)
    events = anomalies.groupby(['station_id', 'anomaly_start'])
    
    latencies = []
    for _, group in events:
        group = group.sort_values('timestamp')
        # Find first index where prediction is 1
        detections = np.where(group['iforest_prediction'] == 1)[0]
        if len(detections) > 0:
            latencies.append(float(detections[0])) # 0 means detected immediately on first row
            
    if not latencies:
        return {"mean_latency_rows": None, "detected_events_ratio": 0.0}
        
    return {
        "mean_latency_rows": float(np.mean(latencies)),
        "median_latency_rows": float(np.median(latencies)),
        "detected_events_ratio": len(latencies) / len(events)
    }

def evaluate_pipeline(input_path: str, metrics_path: str, cm_path: str):
    print(f"Loading predictions from {input_path}")
    df = pd.read_csv(input_path)
    
    if df.empty:
        print("Empty dataset.")
        return

    # Metrics by split
    metrics = {"splits": {}, "severity": {}, "latency": {}}
    all_cm = []
    
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        if split_df.empty:
            continue
            
        y_true = split_df['is_anomaly']
        y_pred = split_df['iforest_prediction']
        
        split_metrics = calculate_metrics(y_true, y_pred)
        metrics["splits"][split] = split_metrics
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        all_cm.append({"split": split, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
        
    # Overall severity performance on Test set (or Val+Test)
    eval_df = df[df['split'].isin(['val', 'test'])]
    if not eval_df.empty:
        for severity in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
            sev_df = eval_df[(eval_df['anomaly_severity'] == severity) & (eval_df['is_anomaly'] == 1)]
            if not sev_df.empty:
                # Recall for this specific severity
                recall = recall_score(sev_df['is_anomaly'], sev_df['iforest_prediction'], zero_division=0)
                metrics["severity"][severity] = {"recall": float(recall), "support": len(sev_df)}
                
        # Calculate latency
        metrics["latency"] = calculate_latency(eval_df)
        
    # Save metrics
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save Confusion Matrix
    Path(cm_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_cm).to_csv(cm_path, index=False)
    
    print(f"Evaluation complete. Metrics saved to {metrics_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Baseline Model.")
    parser.add_argument('--input', type=str, default='data/processed/isolation_forest_predictions.csv')
    parser.add_argument('--metrics-out', type=str, default='reports/isolation_forest_metrics.json')
    parser.add_argument('--cm-out', type=str, default='reports/confusion_matrix.csv')
    args = parser.parse_args()
    
    evaluate_pipeline(args.input, args.metrics_out, args.cm_out)
