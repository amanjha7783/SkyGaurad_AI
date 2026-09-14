import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.data_loader import load_data
from src.data.preprocessing import DataPreprocessor

def evaluate_model(model, X, y_true, model_name, split_name):
    # IF returns 1 (normal) and -1 (anomaly). We map to 0 (normal) and 1 (anomaly)
    preds = model.predict(X)
    y_pred = np.where(preds == -1, 1, 0)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'anomaly_detection_rate': float(np.mean(y_pred == 1)),
        'detected_anomalies': int(np.sum(y_pred == 1))
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    metrics['confusion_matrix'] = {
        'TN': int(cm[0][0]),
        'FP': int(cm[0][1]),
        'FN': int(cm[1][0]),
        'TP': int(cm[1][1])
    }
    
    if len(np.unique(y_true)) > 1:
        # Decision function for AUC
        scores = model.decision_function(X) # Higher score -> more normal
        # ROC AUC needs probability of positive class
        metrics['roc_auc'] = float(roc_auc_score(y_true, -scores))
        
    return metrics, y_pred

def plot_cm(cm_dict, model_name, split_name):
    cm = np.array([[cm_dict['TN'], cm_dict['FP']], [cm_dict['FN'], cm_dict['TP']]])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name} ({split_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'results/plots/confusion_matrix/{model_name}_{split_name}_cm.png')
    plt.close()

def main():
    print("Loading data for evaluation...")
    train_df, val_df, test_df = load_data()
    
    preprocessor = DataPreprocessor()
    preprocessor.load_components()
    
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)
    
    y_val = val_df['is_anomaly'].values if 'is_anomaly' in val_df.columns else np.zeros(len(val_df))
    y_test = test_df['is_anomaly'].values if 'is_anomaly' in test_df.columns else np.zeros(len(test_df))
    
    # Evaluate Isolation Forest
    if_model = joblib.load('models/saved/isolation_forest.pkl')
    val_metrics, val_preds = evaluate_model(if_model, X_val, y_val, 'isolation_forest', 'val')
    test_metrics, test_preds = evaluate_model(if_model, X_test, y_test, 'isolation_forest', 'test')
    
    with open('results/metrics/isolation_forest_metrics.json', 'w') as f:
        json.dump({'validation': val_metrics, 'test': test_metrics}, f, indent=4)
        
    test_df_if = test_df.copy()
    test_df_if['is_anomaly_pred'] = test_preds
    test_df_if.to_csv('results/predictions/isolation_forest_predictions.csv', index=False)
    plot_cm(test_metrics['confusion_matrix'], 'isolation_forest', 'test')
    
    # Evaluate Autoencoder
    try:
        ae_model = joblib.load('models/saved/autoencoder.pkl')
        with open('models/saved/ae_threshold.json', 'r') as f:
            threshold = json.load(f)['threshold']
            
        test_pred = ae_model.predict(X_test)
        test_mse = np.mean(np.power(X_test - test_pred, 2), axis=1)
        
        ae_preds = (test_mse > threshold).astype(int)
        
        ae_metrics = {
            'accuracy': float(accuracy_score(y_test, ae_preds)),
            'precision': float(precision_score(y_test, ae_preds, zero_division=0)),
            'recall': float(recall_score(y_test, ae_preds, zero_division=0)),
            'f1_score': float(f1_score(y_test, ae_preds, zero_division=0)),
            'anomaly_detection_rate': float(np.mean(ae_preds == 1)),
            'detected_anomalies': int(np.sum(ae_preds == 1))
        }
        cm = confusion_matrix(y_test, ae_preds, labels=[0,1])
        ae_metrics['confusion_matrix'] = {
            'TN': int(cm[0][0]), 'FP': int(cm[0][1]), 'FN': int(cm[1][0]), 'TP': int(cm[1][1])
        }
        
        with open('results/metrics/autoencoder_metrics.json', 'w') as f:
            json.dump({'test': ae_metrics}, f, indent=4)
            
        test_df_ae = test_df.copy()
        test_df_ae['is_anomaly_pred'] = ae_preds
        test_df_ae.to_csv('results/predictions/autoencoder_predictions.csv', index=False)
        plot_cm(ae_metrics['confusion_matrix'], 'autoencoder', 'test')
        
    except Exception as e:
        print(f"Skipping Autoencoder evaluation: {e}")
    
    print(f"Evaluation complete. IF Test metrics: {json.dumps(test_metrics, indent=4)}")

if __name__ == "__main__":
    main()
