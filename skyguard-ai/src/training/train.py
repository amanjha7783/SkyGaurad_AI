import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.data_loader import load_data
from src.data.preprocessing import DataPreprocessor

def train_isolation_forest(X_train, X_val):
    print("Training Isolation Forest...")
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X_train)
    joblib.dump(model, 'models/saved/isolation_forest.pkl')
    return model

def create_sequences(X, seq_length=12):
    xs = []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
    return np.array(xs)

def train_autoencoder(X_train, X_val):
    print("Training MLP Autoencoder...")
    # Using MLPRegressor as an autoencoder (predicting input from input with a bottleneck)
    n_features = X_train.shape[1]
    
    # Bottleneck architecture
    model = MLPRegressor(
        hidden_layer_sizes=(max(n_features//2, 2), max(n_features//4, 1), max(n_features//2, 2)),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    model.fit(X_train, X_train)
    
    # Calculate reconstruction error threshold on train
    train_pred = model.predict(X_train)
    train_mse = np.mean(np.power(X_train - train_pred, 2), axis=1)
    threshold = np.percentile(train_mse, 95)
    
    joblib.dump(model, 'models/saved/autoencoder.pkl')
    with open('models/saved/ae_threshold.json', 'w') as f:
        json.dump({'threshold': float(threshold)}, f)
        
    return model

def main():
    print("Loading data...")
    train_df, val_df, test_df = load_data()
    
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    preprocessor.fit(train_df)
    
    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)
    
    train_isolation_forest(X_train, X_val)
    train_autoencoder(X_train, X_val)
    
    print("Training complete. Models saved.")

if __name__ == "__main__":
    main()
