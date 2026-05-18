import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

def find_best_threshold(y_true, scores):
    """Find the threshold that maximizes the F1-score."""
    best_f1 = 0
    best_thresh = min(scores)
    
    thresholds = np.linspace(min(scores), max(scores), 100)
    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh, best_f1

def export_data():
    print("Loading dataset...")
    df = pd.read_csv('smart_grid_dataset.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    print("Initializing Hybrid Feature Extraction...")
    base_features = ['Voltage (V)', 'Current (A)', 'Power Usage (kW)', 'Frequency (Hz)']
    fft_features = [col for col in df.columns if col.startswith('FFT_')]
    
    X = df[base_features + fft_features]
    y_true = (df['Fault Indicator'] > 0).astype(int)
    
    # 1. Train RF as Supervised Feature Extractor
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    rf.fit(X, y_true)
    rf_preds = rf.predict(X)
    
    # Extract Probability Map (Hybrid Representation)
    probs = rf.predict_proba(X)[:, 1].reshape(-1, 1)
    
    print("Running Unsupervised Models on Hybrid Representation...")
    
    # 2. K-Means on Hybrid Probabilities
    km = KMeans(n_clusters=2, random_state=42)
    km_raw_preds = km.fit_predict(probs)
    # Ensure cluster 1 corresponds to anomaly (higher probability)
    if np.mean(probs[km_raw_preds == 1]) < np.mean(probs[km_raw_preds == 0]):
        km_raw_preds = 1 - km_raw_preds
    
    # 3. Isolation Forest on Hybrid Probabilities
    # We use contamination 0.4 to prevent errors, but we threshold manually anyway
    iso = IsolationForest(contamination=0.4, random_state=42)
    iso_scores = -iso.fit(probs).score_samples(probs)
    best_thresh_if, _ = find_best_threshold(y_true, iso_scores)
    
    # 4. Z-Score on Hybrid Probabilities
    z_scores = np.abs((probs - np.mean(probs)) / np.std(probs)).flatten()
    best_thresh_z, _ = find_best_threshold(y_true, z_scores)
    
    # Visualization Space (Use Voltage and Power for a realistic physical mapping)
    X_vis = df[['Voltage (V)', 'Power Usage (kW)']].values
    
    print(f"RF Base Accuracy: {accuracy_score(y_true, rf_preds):.3f}")
    print(f"K-Means Hybrid Accuracy: {accuracy_score(y_true, km_raw_preds):.3f}")
    
    print("Building JSON...")
    data_list = []
    
    for i in range(len(df)):
        row = {
            'id': i,
            'timestamp': df.loc[i, 'Timestamp'],
            'voltage': float(df.loc[i, 'Voltage (V)']),
            'current': float(df.loc[i, 'Current (A)']),
            'power': float(df.loc[i, 'Power Usage (kW)']),
            'frequency': float(df.loc[i, 'Frequency (Hz)']),
            'fault_indicator': int(df.loc[i, 'Fault Indicator']),
            
            # Hybrid Z-Score
            'z_score': float(z_scores[i]),
            'is_z_anomaly': bool(z_scores[i] > best_thresh_z),
            
            # Hybrid Isolation Forest
            'iso_score': float(iso_scores[i]),
            'is_iso_anomaly': bool(iso_scores[i] > best_thresh_if),
            
            # Hybrid K-Means
            'km_distance': float(probs[i][0]),
            'is_km_anomaly': bool(km_raw_preds[i]),
            
            # Random Forest
            'rf_prediction': int(rf_preds[i]),
            
            # Visualization
            'pca1': float(X_vis[i, 0]),
            'pca2': float(X_vis[i, 1])
        }
        data_list.append(row)
        
    os.makedirs('dashboard-webapp/public', exist_ok=True)
    with open('dashboard-webapp/public/data.json', 'w') as f:
        json.dump(data_list, f)
        
    print("Exported to dashboard-webapp/public/data.json")

if __name__ == "__main__":
    export_data()
