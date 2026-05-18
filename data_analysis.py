import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os
import xgboost as xgb

def analyze_data():
    print("Loading data...")
    df = pd.read_csv('smart_grid_dataset.csv')
    
    # 1. Preprocessing
    print("Preprocessing...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    
    # Identify feature columns
    base_features = ['Voltage (V)', 'Current (A)', 'Power Usage (kW)', 'Frequency (Hz)']
    fft_features = [col for col in df.columns if col.startswith('FFT_')]
    target = 'Fault Indicator'
    
    # The Fault Indicator might have multiple classes. Let's make it binary for basic anomaly detection: 0 = Normal, 1 = Anomaly
    # Wait, let's see unique values first
    print(f"Unique Fault Indicator values: {df[target].unique()}")
    # Let's assume 0 is normal, anything else is anomaly
    y = (df[target] > 0).astype(int)
    
    X_base = df[base_features]
    X_fft = df[fft_features]
    
    # Scale base features
    scaler_base = StandardScaler()
    X_base_scaled = scaler_base.fit_transform(X_base)
    
    # Scale and PCA for FFT features
    scaler_fft = StandardScaler()
    X_fft_scaled = scaler_fft.fit_transform(X_fft)
    
    pca = PCA(n_components=10) # Reduce 128 FFTs to 10 principal components
    X_fft_pca = pca.fit_transform(X_fft_scaled)
    print(f"PCA explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Combine features
    X_combined = np.hstack((X_base_scaled, X_fft_pca))
    
    print(f"Combined feature shape: {X_combined.shape}")
    print(f"Total Anomalies: {y.sum()} out of {len(y)} ({y.sum()/len(y)*100:.2f}%)")
    
    results = {}
    
    # --- Layer 1: Z-Score ---
    print("\n--- Layer 1: Z-Score Screening ---")
    z_scores = np.abs((X_combined - np.mean(X_combined, axis=0)) / np.std(X_combined, axis=0))
    # If any feature has Z > 3, flag as anomaly
    y_pred_z = (z_scores > 3).any(axis=1).astype(int)
    print(classification_report(y, y_pred_z))
    results['Z-Score'] = {'f1': f1_score(y, y_pred_z), 'auc': roc_auc_score(y, y_pred_z)}
    
    # --- Layer 2: Isolation Forest ---
    print("\n--- Layer 2: Isolation Forest ---")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    y_pred_if_raw = iso_forest.fit_predict(X_combined)
    # IsolationForest returns 1 for inliers, -1 for outliers
    y_pred_if = (y_pred_if_raw == -1).astype(int)
    print(classification_report(y, y_pred_if))
    iso_scores = -iso_forest.score_samples(X_combined) # Higher is more anomalous
    results['Isolation Forest'] = {'f1': f1_score(y, y_pred_if), 'auc': roc_auc_score(y, iso_scores)}
    
    # --- Layer 3: K-Means ---
    print("\n--- Layer 3: K-Means ---")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X_combined)
    distances = kmeans.transform(X_combined)
    min_distances = distances.min(axis=1)
    threshold = np.percentile(min_distances, 95) # Top 5% furthest points as anomalies
    y_pred_km = (min_distances > threshold).astype(int)
    # --- Benchmark: XGBoost & Random Forest Binary ---
    print("\n--- Benchmark: Binary Classification ---")
    y_binary = (df[target] > 0).astype(int)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
    
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("Random Forest:")
    print(classification_report(y_test, y_pred_rf))
    
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_estimators=200, max_depth=5, learning_rate=0.05)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    print("XGBoost:")
    print(classification_report(y_test, y_pred_xgb))
    
    # Save models and scalers
    print("\nSaving models to disk...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler_base, 'models/scaler_base.pkl')
    joblib.dump(scaler_fft, 'models/scaler_fft.pkl')
    joblib.dump(pca, 'models/pca.pkl')
    joblib.dump(iso_forest, 'models/iso_forest.pkl')
    joblib.dump(kmeans, 'models/kmeans.pkl')
    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(xgb_model, 'models/xgb_model.pkl')
    print("Done!")

if __name__ == "__main__":
    analyze_data()
