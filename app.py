import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

st.set_page_config(page_title="Smart Grid Anomaly Detection", layout="wide", page_icon="⚡")

# Load Data and Models
@st.cache_data
def load_data():
    df = pd.read_csv('smart_grid_dataset.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    return df

@st.cache_resource
def load_models():
    scaler_base = joblib.load('models/scaler_base.pkl')
    scaler_fft = joblib.load('models/scaler_fft.pkl')
    pca = joblib.load('models/pca.pkl')
    iso_forest = joblib.load('models/iso_forest.pkl')
    kmeans = joblib.load('models/kmeans.pkl')
    rf_model = joblib.load('models/rf_model.pkl')
    return scaler_base, scaler_fft, pca, iso_forest, kmeans, rf_model

def preprocess(df, scaler_base, scaler_fft, pca):
    base_features = ['Voltage (V)', 'Current (A)', 'Power Usage (kW)', 'Frequency (Hz)']
    fft_features = [col for col in df.columns if col.startswith('FFT_')]
    
    X_base = df[base_features]
    X_fft = df[fft_features]
    
    X_base_scaled = scaler_base.transform(X_base)
    X_fft_scaled = scaler_fft.transform(X_fft)
    X_fft_pca = pca.transform(X_fft_scaled)
    
    X_combined = np.hstack((X_base_scaled, X_fft_pca))
    y_true = (df['Fault Indicator'] > 0).astype(int)
    
    return X_combined, y_true

st.title("⚡ Smart Grid Anomaly Detection Dashboard")
st.markdown("""
This dashboard implements a hybrid anomaly detection pipeline for real-time smart grid monitoring. 
It features lightweight, interpretable layers (Z-Score, Isolation Forest, K-Means) alongside a Random Forest benchmark model.
""")

try:
    df = load_data()
    scaler_base, scaler_fft, pca, iso_forest, kmeans, rf_model = load_models()
    X_combined, y_true = preprocess(df, scaler_base, scaler_fft, pca)
except Exception as e:
    st.error(f"Error loading data or models: {e}. Please ensure data_analysis.py has been run.")
    st.stop()

tabs = st.tabs(["📊 Data Overview", "🔍 Unsupervised Detection", "🌳 Random Forest Benchmark"])

# --- Tab 1: Data Overview ---
with tabs[0]:
    st.header("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Features", len(df.columns) - 2) # Exclude Timestamp and Fault Indicator
    col3.metric("Anomalies (Fault Indicator > 0)", f"{y_true.sum()} ({y_true.sum()/len(df)*100:.1f}%)")
    
    st.subheader("Time Series Analysis")
    feature_to_plot = st.selectbox("Select Feature to Visualize", ['Voltage (V)', 'Current (A)', 'Power Usage (kW)', 'Frequency (Hz)'])
    fig = px.line(df, x='Timestamp', y=feature_to_plot, color='Fault Indicator', title=f'{feature_to_plot} over Time')
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Unsupervised Detection ---
with tabs[1]:
    st.header("Three-Layer Anomaly Detection (Unsupervised)")
    
    st.subheader("Layer 1: Z-Score Screening")
    z_scores = np.abs((X_combined - np.mean(X_combined, axis=0)) / np.std(X_combined, axis=0))
    max_z = np.max(z_scores, axis=1)
    
    threshold_z = st.slider("Z-Score Threshold", 1.0, 10.0, 3.0, 0.5)
    y_pred_z = (max_z > threshold_z).astype(int)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Flagged Anomalies:** {y_pred_z.sum()} ({y_pred_z.sum()/len(y_pred_z)*100:.1f}%)")
        fig_z = px.histogram(max_z, nbins=50, title="Max Z-Score Distribution")
        fig_z.add_vline(x=threshold_z, line_dash="dash", line_color="red")
        st.plotly_chart(fig_z, use_container_width=True)
    
    st.subheader("Layer 2: Isolation Forest")
    iso_scores = -iso_forest.score_samples(X_combined)
    threshold_if = st.slider("Isolation Forest Score Threshold", float(np.min(iso_scores)), float(np.max(iso_scores)), float(np.percentile(iso_scores, 90)), 0.01)
    y_pred_if = (iso_scores > threshold_if).astype(int)
    
    col3, col4 = st.columns(2)
    with col3:
        st.write(f"**Flagged Anomalies:** {y_pred_if.sum()} ({y_pred_if.sum()/len(y_pred_if)*100:.1f}%)")
        fig_if = px.histogram(iso_scores, nbins=50, title="Isolation Forest Score Distribution")
        fig_if.add_vline(x=threshold_if, line_dash="dash", line_color="red")
        st.plotly_chart(fig_if, use_container_width=True)
        
    st.subheader("Layer 3: K-Means Clustering")
    distances = kmeans.transform(X_combined)
    min_distances = distances.min(axis=1)
    threshold_km = st.slider("K-Means Distance Threshold", float(np.min(min_distances)), float(np.max(min_distances)), float(np.percentile(min_distances, 95)), 0.1)
    y_pred_km = (min_distances > threshold_km).astype(int)
    
    st.write(f"**Flagged Anomalies:** {y_pred_km.sum()} ({y_pred_km.sum()/len(y_pred_km)*100:.1f}%)")
    
    # 2D PCA Visualization
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_combined)
    plot_df = pd.DataFrame({'PCA1': X_pca_2d[:, 0], 'PCA2': X_pca_2d[:, 1], 'Anomaly': ['Anomaly' if val == 1 else 'Normal' for val in y_pred_km]})
    fig_km = px.scatter(plot_df, x='PCA1', y='PCA2', color='Anomaly', title="K-Means Anomalies (PCA Reduced)", color_discrete_map={'Normal': '#1f77b4', 'Anomaly': '#d62728'})
    st.plotly_chart(fig_km, use_container_width=True)

# --- Tab 3: Random Forest Benchmark ---
with tabs[2]:
    st.header("Supervised Benchmark: Random Forest")
    st.markdown("A supervised Random Forest model trained to classify points as Normal (0) or Fault (1 or 2) using 80% of the dataset.")
    
    y_pred_rf = rf_model.predict(X_combined)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred_rf)
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="True"), x=['Normal', 'Fault'], y=['Normal', 'Fault'], color_continuous_scale='Blues')
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col2:
        st.subheader("Performance Metrics (Entire Dataset)")
        report = classification_report(y_true, y_pred_rf, output_dict=True)
        st.write(f"**Accuracy:** {report['accuracy']:.2f}")
        st.write(f"**Precision (Fault):** {report['1']['precision']:.2f}")
        st.write(f"**Recall (Fault):** {report['1']['recall']:.2f}")
        st.write(f"**F1-Score (Fault):** {report['1']['f1-score']:.2f}")

    # Feature Importance
    st.subheader("Feature Importance")
    importances = rf_model.feature_importances_
    features = ['Voltage', 'Current', 'Power Usage', 'Frequency'] + [f'FFT_PCA_{i+1}' for i in range(10)]
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
    fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Random Forest Feature Importances")
    st.plotly_chart(fig_imp, use_container_width=True)

