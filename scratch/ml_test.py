import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('../smart_grid_dataset.csv')

base_features = ['Voltage (V)', 'Current (A)', 'Power Usage (kW)', 'Frequency (Hz)']
fft_features = [col for col in df.columns if col.startswith('FFT_')]
target = 'Fault Indicator'

X = df[base_features + fft_features]
y = df[target]

print(f"Data shape: {X.shape}, Target distribution:\n{y.value_counts()}")

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Evaluating Models without PCA ---")
models = {
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1),
    "SVM": SVC(kernel='rbf', random_state=42),
    "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")

# Let's try with PCA to see if noise reduction helps
print("\n--- Evaluating Models WITH PCA (95% variance) ---")
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA components: {pca.n_components_}")

for name, model in models.items():
    model.fit(X_train_pca, y_train)
    preds = model.predict(X_test_pca)
    acc = accuracy_score(y_test, preds)
    print(f"{name} (PCA) Accuracy: {acc:.4f}")

