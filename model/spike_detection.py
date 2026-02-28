# ==============================
# 1. IMPORTS
# ==============================
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==============================
# 2. LOAD DATA
# ==============================
df = pd.read_csv("../dataset/Demand Spike Detection Wallmart/Walmart.csv")

# parse date
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# sort data (CRITICAL for time series)
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)


# ==============================
# 3. FEATURE ENGINEERING
# ==============================

# Lag features
df['Lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
df['Lag_4'] = df.groupby('Store')['Weekly_Sales'].shift(4)

# Rolling statistics
df['Rolling_Mean_4'] = df.groupby('Store')['Weekly_Sales'].transform(
    lambda x: x.rolling(window=4, min_periods=1).mean()
)

df['Rolling_Std_4'] = df.groupby('Store')['Weekly_Sales'].transform(
    lambda x: x.rolling(window=4, min_periods=1).std()
)

# Drop rows with NaNs caused by lagging
df_model = df.dropna().copy()


# ==============================
# 4. FEATURE SET (NO DATE)
# ==============================
features = [
    'Weekly_Sales',
    'Lag_1',
    'Lag_4',
    'Rolling_Mean_4',
    'Rolling_Std_4',
    'Holiday_Flag',
    'Temperature',
    'Fuel_Price',
    'CPI',
    'Unemployment'
]

X = df_model[features]


# ==============================
# 5. TRAIN FINAL ISOLATION FOREST
# ==============================
iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

iso.fit(X)

df_model['target'] = (iso.predict(X) == -1).astype(int)
df_model['anomaly_score'] = iso.decision_function(X)


# ==============================
# 6. VISUALIZATION (ANALYSIS ONLY)
# ==============================
plt.figure(figsize=(14,6))
plt.plot(df_model['Date'], df_model['Weekly_Sales'], label='Weekly Sales')

plt.scatter(
    df_model.loc[df_model['target'] == 1, 'Date'],
    df_model.loc[df_model['target'] == 1, 'Weekly_Sales'],
    color='red',
    label='Detected Spikes'
)

plt.title("Demand Spike Detection using Isolation Forest")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.grid(True)
plt.show()


# ==============================
# 7. TIME-SERIES CROSS-VALIDATION (DIAGNOSTIC)
# ==============================
tscv = TimeSeriesSplit(n_splits=5)
cv_anomaly_rates = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    iso_cv = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )

    iso_cv.fit(X_train)
    preds = (iso_cv.predict(X_test) == -1).astype(int)
    cv_anomaly_rates.append(preds.mean())

cv_anomaly_rate = float(np.mean(cv_anomaly_rates))
print("Average anomaly rate across folds:", cv_anomaly_rate)


# ==============================
# 8. BASELINE AGREEMENT (NOT TRUE ACCURACY)
# ==============================
df_model['stat_label'] = (
    df_model['Weekly_Sales'] > 1.25 * df_model['Rolling_Mean_4']
).astype(int)

agreement_accuracy = accuracy_score(
    df_model['stat_label'],
    df_model['target']
)

print("\nAgreement with statistical baseline:", agreement_accuracy)
print("\nClassification Report:\n")
print(classification_report(df_model['stat_label'], df_model['target']))


# ==============================
# 9. CONFUSION MATRIX (ANALYSIS)
# ==============================
cm = confusion_matrix(df_model['stat_label'], df_model['target'])

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ==============================
# 10. SAVE MODEL & METRICS
# ==============================
os.makedirs("model_checkpoints", exist_ok=True)


# Save model
joblib.dump(
    iso, "model_checkpoints/demand_spike_isolation_forest.pkl"
)


# Save metrics
metrics = {
    "model": "IsolationForest",
    "contamination": 0.05,
    "accuracy_agreement_with_statistical_baseline": agreement_accuracy,
    "cv_anomaly_rate": cv_anomaly_rate,
    "baseline_agreement": agreement_accuracy,
    "mean_score": float(df_model['anomaly_score'].mean()),
    "min_score": float(df_model['anomaly_score'].min()),
    "max_score": float(df_model['anomaly_score'].max()),
    "num_records": int(len(df_model)),
    "num_anomalies": int(df_model['target'].sum())
}

with open("reports/demand_spike_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel and metrics saved successfully.")
