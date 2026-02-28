# ==========================================
# LOGISTICS TRANSPORTATION DELAY PREDICTION
# (TRAIN–TEST SPLIT VERSION)
# ==========================================

# ---------- 1. IMPORTS ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

# ---------- 2. LOAD DATA ----------
df = pd.read_csv("logistics_delay.csv")   # <-- change path if needed

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

# ---------- 3. FEATURE ENGINEERING ----------
# temporal features
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df['Month'] = df['Timestamp'].dt.month

# categorical encoding
cat_cols = ['Asset_ID', 'Traffic_Status']

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ---------- 4. FEATURE SET (LEAKAGE REMOVED) ----------
features = [
    'Asset_ID',
    'Latitude',
    'Longitude',
    'Inventory_Level',
    'Temperature',
    'Humidity',
    'Traffic_Status',
    'Waiting_Time',
    'User_Transaction_Amount',
    'User_Purchase_Frequency',
    'Asset_Utilization',
    'Demand_Forecast',
    'Hour',
    'DayOfWeek',
    'Month'
]

X = df[features]
y = df['Logistics_Delay']

# ---------- 5. TRAIN–TEST SPLIT (TIME-AWARE) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False   # VERY IMPORTANT
)

# ---------- 6. REGULARIZED XGBOOST MODEL ----------
model = XGBClassifier(
    n_estimators=120,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=1.0,
    reg_alpha=0.5,
    reg_lambda=1.0,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# ---------- 7. EVALUATION ON TEST SET ----------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Test Accuracy:", accuracy)
print("Test ROC-AUC:", roc_auc)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------- 8. CONFUSION MATRIX ----------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Logistics Delay")
plt.show()

# ---------- 9. FEATURE IMPORTANCE ----------
imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data=imp_df.head(10), x='Importance', y='Feature')
plt.title("Top 10 Feature Importances")
plt.show()

# ---------- 10. SAVE MODEL & ENCODERS ----------
joblib.dump(model, "logistics_delay_xgb.pkl")
joblib.dump(encoders, "logistics_encoders.pkl")

print("Model and encoders saved.")

# ---------- 11. SAVE METRICS TO JSON ----------
results = {
    "model_name": "XGBoost_Logistics_Delay",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "test_accuracy": accuracy,
    "roc_auc": roc_auc,
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "num_features": len(features),
    "top_features": imp_df.head(10).to_dict(orient="records"),
    "model_parameters": model.get_params()
}

with open("logistics_delay_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to logistics_delay_results.json")

# ---------- 12. LOAD & VERIFY ----------
loaded_model = joblib.load("logistics_delay_xgb.pkl")
loaded_encoders = joblib.load("logistics_encoders.pkl")

loaded_preds = loaded_model.predict(X_test)
print("Loaded model test accuracy:", accuracy_score(y_test, loaded_preds))
