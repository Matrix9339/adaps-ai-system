import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("../dataset/Supplier Delay Prediction/logistics_shipments_dataset.csv")

print("Initial Data Shape:", df.shape)
print(df.head())

# =========================
# 2. FEATURE ENGINEERING
# =========================
df['Shipment_Date'] = pd.to_datetime(df['Shipment_Date'])
df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'])

df['Actual_Transit_Days'] = (df['Delivery_Date'] - df['Shipment_Date']).dt.days

df['Delayed'] = (df['Actual_Transit_Days'] > df['Transit_Days']).astype(int)

df.drop(
    columns=[
        'Shipment_ID',
        'Shipment_Date',
        'Delivery_Date',
        'Status',
        'Actual_Transit_Days'
    ],
    inplace=True
)

print("\nAfter Feature Engineering:")
print(df.head())

# =========================
# 3. SPLIT FEATURES & TARGET
# =========================
X = df.drop('Delayed', axis=1)
y = df['Delayed']

# =========================
# 4. PREPROCESSING
# =========================
categorical_features = [
    'Origin_Warehouse',
    'Destination',
    'Carrier'
]

numerical_features = [
    'Weight_kg',
    'Cost',
    'Distance_miles',
    'Transit_Days'
]

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_pipeline, categorical_features),
        ('num', numerical_pipeline, numerical_features)
    ]
)


# =========================
# 5. MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# =========================
# 6. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 7. TRAIN MODEL
# =========================
pipeline.fit(X_train, y_train)

# =========================
# 8. EVALUATION
# =========================
y_pred = pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# 9. CROSS VALIDATION
# =========================
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print("\nCross Validation Accuracy:", cv_scores.mean())

# =========================
# 10. PREDICT NEW SHIPMENT
# =========================
new_shipment = pd.DataFrame([{
    "Origin_Warehouse": "Warehouse_MIA",
    "Destination": "Chicago",
    "Carrier": "DHL",
    "Weight_kg": 40.5,
    "Cost": 180.0,
    "Distance_miles": 900,
    "Transit_Days": 3
}])

prediction = pipeline.predict(new_shipment)
probability = pipeline.predict_proba(new_shipment)

print("\nNew Shipment Delay Prediction:")
print("Delayed:", "YES" if prediction[0] == 1 else "NO")
print("Probability:", probability)


# ==============================
# 11. SAVE MODEL & METRICS
# ==============================
os.makedirs("model_checkpoints", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Save trained pipeline
joblib.dump(
    pipeline,
    "model_checkpoints/supplier_delay_random_forest.pkl"
)

# Save metrics
metrics = {
    "model": "RandomForestClassifier",
    "n_estimators": 200,
    "max_depth": 10,
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "cv_accuracy_mean": float(cv_scores.mean()),
    "num_records": int(len(df)),
    "num_features": int(X.shape[1]),
    "num_delayed": int(y.sum()),
    "num_on_time": int((y == 0).sum())
}

with open("reports/supplier_delay_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel and metrics saved successfully.")
