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



# ___________ load dataset ___________
df = pd.read_csv("../dataset\Machine Failure Prediction\predictive_maintenance.csv")  # change file name if needed
df.head()


# ____________ pre processing ______________
# drop ID columns (not useful for prediction)
df.drop(columns=['UDI', 'Product ID'], inplace=True)

# encode machine type (L, M, H)
le_type = LabelEncoder()
df['Type'] = le_type.fit_transform(df['Type'])

# binary target already present (0 = No Failure, 1 = Failure)
y = df['Target']



# _____________ Feature Engineering ____________
# # temperature difference (important feature)
df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']

# mechanical power proxy
df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']

# feature set
# clean column names for XGBoost compatibility
df.columns = (
    df.columns
    .str.replace('[', '', regex=False)
    .str.replace(']', '', regex=False)
    .str.replace('<', '', regex=False)
    .str.replace('>', '', regex=False)
)


features = [
    'Type',
    'Air temperature K',
    'Process temperature K',
    'Temp_Diff',
    'Rotational speed rpm',
    'Torque Nm',
    'Tool wear min',
    'Power'
]

X = df[features]


# __________ Data Split __________
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# __________ Train Model ___________
model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    gamma=1.0,
    reg_alpha=0.3,
    reg_lambda=1.0,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)


# __________ Model Evaluation ____________

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Test Accuracy:", accuracy)
print("ROC-AUC:", roc_auc)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ________________ Confusion Matrics_______________
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Machine Failure")
plt.show()


# _________________ Feature Importance ________________

imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data=imp_df, x='Importance', y='Feature')
plt.title("Feature Importance – Machine Failure Prediction")
plt.show()


# ___________ Save Model and Encoder _______________

joblib.dump(model, "./model_checkpoints/machine_failure_xgb.pkl")
joblib.dump(le_type, "./model_checkpoints/machine_type_encoder.pkl")

print("Model and encoder saved successfully.")


# ___________ save to json ___________
results = {
    "model_name": "XGBoost_Machine_Failure",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "test_accuracy": accuracy,
    "roc_auc": roc_auc,
    "num_samples": int(len(df)),
    "num_features": len(features),
    "top_features": imp_df.head(10).to_dict(orient="records"),
    "model_parameters": model.get_params()
}

with open("reports/machine_failure_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to machine_failure_results.json")


# __________ load and varify _________
loaded_model = joblib.load("./model_checkpoints/machine_failure_xgb.pkl")
loaded_encoder = joblib.load("./model_checkpoints/machine_type_encoder.pkl")

loaded_preds = loaded_model.predict(X_test)
print("Loaded model accuracy:", accuracy_score(y_test, loaded_preds))
