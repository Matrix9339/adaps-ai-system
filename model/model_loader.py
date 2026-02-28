# model/model_loader.py

import joblib
import pandas as pd
from pathlib import Path

# -------------------------------------------------
# Resolve PROJECT ROOT safely
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = PROJECT_ROOT / "model" / "model_checkpoints"

# -------------------------------------------------
# Load trained models
# -------------------------------------------------
machine_model = joblib.load(CKPT_DIR / "machine_failure_xgb.pkl")
machine_type_encoder = joblib.load(CKPT_DIR / "machine_type_encoder.pkl")

supplier_model = joblib.load(CKPT_DIR / "supplier_delay_random_forest.pkl")

# =================================================
# Machine Failure Prediction (ML – SAFE)
# =================================================
def predict_machine_failure(run):
    """
    Returns:
    - machine_risk: float (0–1)
    """

    df = pd.DataFrame([{
        "Type": run.get("type", "M"),
        "Air temperature K": run.get("air_temp", 300),
        "Process temperature K": run.get("process_temp", 310),
        "Rotational speed rpm": run.get("rpm", 1500),
        "Torque Nm": run.get("torque", 40),
        "Tool wear min": run.get("tool_wear", 50)
    }])

    df["Type"] = machine_type_encoder.transform(df["Type"])

    df["Temp_Diff"] = df["Process temperature K"] - df["Air temperature K"]
    df["Power"] = df["Rotational speed rpm"] * df["Torque Nm"]

    features = [
        "Type",
        "Air temperature K",
        "Process temperature K",
        "Temp_Diff",
        "Rotational speed rpm",
        "Torque Nm",
        "Tool wear min",
        "Power"
    ]

    df = df[features]

    return float(machine_model.predict_proba(df)[0][1])


# =================================================
# Supplier Delay Prediction (FULL ML – STANDARDIZED)
# =================================================
def predict_supplier_delay(run):
    """
    Returns:
    - supplier_delay_days: int
    - supplier_risk: float (0–1)
    """

    df = pd.DataFrame([{
        "Origin_Warehouse": run["origin_warehouse"],
        "Destination": run["destination"],
        "Carrier": run["carrier"],
        "Weight_kg": run["weight_kg"],
        "Cost": run["cost"],
        "Distance_miles": run["distance_miles"],
        "Transit_Days": run["transit_days"]
    }])

    supplier_risk = float(supplier_model.predict_proba(df)[0][1])

    # Map probability → delay days (business mapping)
    supplier_delay_days = max(0, int(round(supplier_risk * 5)))

    return supplier_delay_days, supplier_risk


# =================================================
# Demand Spike Detection (RULE + RISK SCORE)
# =================================================
def predict_demand_spike(run):
    """
    Returns:
    - demand_spike: bool
    - demand_risk: float (0–1)
    """

    expected = run["expected_demand"]
    historical_avg = run.get("historical_avg", expected)

    ratio = expected / max(historical_avg, 1)

    demand_spike = ratio > 1.25

    # Convert ratio to bounded risk score
    demand_risk = min(1.0, max(0.0, (ratio - 1.0) / 1.0))

    return demand_spike, demand_risk


# =================================================
# Logistics Delay Prediction (RULE + RISK SCORE)
# =================================================
def predict_logistics_delay(run):
    """
    Returns:
    - logistics_risk: float (0–1)
    """

    distance = run.get("distance_miles", 0)

    if distance > 1500:
        return 0.7
    elif distance > 800:
        return 0.4
    else:
        return 0.1
