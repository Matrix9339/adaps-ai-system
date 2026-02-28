# core/orchestrator.py

import json
import datetime
import os

from core.planner import generate_optimized_plan
from core.constraint_engine import normalize_constraints
from core.loss_function import compute_loss
from core.risk_fusion import fuse_risks
from llm.llm_parser import extract_constraints

from model.model_loader import (
    predict_machine_failure,
    predict_supplier_delay,
    predict_demand_spike,
    predict_logistics_delay
)

# -------------------------------------------------
# Ensure log directory exists
# -------------------------------------------------
os.makedirs("logs", exist_ok=True)


def run_autonomous_planning(input_data):
    new_plans = []

    # -------------------------------------------------
    # Phase 1: Independent planning per run
    # -------------------------------------------------
    for run in input_data["production_runs"]:

        # -------------------------------
        # Predict risks (STANDARDIZED)
        # -------------------------------
        machine_risk = predict_machine_failure(run)
        supplier_delay_days, supplier_risk = predict_supplier_delay(run)
        demand_spike, demand_risk = predict_demand_spike(run)
        logistics_risk = predict_logistics_delay(run)

        # -------------------------------
        # Fuse risks
        # -------------------------------
        risks = fuse_risks(
            machine_risk,
            supplier_risk,
            demand_risk,
            logistics_risk
        )

        # -------------------------------
        # Extract & normalize constraints
        # -------------------------------
        raw_constraints = extract_constraints(input_data["messages"])
        constraints = normalize_constraints(raw_constraints)

        # -------------------------------
        # Loss before optimization
        # -------------------------------
        loss_before = compute_loss(run, risks, constraints)

        # -------------------------------
        # Generate optimized plan
        # -------------------------------
        new_plan = generate_optimized_plan(
            current_plan=run,
            risks=risks,
            constraints=constraints
        )

        # -------------------------------
        # Loss after optimization
        # -------------------------------
        loss_after = compute_loss(new_plan, risks, constraints)

        # -------------------------------
        # Attach risk metadata (CRITICAL)
        # -------------------------------
        new_plan.update({
            "machine_risk": machine_risk,
            "supplier_delay_days": supplier_delay_days,
            "supplier_risk": supplier_risk,
            "demand_spike": demand_spike,
            "demand_risk": demand_risk,
            "logistics_risk": logistics_risk,
            "risk_vector": risks,
            "loss_before": loss_before,
            "loss_after": loss_after
        })

        new_plans.append(new_plan)

    # -------------------------------------------------
    # Phase 2: Limit SPEED UP actions
    # -------------------------------------------------
    MAX_SPEED_UP = 2

    sorted_by_priority = sorted(
        new_plans,
        key=lambda x: x.get("priority", 0),
        reverse=True
    )

    speed_up_count = 0

    for plan in sorted_by_priority:
        if plan.get("action") == "SPEED UP":
            if speed_up_count < MAX_SPEED_UP:
                speed_up_count += 1
            else:
                plan["action"] = "NORMAL"

    # -------------------------------------------------
    # Phase 3: Audit logging
    # -------------------------------------------------
    for plan in new_plans:
        log_decision(
            plan["loss_before"],
            plan["loss_after"],
            plan
        )

    return new_plans


def log_decision(loss_before, loss_after, plan):
    with open("logs/auto_decisions.log", "a") as f:
        f.write(
            f"{datetime.datetime.now()} | "
            f"LOSS_BEFORE={loss_before} | "
            f"LOSS_AFTER={loss_after} | "
            f"PLAN={json.dumps(plan)}\n"
        )
