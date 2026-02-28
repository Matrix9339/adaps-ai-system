from core.loss_function import compute_loss
from core.decision_engine import decide_action


def generate_optimized_plan(current_plan, risks, constraints):

    new_plan = current_plan.copy()

    # -------------------------------------------------
    # Ensure numeric fields exist (SAFETY)
    # -------------------------------------------------
    new_plan["daily_output"] = int(new_plan.get("daily_output", 0))
    new_plan["buffer_stock"] = int(new_plan.get("buffer_stock", 0))
    new_plan["overtime_hours"] = int(new_plan.get("overtime_hours", 0))
    new_plan["expected_demand"] = int(new_plan.get("expected_demand", 0))

    # -------------------------------------------------
    # BASE PRIORITY (risk-based)
    # -------------------------------------------------
    priority = (
        0.4 * risks.get("demand_spike", 0) +
        0.3 * risks.get("supplier_delay", 0) +
        0.3 * risks.get("machine_failure", 0)
    )

    # -------------------------------------------------
    # Distance influence (logistics urgency)
    # -------------------------------------------------
    priority += min(new_plan.get("distance_miles", 0) / 2000, 0.2)

    # -------------------------------------------------
    # LOSS AWARENESS (BUSINESS IMPACT)
    # -------------------------------------------------
    base_unmet = max(
        0,
        new_plan["expected_demand"] - new_plan["daily_output"]
    )

    base_loss = base_unmet * constraints.get("demand_penalty", 0)

    if base_loss > 1_000_000:
        priority += 0.15
    elif base_loss > 500_000:
        priority += 0.08

    # Cap priority to 1.0 for stability
    priority = min(priority, 1.0)

    new_plan["priority"] = round(priority, 3)

    # -------------------------------------------------
    # Decide action (AFTER loss-aware priority)
    # -------------------------------------------------
    action = decide_action(priority, risks)
    new_plan["action"] = action

    # -------------------------------------------------
    # Action-based planning (OPERATOR FRIENDLY)
    # -------------------------------------------------
    if action == "SPEED UP":
        new_plan["daily_output"] += 300
        new_plan["buffer_stock"] += 150
        new_plan["overtime_hours"] += 2

    elif action == "WAIT":
        new_plan["daily_output"] = int(new_plan["daily_output"] * 0.7)
        new_plan["overtime_hours"] = 0

    elif action == "DO LATER":
        new_plan["daily_output"] = int(new_plan["daily_output"] * 0.6)
        new_plan["buffer_stock"] = max(0, new_plan["buffer_stock"] - 100)

    # NORMAL → no change

    # -------------------------------------------------
    # Capacity enforcement
    # -------------------------------------------------
    max_cap = constraints.get("max_daily_capacity", new_plan["daily_output"])
    if new_plan["daily_output"] > max_cap:
        new_plan["daily_output"] = max_cap

    # -------------------------------------------------
    # Unmet demand
    # -------------------------------------------------
    new_plan["unmet_demand"] = max(
        0,
        new_plan["expected_demand"] - new_plan["daily_output"]
    )

    # -------------------------------------------------
    # Final loss
    # -------------------------------------------------
    new_plan["loss"] = compute_loss(new_plan, risks, constraints)

    return new_plan
