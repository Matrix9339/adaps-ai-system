# core/loss_function.py

def compute_loss(plan, risks, constraints):

    # -------------------------------
    # Ensure unmet_demand exists
    # -------------------------------
    if "unmet_demand" not in plan:
        unmet_demand = max(
            0,
            plan["expected_demand"] - plan["daily_output"]
        )
    else:
        unmet_demand = plan["unmet_demand"]

    production_loss = unmet_demand * constraints["demand_penalty"]
    delay_penalty = risks["supplier_delay"] * constraints["delay_penalty"]
    failure_penalty = risks["machine_failure"] * constraints["failure_penalty"]
    overtime_cost = plan.get("overtime_hours", 0) * constraints["overtime_cost"]

    total_loss = (
        production_loss +
        delay_penalty +
        failure_penalty +
        overtime_cost
    )

    return total_loss
