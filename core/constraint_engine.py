def normalize_constraints(raw):
    return {
        "max_daily_capacity": raw.get("max_capacity", 1000),
        "delay_penalty": raw.get("delay_penalty", 5000),
        "failure_penalty": raw.get("failure_penalty", 8000),
        "overtime_cost": raw.get("overtime_cost", 200),
        "demand_penalty": raw.get("demand_penalty", 3000)
    }
