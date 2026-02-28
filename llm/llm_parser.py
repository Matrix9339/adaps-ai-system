# llm/llm_parser.py

from llm.schema import REQUIRED_CONSTRAINT_FIELDS

def extract_constraints(messages):
    constraints = {}

    for msg in messages:
        msg = msg.lower()

        if "delay" in msg:
            constraints["delay_penalty"] = 6000
        if "machine" in msg:
            constraints["failure_penalty"] = 10000
        if "overtime" in msg:
            constraints["overtime_cost"] = 150
        if "capacity" in msg:
            constraints["max_capacity"] = 900
        if "high demand" in msg:
            constraints["demand_penalty"] = 4000

    # enforce defaults
    for field in REQUIRED_CONSTRAINT_FIELDS:
        constraints.setdefault(field, default_value(field))

    return constraints


def default_value(field):
    defaults = {
        "max_capacity": 1000,
        "delay_penalty": 5000,
        "failure_penalty": 8000,
        "overtime_cost": 200,
        "demand_penalty": 3000
    }
    return defaults[field]
