import json
from core.orchestrator import run_autonomous_planning

with open("dataset/current_plan.json") as f:
    plan = json.load(f)

with open("dataset/messages.txt") as f:
    messages = f.readlines()

input_data = {
    "current_plan": plan,
    "messages": messages
}

new_plan = run_autonomous_planning(input_data)
print("NEW AUTONOMOUS PLAN:")
print(new_plan)
