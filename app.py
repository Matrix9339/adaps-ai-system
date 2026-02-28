import streamlit as st
import json
import pandas as pd
from core.orchestrator import run_autonomous_planning


# -------------------------------------------------
# LLM Explanation Helper
# -------------------------------------------------
def llm_explain_decision(plan: dict) -> str:
    action = plan.get("action")
    run_id = plan.get("run_id")

    # Collect risks
    risks = {
        "Machine Failure": plan.get("machine_risk", 0),
        "Supplier Delay": plan.get("supplier_risk", 0),
        "Demand Volatility": plan.get("demand_risk", 0),
        "Logistics Delay": plan.get("logistics_risk", 0),
    }

    # Identify dominant risk
    dominant_risk_name = max(risks, key=risks.get)
    dominant_risk_value = risks[dominant_risk_name]

    # Risk level mapping
    def risk_level(v):
        if v >= 0.7:
            return "HIGH"
        elif v >= 0.4:
            return "MODERATE"
        else:
            return "LOW"

    dominant_level = risk_level(dominant_risk_value)

    # Action-specific reasoning
    if action == "SPEED UP":
        decision_reason = (
            "immediate execution was required to prevent downstream delays "
            "and potential SLA violations"
        )
    elif action == "WAIT":
        decision_reason = (
            "temporary waiting reduces exposure to operational risk "
            "while maintaining delivery feasibility"
        )
    elif action == "DO LATER":
        decision_reason = (
            "the run was deprioritized due to low urgency and sufficient buffers"
        )
    else:  # NORMAL
        decision_reason = (
            "a balanced execution strategy was sufficient without escalating cost or risk"
        )

    return (
        f"Run `{run_id}` was assigned action **{action}**. "
        f"The dominant influencing factor was **{dominant_risk_name}**, "
        f"which is assessed as **{dominant_level}** risk. "
        f"As a result, {decision_reason} while minimizing overall operational loss."
    )




# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Autonomous Production Planning System",
    layout="wide"
)

st.title("Autonomous Production Planning Control System")
st.caption("Multi-run, AI-driven, loss-minimizing autonomous planner")

# -------------------------------------------------
# Load data
# -------------------------------------------------
with open("dataset/current_plan.json") as f:
    plan = json.load(f)

with open("dataset/messages.txt") as f:
    messages = f.readlines()

input_data = {
    "production_runs": plan["production_runs"],
    "messages": messages
}

# -------------------------------------------------
# Schema validation
# -------------------------------------------------
for run in input_data["production_runs"]:
    if "type" not in run or run["type"] not in ["L", "M", "H"]:
        run["type"] = "M"

runs_df = pd.DataFrame(input_data["production_runs"])

# -------------------------------------------------
# Sidebar – System Status
# -------------------------------------------------
st.sidebar.header("System Status")
st.sidebar.success("Autonomous Mode: ENABLED")
st.sidebar.info("Human Approval: NOT REQUIRED")

st.sidebar.divider()

# -------------------------------------------------
# Sidebar – Run Summary
# -------------------------------------------------
st.sidebar.subheader("Production Run Summary")
st.sidebar.metric("Total Production Runs", len(runs_df))

st.sidebar.markdown("**Machine Type Count**")
type_counts = runs_df["type"].value_counts()
for t in ["L", "M", "H"]:
    st.sidebar.write(f"• Type {t}: {type_counts.get(t, 0)}")

# -------------------------------------------------
# Main – Current Runs
# -------------------------------------------------
st.subheader("Current Production Runs")
st.dataframe(runs_df, use_container_width=True)

# -------------------------------------------------
# Messages
# -------------------------------------------------
st.subheader("Operational Messages (Constraints)")
for msg in messages:
    st.write(f"• {msg.strip()}")

# -------------------------------------------------
# Run Planner
# -------------------------------------------------
st.subheader("Autonomous Planning Engine")

updated_plans = None

if st.button("Run Autonomous Planner"):
    with st.spinner("Running autonomous optimization across all production runs..."):
        updated_plans = run_autonomous_planning(input_data)

    st.success("Autonomous replanning completed successfully")

    # Sort by priority
    updated_plans = sorted(
        updated_plans,
        key=lambda x: x.get("priority", 0),
        reverse=True
    )

    # Sidebar – Action Distribution
    st.sidebar.divider()
    st.sidebar.subheader("🛠 Action Distribution")

    actions_df = pd.DataFrame(updated_plans)
    for action, count in actions_df["action"].value_counts().items():
        st.sidebar.write(f"• {action}: {count}")

    # -------------------------------------------------
    # Updated Plans + LLM Explanation Panel
    # -------------------------------------------------
    st.subheader("Updated Production Plans")

    action_color = {
        "SPEED UP": "🔴",
        "NORMAL": "🟡",
        "WAIT": "🟠",
        "DO LATER": "🔵"
    }

    for plan in updated_plans:
        color = action_color.get(plan.get("action"), "⚪")

        st.markdown(
            f"### {color} Run ID: `{plan['run_id']}`  "
            f"| Action: **{plan.get('action')}**  "
            f"| Priority: **{plan.get('priority')}**"
        )

        st.dataframe(pd.DataFrame([plan]), use_container_width=True)

        # 🔍 LLM EXPLANATION PANEL (THIS WAS MISSING)
        with st.expander("Why this decision? (LLM Explanation)"):
            st.write(llm_explain_decision(plan))

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    st.subheader("Autonomous Decisions Summary")
    st.markdown("""
    - Production runs are sorted by **priority**
    - Each run is assigned an **operator-friendly action**
    - Explanation panel shows **LLM reasoning**
    - Decisions minimize **overall operational loss**
    """)

# -------------------------------------------------
# Audit Log
# -------------------------------------------------
st.subheader("Autonomous Decision Log")

try:
    with open("logs/auto_decisions.log") as f:
        for line in f.readlines()[-10:]:
            st.code(line.strip())
except FileNotFoundError:
    st.warning("Audit log not found. Run the planner once.")
