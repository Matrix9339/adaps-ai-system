# core/risk_fusion.py

def fuse_risks(machine_failure, supplier_delay, demand_spike, logistics_delay):
    """
    Combines all ML model outputs into a unified risk vector
    """
    return {
        "machine_failure": float(machine_failure),
        "supplier_delay": float(supplier_delay),
        "demand_spike": float(demand_spike),
        "logistics_delay": float(logistics_delay)
    }
