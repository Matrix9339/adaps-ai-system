def decide_action(priority, risks):
    if priority > 0.40:
        return "SPEED UP"        # high risk, act now
    elif priority > 0.28:
        return "NORMAL"      # monitor
    elif risks.get("machine_failure",0 )> 0.4:
        return "WAIT"       # wait to avoid failure loss
    else:
        return "DO LATER"           # low priority, save cost
