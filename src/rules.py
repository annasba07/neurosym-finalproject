def observation_predicates(facts):
    preds = []

    if facts.get("fever") == "yes":
        preds.append("fever_present")

    if facts.get("alert") == "reduced":
        preds.append("lethargy")

    if facts.get("vomiting") == "repeated":
        preds.append("repeated_vomiting")

    if facts.get("intake") == "reduced":
        preds.append("poor_intake")

    if facts.get("intake") == "none":
        preds.append("no_intake")

    if facts.get("urination") == "none":
        preds.append("no_urine")

    return preds


def concern_predicates(obs, facts):
    concerns = []

    # --- danger ---
    if "lethargy" in obs:
        concerns.append("danger_red_flag")

    # --- dehydration scoring ---
    dehydration_score = 0

    if "no_urine" in obs:
        dehydration_score += 2

    if "no_intake" in obs:
        dehydration_score += 2

    if "repeated_vomiting" in obs:
        dehydration_score += 1

    if "poor_intake" in obs:
        dehydration_score += 1

    if dehydration_score >= 2:
        concerns.append("dehydration_concern")

    # --- fever amplification ---
    elif dehydration_score == 1 and "fever_present" in obs:
        if "no_urine" not in obs and facts.get("urination") == "normal":
            pass  # normal urination cancels mild dehydration concern
        else:
            concerns.append("dehydration_concern")

    return concerns


def decide(concerns, obs, facts):
    # --- ER ---
    if "danger_red_flag" in concerns:
        return "er_now"

    # --- URGENT ---
    if "dehydration_concern" in concerns:
        return "urgent_eval"

    # --- SAFE (must have positive evidence) ---
    safe_conditions = (
        "lethargy" not in obs and
        "no_urine" not in obs and
        "no_intake" not in obs and
        "repeated_vomiting" not in obs
    )

    sufficient_info = (
        facts.get("alert") == "normal" and
        facts.get("urination") == "normal"
    )

    if safe_conditions and sufficient_info:
        return "home_monitor"

    # --- fallback ---
    return "unsupported"


def rules_agent(state):
    facts = state["facts"]

    # --- Layer 1: observation ---
    obs = observation_predicates(facts)

    # --- Layer 2: concern ---
    concerns = concern_predicates(obs, facts)

    # --- Layer 3: decision ---
    decision = decide(concerns, obs, facts)

    # --- trace ---
    rules_triggered = []

    # observation trace
    if "lethargy" in obs:
        rules_triggered.append("obs:lethargy")

    if "no_urine" in obs:
        rules_triggered.append("obs:no_urine")

    if "no_intake" in obs:
        rules_triggered.append("obs:no_intake")

    if "repeated_vomiting" in obs:
        rules_triggered.append("obs:repeated_vomiting")

    if "poor_intake" in obs:
        rules_triggered.append("obs:poor_intake")

    if "fever_present" in obs:
        rules_triggered.append("obs:fever_present")

    # concern trace
    if "danger_red_flag" in concerns:
        rules_triggered.append("concern:danger_red_flag")

    if "dehydration_concern" in concerns:
        rules_triggered.append("concern:dehydration_concern")

    # safe trace
    if decision == "home_monitor":
        rules_triggered.append("safe:no_red_flags")

    # decision trace
    rules_triggered.append(f"decision:{decision}")

    # update state
    state["observation_predicates"] = obs
    state["concern_predicates"] = concerns
    state["decision"] = decision
    state["rules_triggered"] = rules_triggered

    return state