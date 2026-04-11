"""End-to-end test of CareTrace rules + KG merge on clinical scenarios.

Bypasses the LLM extraction layer and feeds facts directly into
state["facts"] in Alex's vocabulary. Verifies the deterministic core
(Alex's rules + fallback KG red flags) produces the right dispositions.

These tests force the KG adapter into dict mode so they don't depend on
what's in the live Neo4j instance. The live graph is covered by
run_live.py instead.

Run with: python -m scenarios.test_scenarios
"""

import os

# Force dict-only KG BEFORE importing caretrace.agents.knowledge so the
# module-level adapter initializes with the right backend.
os.environ["CARETRACE_KG_BACKEND"] = "dict"

from caretrace.state import initial_state  # noqa: E402
from caretrace.agents.knowledge import normalize  # noqa: E402
from caretrace.agents.safety import evaluate_rules  # noqa: E402


def _run_turn(state: dict, turn_num: int, label: str, updates: dict) -> dict:
    """Apply raw + facts updates, run KG + rules, print results."""
    print(f"\n--- Turn {turn_num}: {label} ---")
    print(f"  Updates: {updates}")

    # Merge raw fields
    for key, val in updates.items():
        if key == "facts":
            state["facts"] = {**(state.get("facts") or {}), **val}
        else:
            state[key] = val

    # Derive raw_symptoms deterministically from the facts so the KG has
    # something to ground against without the LLM in the loop.
    state["raw_symptoms"] = _symptoms_from_facts(state)

    # Run KG normalization (adapter + fallback red flags)
    state.update(normalize(state))

    # Run rules
    state.update(evaluate_rules(state))

    print(f"  KG backend:  {state.get('kg_backend')}")
    print(f"  Disposition: {state.get('disposition') or 'undecided'}")
    print(f"  Phase:       {state.get('phase')}")

    missing = state.get("missing_required", [])
    if missing:
        print(f"  Missing:     {missing}")

    obs = state.get("observation_predicates", [])
    if obs:
        print(f"  Observations: {obs}")

    concerns = state.get("concern_predicates", [])
    if concerns:
        print(f"  Concerns:    {concerns}")

    rules = state.get("rules_triggered", [])
    if rules:
        print("  Rule trace:")
        for r in rules:
            print(f"    - {r}")

    red_flags = state.get("kg_red_flags", [])
    if red_flags:
        print("  Red flags:")
        for f in red_flags:
            print(f"    [{f.get('source', 'kg')}] {f['rule_id']} → {f['disposition']}")

    return state


def _symptoms_from_facts(state: dict) -> list[str]:
    """Deterministic fact → canonical-phrase map for KG grounding in tests."""
    facts = state.get("facts") or {}
    symptoms: list[str] = []
    if facts.get("fever") == "yes":
        temp = state.get("temperature_f")
        symptoms.append("high fever" if (temp and temp >= 104.0) else "fever")
    if facts.get("alert") == "reduced":
        symptoms.append("lethargy")
    if facts.get("intake") == "none":
        symptoms.append("reduced fluid intake")
        symptoms.append("dehydration")
    elif facts.get("intake") == "reduced":
        symptoms.append("reduced fluid intake")
    if facts.get("urination") == "none":
        symptoms.append("decreased urine output")
    if facts.get("vomiting") == "repeated":
        symptoms.append("repeated vomiting")
    elif facts.get("vomiting") == "once":
        symptoms.append("vomiting")
    if facts.get("breathing") == "difficulty":
        symptoms.append("breathing difficulty")
    if facts.get("seizure") == "yes":
        symptoms.append("seizure")
    if facts.get("rash") == "yes":
        symptoms.append("rash")
    return symptoms


# ── Scenarios ─────────────────────────────────────────────────────────────────

def scenario_1_home():
    """Scenario 1: 6yo, moderate fever, alert, drinking normally → home_monitor.

    Note: Alex's rules amplify any `intake=reduced + fever` to urgent_eval
    via fever-boosted dehydration scoring, so this home-path test uses
    intake=normal. Scenario 1b below covers the reduced-intake + fever case.
    """
    print("=" * 70)
    print("SCENARIO 1: Moderate fever, alert, drinking normally → expect home_monitor")
    print("=" * 70)

    state = initial_state()

    state = _run_turn(state, 1, "Initial vague complaint", {
        "age_months": 72,
        "facts": {"fever": "yes", "vomiting": "once"},
    })
    assert state["disposition"] is None, "Should still be undecided (missing facts)"

    state = _run_turn(state, 2, "Vitals + responsiveness + normal intake", {
        "temperature_f": 101.8,
        "facts": {"alert": "normal", "breathing": "normal", "intake": "normal"},
    })
    assert "urination" in state.get("missing_required", []), \
        "Should still need urination info"

    state = _run_turn(state, 3, "Urine info + medications", {
        "current_medication": "amoxicillin",
        "facts": {"urination": "normal"},
    })

    print("\n" + "=" * 70)
    print(f"FINAL DISPOSITION: {state['disposition']}")
    assert state["disposition"] == "home_monitor", \
        f"Expected 'home_monitor', got '{state['disposition']}'"
    print("PASS — Scenario 1 correctly routed to home_monitor")
    print("=" * 70)
    return state


def scenario_1b_urgent_eval():
    """Scenario 1b: same vitals but reduced intake → Alex's rules → urgent_eval.

    This documents Alex's more conservative dehydration-scoring behavior:
    poor_intake + fever_present fires dehydration_concern even without
    urination or vomiting signals.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 1b: Reduced intake + fever → expect urgent_eval")
    print("=" * 70)

    state = initial_state()
    state = _run_turn(state, 1, "Reduced intake with fever", {
        "age_months": 72,
        "temperature_f": 101.8,
        "facts": {
            "fever":     "yes",
            "alert":     "normal",
            "breathing": "normal",
            "intake":    "reduced",
            "urination": "normal",
        },
    })

    print(f"\nFINAL DISPOSITION: {state['disposition']}")
    assert state["disposition"] == "urgent_eval", \
        f"Expected 'urgent_eval' (dehydration fever amplification), got '{state['disposition']}'"
    print("PASS — Scenario 1b correctly routes reduced-intake to urgent_eval")
    print("=" * 70)


def scenario_2_er():
    """Scenario 2: 6yo, high fever, reduced alertness, not drinking → er_now."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: High fever, reduced alert, not drinking → expect er_now")
    print("=" * 70)

    state = initial_state()

    state = _run_turn(state, 1, "Initial worried complaint", {
        "age_months": 72,
        "facts": {"fever": "yes", "vomiting": "once"},
    })

    state = _run_turn(state, 2, "Severe vitals — lethargy + no intake", {
        "temperature_f": 103.5,
        "facts": {
            "alert":     "reduced",
            "intake":    "none",
            "breathing": "normal",
        },
    })
    # Alex's rules should fire danger_red_flag on lethargy → er_now
    assert state["disposition"] == "er_now", \
        f"Should escalate to er_now on reduced alertness, got {state['disposition']}"

    state = _run_turn(state, 3, "Urine info confirms dehydration", {
        "facts": {"urination": "none"},
    })

    print("\n" + "=" * 70)
    print(f"FINAL DISPOSITION: {state['disposition']}")
    assert state["disposition"] == "er_now", \
        f"Expected 'er_now', got '{state['disposition']}'"
    print("PASS — Scenario 2 correctly routed to er_now")
    print("=" * 70)
    return state


def scenario_3_seizure_short_circuit():
    """Fallback red flag: seizure immediately → er_now."""
    print("\n" + "=" * 70)
    print("SCENARIO 3: Seizure fallback red flag → expect er_now")
    print("=" * 70)

    state = initial_state()
    state = _run_turn(state, 1, "Seizure reported with minimal info", {
        "age_months": 24,
        "facts": {"fever": "yes", "seizure": "yes"},
    })

    assert state["disposition"] == "er_now", \
        f"Seizure should short-circuit to er_now, got {state['disposition']}"
    print("PASS — Seizure fallback correctly short-circuits to er_now")
    print("=" * 70)


def scenario_4_infant_fever():
    """Fallback red flag: infant (<3mo) with any fever → er_now."""
    print("\n" + "=" * 70)
    print("SCENARIO 4: Infant <3 months with fever → expect er_now")
    print("=" * 70)

    state = initial_state()
    state = _run_turn(state, 1, "2-month-old with fever", {
        "age_months": 2,
        "temperature_f": 101.0,
        "facts": {"fever": "yes"},
    })

    assert state["disposition"] == "er_now", \
        f"Infant fever should short-circuit to er_now, got {state['disposition']}"
    print("PASS — Infant fever correctly routed to er_now")
    print("=" * 70)


def scenario_5_breathing_difficulty():
    """Fallback red flag: breathing difficulty → er_now."""
    print("\n" + "=" * 70)
    print("SCENARIO 5: Breathing difficulty → expect er_now")
    print("=" * 70)

    state = initial_state()
    state = _run_turn(state, 1, "Child with breathing trouble", {
        "age_months": 48,
        "temperature_f": 102.0,
        "facts": {"fever": "yes", "breathing": "difficulty", "alert": "normal"},
    })

    assert state["disposition"] == "er_now", \
        f"Breathing difficulty should short-circuit to er_now, got {state['disposition']}"
    print("PASS — Breathing difficulty correctly routes to er_now")
    print("=" * 70)


def main():
    scenario_1_home()
    scenario_1b_urgent_eval()
    scenario_2_er()
    scenario_3_seizure_short_circuit()
    scenario_4_infant_fever()
    scenario_5_breathing_difficulty()

    print("\n" + "=" * 70)
    print("ALL SCENARIOS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
