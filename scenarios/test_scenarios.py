"""End-to-end test of CareTrace rule engine on both clinical scenarios.

This bypasses the LLM extraction layer and feeds clinical state turn-by-turn
into the safety agent. It verifies the deterministic core (rules + KG)
produces the correct dispositions for both scenarios.

Run with: python -m scenarios.test_scenarios
"""

from caretrace.state import initial_state
from caretrace.agents.knowledge import normalize
from caretrace.agents.safety import evaluate_rules


def _run_turn(state: dict, turn_num: int, label: str, updates: dict) -> dict:
    """Apply state updates and run KG + rules. Print results."""
    print(f"\n--- Turn {turn_num}: {label} ---")
    print(f"  New facts: {updates}")

    state.update(updates)

    # Run KG normalization
    kg_out = normalize(state)
    state.update(kg_out)

    # Run rules
    safety_out = evaluate_rules(state)
    state.update(safety_out)

    print(f"  Disposition: {state.get('disposition') or 'undecided'}")
    print(f"  Phase: {state.get('phase')}")

    missing = state.get("missing_required", [])
    if missing:
        print(f"  Missing: {missing}")

    rules_fired = state.get("rules_fired", [])
    if rules_fired:
        print("  Rules fired:")
        for r in rules_fired:
            print(f"    [{r['category']}] {r['reason']}")

    concepts = state.get("snomed_concepts", [])
    if concepts:
        print(f"  SNOMED: {[c['term'] for c in concepts]}")

    return state


def scenario_1_home():
    """Scenario 1: 6yo with moderate fever, vomited once, sipping fluids → HOME."""
    print("=" * 65)
    print("SCENARIO 1: Moderate fever, alert, drinking → expect HOME")
    print("=" * 65)

    state = initial_state()

    # Turn 1: Initial complaint — vague
    state = _run_turn(state, 1, "Initial vague complaint", {
        "age_months": 72,
        "fever": "yes",
        "vomiting": "once",
    })
    assert state["disposition"] is None, "Should be undecided after vague turn"

    # Turn 2: Vital signs
    state = _run_turn(state, 2, "Vitals + responsiveness", {
        "temperature_f": 101.8,
        "alert": "yes",
        "breathing_issue": "no",
        "drinking": "some",
    })
    # Still missing urine info
    assert "urine_hours" in state.get("missing_required", []), "Should still need urine info"

    # Turn 3: Final details
    state = _run_turn(state, 3, "Medications + urine", {
        "medications": "amoxicillin",
        "urine_hours": 3,
    })

    print("\n" + "=" * 65)
    print(f"FINAL DISPOSITION: {state['disposition']}")
    assert state["disposition"] == "home", \
        f"Expected 'home', got '{state['disposition']}'"
    print("PASS — Scenario 1 correctly routed to home management")
    print("=" * 65)
    return state


def scenario_2_er():
    """Scenario 2: 6yo with high fever, lethargic, not drinking → ER."""
    print("\n" + "=" * 65)
    print("SCENARIO 2: High fever, not alert, not drinking → expect ER")
    print("=" * 65)

    state = initial_state()

    # Turn 1: Initial complaint
    state = _run_turn(state, 1, "Initial worried complaint", {
        "age_months": 72,
        "fever": "yes",
        "vomiting": "once",
    })

    # Turn 2: Severe vitals — should trigger ER red flags immediately
    state = _run_turn(state, 2, "High fever + not alert + not drinking", {
        "temperature_f": 103.5,
        "alert": "no",
        "drinking": "no",
        "breathing_issue": "no",
        "local_context": "stomach virus going around school",
    })
    # ER red flag should fire even without urine info
    assert state["disposition"] == "er", \
        f"Should escalate to ER on red flags, got {state['disposition']}"

    # Turn 3: Confirm with urine info
    state = _run_turn(state, 3, "Urine info confirms dehydration", {
        "urine_hours": 8,
    })

    print("\n" + "=" * 65)
    print(f"FINAL DISPOSITION: {state['disposition']}")
    assert state["disposition"] == "er", \
        f"Expected 'er', got '{state['disposition']}'"
    print("PASS — Scenario 2 correctly routed to ER")
    print("=" * 65)
    return state


def scenario_3_red_flag_short_circuit():
    """Smoke test: a red flag should immediately escalate even with minimal info."""
    print("\n" + "=" * 65)
    print("SCENARIO 3: Red flag short-circuit (seizure)")
    print("=" * 65)

    state = initial_state()
    state = _run_turn(state, 1, "Seizure reported with minimal info", {
        "age_months": 24,
        "fever": "yes",
        "seizure": "yes",
    })

    assert state["disposition"] == "er", \
        f"Seizure should immediately route to ER, got {state['disposition']}"
    print("PASS — Red flag correctly short-circuits to ER")
    print("=" * 65)


def scenario_4_infant_fever():
    """Smoke test: infant <3mo with any fever should go to ER."""
    print("\n" + "=" * 65)
    print("SCENARIO 4: Infant <3 months with fever → ER")
    print("=" * 65)

    state = initial_state()
    state = _run_turn(state, 1, "2-month-old with fever", {
        "age_months": 2,
        "temperature_f": 101.0,
        "fever": "yes",
    })

    assert state["disposition"] == "er", \
        f"Infant fever should be ER, got {state['disposition']}"
    print("PASS — Infant fever correctly routed to ER")
    print("=" * 65)


def main():
    scenario_1_home()
    scenario_2_er()
    scenario_3_red_flag_short_circuit()
    scenario_4_infant_fever()

    print("\n" + "=" * 65)
    print("ALL SCENARIOS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
