"""Safety agent — runs pyDatalog rules to determine disposition.

Wraps the TriageRuleEngine and checks whether enough information
has been gathered for a safe home disposition.
"""

from caretrace.state import ClinicalState, REQUIRED_FIELDS_FOR_HOME
from caretrace.rules.fever_triage import TriageRuleEngine

# Single engine instance — _setup() is called fresh each evaluate()
_engine = TriageRuleEngine()


def evaluate_rules(state: ClinicalState) -> dict:
    """LangGraph node: evaluate triage rules against current clinical state.

    Key behavior:
    - ER red flags short-circuit immediately regardless of missing fields.
    - For home/urgent, checks whether required fields are present.
    - Returns disposition, rules_fired, missing_required, etc.
    """
    result = _engine.evaluate(state)
    disposition = result["disposition"]

    # Check missing fields (only matters for home disposition)
    missing = []
    if disposition != "er":
        for field in REQUIRED_FIELDS_FOR_HOME:
            if state.get(field) is None:
                missing.append(field)

    # If missing critical fields and no ER flag, withhold home disposition
    if missing and disposition == "home":
        disposition = None  # Can't safely say "home" yet

    # Determine phase
    if disposition is not None:
        phase = "plan_ready"
    elif missing:
        phase = "intake"
    else:
        phase = "triage"

    return {
        "disposition": disposition,
        "rules_fired": result["rules_fired"],
        "missing_required": missing,
        "key_positives": result["key_positives"],
        "key_negatives": result["key_negatives"],
        "medication_decision": result["medication_decision"],
        "go_now_thresholds": result["go_now_thresholds"],
        "overnight_plan": result["overnight_plan"],
        "phase": phase,
    }
