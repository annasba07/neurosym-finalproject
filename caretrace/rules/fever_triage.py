"""Deterministic triage rules from the Seattle Children's Fever CPG.

Uses pyDatalog for executable clinical practice guidelines.
Follows the professor's Lecture 9 patterns: EDB facts, IDB rules,
because() for explanation hooks, escalation_level for disposition.

IMPORTANT: pyDatalog's create_terms injects names into the *caller's*
local scope by mutating f_locals. Inside a function, this is a no-op
because Python uses slot-based fast locals. Therefore terms and rules
must be defined at MODULE level (where f_locals is the module globals).

To support multiple evaluations without calling pyDatalog.clear() (which
would also wipe the rules), we use a fresh case_id per evaluation.
"""

import itertools
from pyDatalog import pyDatalog


# =============================================================================
# MODULE-LEVEL TERM DECLARATIONS
# =============================================================================

pyDatalog.create_terms(
    # Variables
    "X, Y, T, Val, Age, Med, Reason, Level, Action",
    # Patient EDB facts
    "age_months, temperature_f, fever, vomiting, alert, drinking",
    "urine_hours, breathing_issue, medications, seizure, rash, local_context",
    # Derived observations (IDB)
    "sign, symptom, has_red_flag, dehydration_concern",
    "escalation_level, needs_escalation",
    # Medication logic (IDB)
    "med_allowed, med_avoid, recommended_med, forbidden_med",
    # Actions (IDB)
    "recommended_action, forbidden_action",
    # Explanation hooks (IDB)
    "because, explain_line",
    # Anchor
    "case",
)

# =============================================================================
# MODULE-LEVEL RULE DEFINITIONS (IDB)
# =============================================================================

# Anchor predicate — binds the case variable
case(X) <= age_months(X, Age)

# Fever severity
sign(X, "fever") <= fever(X, "yes")
sign(X, "high_fever") <= (temperature_f(X, T) & (T >= 104.0))
sign(X, "moderate_fever") <= (
    temperature_f(X, T) & (T >= 100.4) & (T < 104.0)
)

# Dehydration signals
symptom(X, "low_urine") <= (urine_hours(X, Val) & (Val >= 8))
symptom(X, "no_drinking") <= drinking(X, "no")
symptom(X, "some_drinking") <= drinking(X, "some")

# Dehydration concern (composite)
dehydration_concern(X) <= symptom(X, "low_urine")
dehydration_concern(X) <= symptom(X, "no_drinking")

# Vomiting signals
symptom(X, "vomited_once") <= vomiting(X, "once")
symptom(X, "repeated_vomiting") <= vomiting(X, "repeated")

# Red flags
has_red_flag(X) <= seizure(X, "yes")
has_red_flag(X) <= alert(X, "no")
has_red_flag(X) <= breathing_issue(X, "yes")
has_red_flag(X) <= (age_months(X, Age) & (Age < 3) & fever(X, "yes"))
has_red_flag(X) <= sign(X, "high_fever")

# Red flag → ER escalation
escalation_level(X, "er") <= seizure(X, "yes")
escalation_level(X, "er") <= alert(X, "no")
escalation_level(X, "er") <= breathing_issue(X, "yes")
escalation_level(X, "er") <= (
    age_months(X, Age) & (Age < 3) & fever(X, "yes")
)
escalation_level(X, "er") <= sign(X, "high_fever")

# Severe dehydration combination → ER (range-restricted via case)
escalation_level(X, "er") <= (
    case(X) & symptom(X, "no_drinking") & symptom(X, "low_urine")
)

# Concern flags → Urgent (stratified after has_red_flag is fully defined)
escalation_level(X, "urgent") <= (
    case(X) & dehydration_concern(X) & ~has_red_flag(X)
)
escalation_level(X, "urgent") <= (
    case(X) & symptom(X, "repeated_vomiting") & ~has_red_flag(X)
)

# Home management — only if no red flags and no urgent concerns
escalation_level(X, "home") <= (
    case(X) & ~has_red_flag(X) & ~dehydration_concern(X)
    & ~symptom(X, "repeated_vomiting")
)

# Medication safety
med_allowed(X, "acetaminophen") <= (age_months(X, Age) & (Age >= 3))
med_allowed(X, "ibuprofen") <= (
    case(X) & age_months(X, Age) & (Age >= 6) & ~dehydration_concern(X)
)
med_avoid(X, "ibuprofen") <= dehydration_concern(X)
forbidden_med(X, "aspirin") <= case(X)
recommended_med(X, "acetaminophen") <= (
    sign(X, "fever") & med_allowed(X, "acetaminophen")
)

# Actions
recommended_action(X, "go_to_er_now") <= escalation_level(X, "er")
recommended_action(X, "urgent_evaluation") <= (
    case(X) & escalation_level(X, "urgent") & ~escalation_level(X, "er")
)
recommended_action(X, "home_monitoring") <= escalation_level(X, "home")

# Explanation hooks — because(case, conclusion, evidence)
because(X, "er_referral", "not_alert") <= alert(X, "no")
because(X, "er_referral", "seizure") <= seizure(X, "yes")
because(X, "er_referral", "breathing_difficulty") <= breathing_issue(X, "yes")
because(X, "er_referral", "high_fever_over_104") <= sign(X, "high_fever")
because(X, "er_referral", "infant_under_3mo_with_fever") <= (
    age_months(X, Age) & (Age < 3) & fever(X, "yes")
)

because(X, "dehydration_concern", "no_urine_8h") <= symptom(X, "low_urine")
because(X, "dehydration_concern", "not_drinking") <= symptom(X, "no_drinking")

because(X, "home_safe", "alert_and_responsive") <= alert(X, "yes")
because(X, "home_safe", "drinking_fluids") <= drinking(X, "yes")
because(X, "home_safe", "no_breathing_issues") <= breathing_issue(X, "no")
because(X, "home_safe", "recent_urination") <= (
    urine_hours(X, Val) & (Val < 8)
)

# Canonical explain lines
explain_line(X, "escalation", Reason) <= because(X, "er_referral", Reason)
explain_line(X, "dehydration", Reason) <= (
    because(X, "dehydration_concern", Reason)
)
explain_line(X, "safety", Reason) <= because(X, "home_safe", Reason)


# =============================================================================
# SENTINEL FACTS — ensure every EDB predicate has at least one fact so
# pyDatalog can resolve queries even when some clinical fields are missing.
# These use a sentinel case ID that no real evaluation will ever query.
# =============================================================================

_SENTINEL = "__edb_sentinel__"
+age_months(_SENTINEL, -1)
+temperature_f(_SENTINEL, -1.0)
+fever(_SENTINEL, "n/a")
+vomiting(_SENTINEL, "n/a")
+alert(_SENTINEL, "n/a")
+drinking(_SENTINEL, "n/a")
+urine_hours(_SENTINEL, -1)
+breathing_issue(_SENTINEL, "n/a")
+medications(_SENTINEL, "n/a")
+seizure(_SENTINEL, "n/a")
+rash(_SENTINEL, "n/a")
+local_context(_SENTINEL, "n/a")


# =============================================================================
# CASE ID GENERATOR — fresh ID per evaluation so old facts don't interfere
# =============================================================================

_case_counter = itertools.count()


def _next_case_id() -> str:
    return f"case_{next(_case_counter)}"


# =============================================================================
# FACT ASSERTION — runs at module level via inline lambda binding
# =============================================================================

def _assert_facts(c: str, state: dict) -> None:
    """Assert EDB facts from clinical state into pyDatalog."""
    age = state.get("age_months")
    if age is not None:
        +age_months(c, age)
    else:
        +age_months(c, 72)  # default 6yo — anchors the case

    temp = state.get("temperature_f")
    if temp is not None:
        +temperature_f(c, float(temp))

    f = state.get("fever")
    if f is not None:
        +fever(c, f)
    elif temp is not None and temp >= 100.4:
        +fever(c, "yes")

    if state.get("vomiting") is not None:
        +vomiting(c, state["vomiting"])
    if state.get("alert") is not None:
        +alert(c, state["alert"])
    if state.get("drinking") is not None:
        +drinking(c, state["drinking"])
    if state.get("urine_hours") is not None:
        +urine_hours(c, int(state["urine_hours"]))
    if state.get("breathing_issue") is not None:
        +breathing_issue(c, state["breathing_issue"])
    if state.get("medications") is not None:
        +medications(c, state["medications"])
    if state.get("seizure") is not None:
        +seizure(c, state["seizure"])
    if state.get("rash") is not None:
        +rash(c, state["rash"])
    if state.get("local_context") is not None:
        +local_context(c, state["local_context"])


def _query_disposition(c: str) -> str | None:
    """Determine disposition with priority: er > urgent > home."""
    if escalation_level(c, "er"):
        return "er"
    if escalation_level(c, "urgent"):
        return "urgent"
    if escalation_level(c, "home"):
        return "home"
    return None


def _query_rules_fired(c: str) -> list[dict]:
    """Collect explanation traces."""
    fired = []
    for category in ("escalation", "dehydration", "safety"):
        result = explain_line(c, category, Reason)
        if result:
            for row in result.data:
                fired.append({
                    "rule_id": f"{category}_{row[0]}",
                    "category": category,
                    "reason": row[0],
                })
    return fired


def _query_medication(c: str, state: dict) -> dict:
    """Determine medication safety decision."""
    if state.get("fever") != "yes":
        return {
            "allowed": False,
            "reason": "No fever — medication not indicated",
        }

    allowed_meds = []
    if med_allowed(c, "acetaminophen"):
        allowed_meds.append("acetaminophen")
    if med_allowed(c, "ibuprofen"):
        allowed_meds.append("ibuprofen")

    avoided = []
    if med_avoid(c, "ibuprofen"):
        avoided.append("ibuprofen (dehydration risk)")

    recommended = None
    rec_result = recommended_med(c, X)
    if rec_result and rec_result.data:
        recommended = rec_result.data[0][0]

    parts = []
    if recommended:
        parts.append(f"Recommended: {recommended}")
    if avoided:
        parts.append(f"Avoid: {', '.join(avoided)}")
    parts.append("Do not switch between medications without doctor guidance")

    return {
        "allowed": bool(allowed_meds),
        "allowed_meds": allowed_meds,
        "avoided": avoided,
        "recommended": recommended,
        "reason": ". ".join(parts) + ".",
    }


def _build_key_findings(state: dict) -> tuple[list[str], list[str]]:
    """Build key positives and negatives for the explanation layer."""
    positives = []
    negatives = []

    if state.get("fever") == "yes":
        temp = state.get("temperature_f")
        if temp:
            positives.append(f"Fever ({temp}°F)")
        else:
            positives.append("Fever reported")

    if state.get("vomiting") in ("once", "repeated"):
        positives.append(f"Vomiting ({state['vomiting']})")

    if state.get("alert") == "no":
        positives.append("Reduced responsiveness")
    elif state.get("alert") == "yes":
        negatives.append("Alert and responsive")

    if state.get("drinking") == "no":
        positives.append("Not drinking fluids")
    elif state.get("drinking") == "yes":
        negatives.append("Drinking fluids")

    urine = state.get("urine_hours")
    if urine is not None and urine >= 8:
        positives.append(f"No urination for {urine}+ hours")
    elif urine is not None and urine < 8:
        negatives.append("Recent urination")

    if state.get("breathing_issue") == "yes":
        positives.append("Breathing difficulty")
    elif state.get("breathing_issue") == "no":
        negatives.append("No breathing difficulty")

    if state.get("seizure") == "yes":
        positives.append("Seizure reported")
    elif state.get("seizure") == "no":
        negatives.append("No seizure")

    return positives, negatives


def _build_go_now_thresholds(disposition: str | None) -> list[str]:
    """Build explicit go-now thresholds for safety netting."""
    thresholds = [
        "Child becomes hard to wake or unresponsive",
        "Breathing becomes fast or difficult",
        "Seizure occurs",
        "Fever rises above 104°F",
    ]
    if disposition == "home":
        thresholds.extend([
            "Vomiting continues repeatedly and cannot keep fluids down",
            "No urination for 8+ hours",
            "Child stops drinking fluids entirely",
        ])
    return thresholds


def _build_overnight_plan(disposition: str | None, state: dict) -> list[str]:
    """Build overnight care plan for home management."""
    if disposition != "home":
        return []
    plan = [
        "Offer small, frequent sips of clear fluids (water, electrolyte solution)",
        "Dress in comfortable, light clothing",
        "Let child rest — do not wake for fever medicine",
    ]
    if state.get("fever") == "yes":
        plan.append(
            "For fever discomfort: use ONE fever medicine at correct weight-based dose"
        )
    plan.extend([
        "Monitor for escalation signs (listed above)",
        "Follow up with pediatrician next day",
    ])
    return plan


class TriageRuleEngine:
    """Evaluates clinical state against Fever CPG rules using pyDatalog."""

    def evaluate(self, state: dict) -> dict:
        """Run triage rules against the current clinical state.

        Returns a dict with disposition, rules_fired, key_positives,
        key_negatives, medication_decision, go_now_thresholds, overnight_plan.
        """
        # Use a fresh case_id so we don't pollute across evaluations
        c = _next_case_id()
        _assert_facts(c, state)

        disposition = _query_disposition(c)
        positives, negatives = _build_key_findings(state)

        return {
            "disposition": disposition,
            "rules_fired": _query_rules_fired(c),
            "key_positives": positives,
            "key_negatives": negatives,
            "medication_decision": _query_medication(c, state),
            "go_now_thresholds": _build_go_now_thresholds(disposition),
            "overnight_plan": _build_overnight_plan(disposition, state),
        }
