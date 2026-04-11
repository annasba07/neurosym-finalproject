"""Safety agent — runs Alex's rules engine and merges with KG red flags.

The canonical rules engine is `src.rules.rules_agent` — a plain-Python
3-layer pipeline (observation → concern → decision) on the nested
facts dict. Alex's rules cover the dehydration/alertness/intake story.

KG red flags (from knowledge.py) cover what Alex's rules don't:
infant fever, seizure, breathing difficulty, very high fever, rash+fever.

IMPORTANT: KG red flags are advisory, not authoritative. Because the
Neo4j synonym matcher uses substring containment, grounding "fever"
can spuriously pull in "febrile seizure" and similar adjacent concepts.
We therefore validate each KG red flag against the actual facts before
applying it — this prevents false-positive escalations while still
honoring legitimate red flags (e.g., infant fever, real lethargy, etc.).

Final disposition = most severe across Alex's decision and validated
red flags (both KG-sourced and fallback-sourced).
"""

from caretrace.state import (
    ClinicalState,
    REQUIRED_FACTS_FOR_HOME,
    merge_dispositions,
)

# Alex's rules engine lives at repo root under src/
from src.rules import rules_agent


# ── KG red-flag validators ────────────────────────────────────────────────────
# Each entry maps rule_id → predicate(state) → bool. The predicate must
# return True for the flag to be applied. Unknown rule_ids default to
# "apply only if fallback-sourced" (see _validate_red_flag below).

def _rf_infant_fever(state: dict) -> bool:
    age = state.get("age_months")
    return age is not None and age < 3 and state.get("facts", {}).get("fever") == "yes"


def _rf_febrile_seizure(state: dict) -> bool:
    return state.get("facts", {}).get("seizure") == "yes"


def _rf_petechial_rash(state: dict) -> bool:
    f = state.get("facts", {})
    return f.get("rash") == "yes" and f.get("fever") == "yes"


def _rf_neck_stiffness(state: dict) -> bool:
    # We don't extract neck stiffness into facts yet; treat as never-fires
    # unless an extension writes it. This prevents spurious matches.
    return state.get("facts", {}).get("neck_stiffness") == "yes"


def _rf_lethargy(state: dict) -> bool:
    return state.get("facts", {}).get("alert") == "reduced"


def _rf_respiratory_distress(state: dict) -> bool:
    return state.get("facts", {}).get("breathing") == "difficulty"


def _rf_severe_dehydration(state: dict) -> bool:
    f = state.get("facts", {})
    return f.get("urination") == "none" and f.get("intake") == "none"


def _rf_fever_persistent(state: dict) -> bool:
    dur = state.get("fever_duration_days")
    return dur is not None and dur > 5


def _rf_vomiting_no_intake(state: dict) -> bool:
    f = state.get("facts", {})
    return f.get("vomiting") == "repeated" and f.get("intake") == "none"


def _rf_moderate_dehydration(state: dict) -> bool:
    return state.get("facts", {}).get("urination") == "none"


# Map David's rule_ids → gate predicates
_RED_FLAG_VALIDATORS = {
    "RF_001": _rf_infant_fever,
    "RF_002": _rf_febrile_seizure,
    "RF_003": _rf_petechial_rash,
    "RF_004": _rf_neck_stiffness,
    "RF_005": _rf_lethargy,
    "RF_006": _rf_respiratory_distress,
    "RF_007": _rf_severe_dehydration,
    "RF_008": _rf_fever_persistent,
    "RF_009": _rf_vomiting_no_intake,
    "RF_010": _rf_moderate_dehydration,
}


def _validate_red_flag(flag: dict, state: dict) -> bool:
    """Return True if this red flag should be applied given current state.

    - Fallback-sourced red flags are trusted as-is (they've already checked
      the facts before being emitted).
    - KG/Neo4j-sourced red flags must pass a rule_id-specific predicate.
      Unknown rule_ids are rejected to fail safe against the substring-
      matching false positives.
    """
    if flag.get("source") == "fallback":
        return True

    validator = _RED_FLAG_VALIDATORS.get(flag.get("rule_id", ""))
    if validator is None:
        return False
    return validator(state)


def _build_key_findings(facts: dict, rules_triggered: list, red_flags: list) -> tuple[list, list]:
    """Build caregiver-facing positive/negative bullets from the symbolic trace."""
    positives: list[str] = []
    negatives: list[str] = []

    # Positive = concerning findings
    if facts.get("fever") == "yes":
        positives.append("Fever present")
    if facts.get("alert") == "reduced":
        positives.append("Child is lethargic / less responsive than normal")
    if facts.get("intake") == "none":
        positives.append("Child is refusing fluids")
    elif facts.get("intake") == "reduced":
        positives.append("Child's fluid intake is reduced")
    if facts.get("urination") == "none":
        positives.append("No urination in the last 8 hours")
    if facts.get("vomiting") == "repeated":
        positives.append("Repeated vomiting")
    if facts.get("breathing") == "difficulty":
        positives.append("Difficulty breathing")
    if facts.get("seizure") == "yes":
        positives.append("Seizure reported")
    if facts.get("rash") == "yes":
        positives.append("New rash")

    # Red-flag descriptions are always positive findings worth surfacing
    for flag in red_flags:
        desc = flag.get("description")
        if desc and desc not in positives:
            positives.append(desc)

    # Negative = reassuring findings
    if facts.get("alert") == "normal":
        negatives.append("Child is awake and responding normally")
    if facts.get("breathing") == "normal":
        negatives.append("Breathing is normal")
    if facts.get("intake") == "normal":
        negatives.append("Child is drinking fluids normally")
    if facts.get("urination") == "normal":
        negatives.append("Child is urinating normally")

    return positives, negatives


def evaluate_rules(state: ClinicalState) -> dict:
    """LangGraph node: run rules engine + merge KG red flags into disposition.

    Key behavior:
    - Red flags short-circuit to er_now regardless of missing facts.
    - For home_monitor, requires all REQUIRED_FACTS_FOR_HOME to be present.
    - If missing required facts and no red flags, disposition is None
      (routes to ask_followup).
    """
    facts = state.get("facts") or {}
    raw_red_flags = state.get("kg_red_flags") or []

    # Validate each red flag against the actual facts. KG-sourced flags from
    # substring matching can be false positives ("fever" → "febrile seizure"),
    # so we gate them behind per-rule_id predicates. Fallback flags pass
    # through unchanged — they already checked facts before being emitted.
    red_flags = [f for f in raw_red_flags if _validate_red_flag(f, state)]

    # ── 1. Alex's rules engine ────────────────────────────────────────────────
    # rules_agent mutates and returns a dict; wrap in a shim so we pass only
    # the nested facts (it reads state["facts"]).
    alex_out = rules_agent({"facts": facts})
    alex_decision: str = alex_out["decision"]
    obs_preds: list = alex_out["observation_predicates"]
    concern_preds: list = alex_out["concern_predicates"]
    alex_trace: list = alex_out["rules_triggered"]

    # ── 2. KG red-flag disposition (most severe, validated only) ─────────────
    kg_disposition = None
    if red_flags:
        priority = {"er_now": 0, "urgent_eval": 1, "home_monitor": 2}
        kg_disposition = min(
            red_flags,
            key=lambda r: priority.get(r.get("disposition", ""), 99),
        )["disposition"]

    # ── 3. Merge: worst wins, BUT we downgrade 'unsupported' when a real
    #    KG signal exists (infant fever is decisive even without fact info). ──
    merged = merge_dispositions(alex_decision, kg_disposition)

    # ── 4. Check missing required facts for safe home_monitor ───────────────
    missing = [k for k in REQUIRED_FACTS_FOR_HOME if k not in facts]

    # If Alex voted home_monitor but we're missing required facts, it's not safe
    # to commit — route to ask_followup unless red flags also fired.
    if merged == "home_monitor" and missing:
        merged = "unsupported"

    # If everything is still 'unsupported' (no red flags, not enough data), we
    # return disposition=None so the graph routes to ask_followup.
    disposition: str | None = merged if merged != "unsupported" else None

    # ── 5. Build rules_triggered merged trace ────────────────────────────────
    rules_triggered: list = list(alex_trace)
    for flag in red_flags:
        rules_triggered.append(
            f"{flag.get('source', 'kg')}:{flag['rule_id']}→{flag['disposition']}"
        )

    # ── 6. Key positives/negatives for the explanation layer ────────────────
    positives, negatives = _build_key_findings(facts, rules_triggered, red_flags)

    # ── 7. Phase ─────────────────────────────────────────────────────────────
    if disposition is not None:
        phase = "plan_ready"
    elif missing:
        phase = "intake"
    else:
        phase = "triage"

    return {
        "decision":               alex_decision,
        "observation_predicates": obs_preds,
        "concern_predicates":     concern_preds,
        "rules_triggered":        rules_triggered,
        "disposition":            disposition,
        "missing_required":       missing,
        "key_positives":          positives,
        "key_negatives":          negatives,
        "phase":                  phase,
    }
