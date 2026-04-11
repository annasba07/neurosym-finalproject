"""Unified ClinicalState for the CareTrace pipeline.

Consolidation of all four team modules:
- Yoko's interpretation agent (raw extraction fields, Pydantic schema)
- Alex's rules engine (nested facts dict, alert/intake/urination vocabulary)
- David's knowledge_graph_agent (grounded concepts, KG red flags, sctids)
- Orchestrator (graph wiring, phase tracking, explanation outputs)

Conventions:
- state["facts"] is the symbolic fact layer Alex's rules read. Keys use
  Alex's vocabulary: alert=normal|reduced, intake=normal|reduced|none,
  urination=normal|none, vomiting=once|repeated, fever=yes|no,
  breathing=normal|difficulty, seizure=yes|no, rash=yes|no.
- Raw numeric/typed fields (age_months, temperature_f, current_medication)
  live at the top level because they're used by red-flag fallbacks and
  by the KG adapter for dosing.
- raw_symptoms is the free-text list the KG grounds on.
- Dispositions are normalized to Alex's vocabulary:
  er_now | urgent_eval | home_monitor | unsupported
"""

from typing import Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# Fact keys Alex's rules read. Values use Alex's vocabulary.
FACT_KEYS = [
    "fever",       # yes | no
    "alert",       # normal | reduced
    "intake",      # normal | reduced | none
    "urination",   # normal | none
    "vomiting",    # once | repeated
    "breathing",   # normal | difficulty
    "seizure",     # yes | no
    "rash",        # yes | no
]


class ClinicalState(TypedDict):
    # -- Conversation ---------------------------------------------------------
    messages: Annotated[list, add_messages]
    turn: int

    # -- Symbolic facts (Alex's rules read these) -----------------------------
    facts: dict

    # -- Raw extracted fields (Yoko's extraction + KG lookups need these) -----
    age_months: Optional[int]
    temperature_f: Optional[float]
    fever_duration_days: Optional[float]
    current_medication: Optional[str]
    medication_last_dose: Optional[str]
    weight_kg: Optional[float]          # for KG dosing when available
    raw_symptoms: list                  # free-text mentions for KG grounding

    # -- Knowledge graph outputs ----------------------------------------------
    kg_backend: Optional[str]           # "neo4j" | "dict"
    grounded_concepts: list             # [{mention, concepts, all_sctids, grounded}]
    all_sctids: list                    # deduped flat list of sctids across mentions
    kg_red_flags: list                  # [{rule_id, description, disposition, source}]

    # -- Rules agent outputs (Alex's vocabulary) ------------------------------
    observation_predicates: list        # Alex's layer-1 predicates
    concern_predicates: list            # Alex's layer-2 predicates
    decision: Optional[str]             # Alex's raw decision before merge
    rules_triggered: list               # merged string trace (Alex's + KG + fallback)

    # -- Final unified disposition --------------------------------------------
    disposition: Optional[str]          # er_now | urgent_eval | home_monitor | unsupported | None

    # -- Missing-info + follow-up ---------------------------------------------
    missing_required: list
    follow_up_question: Optional[str]

    # -- Explanation outputs --------------------------------------------------
    explanation: Optional[str]
    key_positives: list
    key_negatives: list

    # -- Control --------------------------------------------------------------
    is_complete: bool
    phase: str                          # "intake" | "triage" | "plan_ready"


# Fields the safety layer needs before it will issue a home_monitor disposition.
# Red-flag short-circuits bypass this list.
REQUIRED_FACTS_FOR_HOME = ["alert", "breathing", "intake", "urination"]

# Priority-ordered follow-up questions for missing facts.
FACT_QUESTIONS = {
    "alert":      "Is your child awake and responding normally when you talk to them?",
    "breathing":  "Is your child having any trouble breathing or breathing fast?",
    "intake":     "Is your child drinking fluids normally, a little, or refusing?",
    "urination":  "Has your child urinated in the last 8 hours?",
    "vomiting":   "Has your child been vomiting? If so, how many times?",
    "fever":      "Does your child have a fever?",
}

# Severity ranking for merging dispositions. Lower number = more severe.
DISPOSITION_SEVERITY = {
    "er_now":       0,
    "urgent_eval":  1,
    "home_monitor": 2,
    "unsupported":  3,
    None:           4,
}


def merge_dispositions(*candidates: Optional[str]) -> Optional[str]:
    """Return the most severe disposition from a list, ignoring None/unsupported
    when a real decision is present."""
    # Filter to candidates we've actually seen
    seen = [c for c in candidates if c is not None]
    if not seen:
        return None
    # Pick the worst (lowest severity rank)
    return min(seen, key=lambda c: DISPOSITION_SEVERITY.get(c, 99))


def initial_state() -> ClinicalState:
    """Return a blank ClinicalState for a new triage case."""
    return ClinicalState(
        messages=[],
        turn=0,
        facts={},
        age_months=None,
        temperature_f=None,
        fever_duration_days=None,
        current_medication=None,
        medication_last_dose=None,
        weight_kg=None,
        raw_symptoms=[],
        kg_backend=None,
        grounded_concepts=[],
        all_sctids=[],
        kg_red_flags=[],
        observation_predicates=[],
        concern_predicates=[],
        decision=None,
        rules_triggered=[],
        disposition=None,
        missing_required=[],
        follow_up_question=None,
        explanation=None,
        key_positives=[],
        key_negatives=[],
        is_complete=False,
        phase="intake",
    )
