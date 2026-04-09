"""Shared ClinicalState — the single contract all agents read/write."""

from typing import Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class ClinicalState(TypedDict):
    """State object that flows through the entire CareTrace pipeline.

    Every agent node reads from and writes back to this state.
    Fields start as None (unknown) and get populated as the conversation progresses.
    """

    # -- Conversation history --------------------------------------------------
    messages: Annotated[list, add_messages]
    turn: int

    # -- Extracted clinical facts (None = unknown) -----------------------------
    age_months: Optional[int]
    temperature_f: Optional[float]
    fever: Optional[str]                # "yes" | "no"
    vomiting: Optional[str]             # "none" | "once" | "repeated"
    alert: Optional[str]                # "yes" | "no"
    drinking: Optional[str]             # "yes" | "some" | "no"
    urine_hours: Optional[int]          # hours since last urination
    breathing_issue: Optional[str]      # "yes" | "no"
    medications: Optional[str]          # current medication name or "none"
    seizure: Optional[str]              # "yes" | "no"
    rash: Optional[str]                 # "yes" | "no"
    local_context: Optional[str]        # e.g. "stomach virus at school"

    # -- KG normalization outputs ----------------------------------------------
    snomed_concepts: list               # [{term, sctid, ancestors}, ...]

    # -- Safety logic outputs --------------------------------------------------
    rules_fired: list                   # [{rule_id, description, result}, ...]
    disposition: Optional[str]          # "home" | "urgent" | "er" | None
    missing_required: list              # field names safety still needs
    medication_decision: Optional[dict] # {allowed, med, reason}

    # -- Explanation outputs ---------------------------------------------------
    explanation: Optional[str]          # caregiver-facing text
    key_positives: list                 # symptoms present
    key_negatives: list                 # important negatives
    go_now_thresholds: list             # explicit escalation triggers
    overnight_plan: list                # home care steps

    # -- Control flow ----------------------------------------------------------
    is_complete: bool
    phase: str                          # "intake" | "triage" | "plan_ready"


# All clinical fields that extraction can write
CLINICAL_FIELDS = [
    "age_months", "temperature_f", "fever", "vomiting", "alert",
    "drinking", "urine_hours", "breathing_issue", "medications",
    "seizure", "rash", "local_context",
]

# Fields the safety agent requires before issuing a home disposition.
# ER red flags can short-circuit without all fields present.
REQUIRED_FIELDS_FOR_HOME = [
    "alert", "breathing_issue", "temperature_f", "drinking",
    "urine_hours", "vomiting",
]

# Priority-ordered questions for missing fields
FIELD_QUESTIONS = {
    "alert": "Is your child awake and responding normally when you talk to them?",
    "breathing_issue": "Is your child having any trouble breathing or breathing fast?",
    "temperature_f": "What is the temperature right now?",
    "urine_hours": "About how long has it been since your child last urinated?",
    "drinking": "Is your child drinking any fluids?",
    "vomiting": "Has your child been vomiting? If so, how many times?",
    "medications": "Is your child currently taking any medications?",
    "seizure": "Has your child had any seizures?",
    "age_months": "How old is your child?",
}


def initial_state() -> ClinicalState:
    """Return a blank state for a new triage case."""
    return ClinicalState(
        messages=[],
        turn=0,
        # Clinical facts
        age_months=None,
        temperature_f=None,
        fever=None,
        vomiting=None,
        alert=None,
        drinking=None,
        urine_hours=None,
        breathing_issue=None,
        medications=None,
        seizure=None,
        rash=None,
        local_context=None,
        # KG
        snomed_concepts=[],
        # Safety
        rules_fired=[],
        disposition=None,
        missing_required=[],
        medication_decision=None,
        # Explanation
        explanation=None,
        key_positives=[],
        key_negatives=[],
        go_now_thresholds=[],
        overnight_plan=[],
        # Control
        is_complete=False,
        phase="intake",
    )
