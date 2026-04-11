"""Interpretation agent — LLM structured extraction → Alex's fact vocabulary.

Ported from Yoko's interpretation_agent.ipynb. Key changes from the notebook:

1. Writes extracted symbolic state into `state["facts"]` (the nested dict
   Alex's rules read), not as flat keys on the state.
2. Translates Yoko's binary fields into Alex's categorical vocabulary:
     - alert:      yes/no           → normal/reduced
     - drinking:   yes/some/no      → intake:    normal/reduced/none
     - urination:  yes/no           → urination: normal/none
     - vomiting:   none/once/repeated (drops 'none' — absence isn't written)
     - breathing:  yes/no issues    → breathing: difficulty/normal
3. Emits `state["raw_symptoms"]` — a list of canonical free-text phrases
   derived from the facts — that the KG adapter grounds against.
4. Computes `missing_required` against REQUIRED_FACTS_FOR_HOME.
"""

import os
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from caretrace.state import (
    ClinicalState,
    FACT_KEYS,
    REQUIRED_FACTS_FOR_HOME,
    FACT_QUESTIONS,
)


# ── Pydantic extraction schema ────────────────────────────────────────────────

class ExtractionResult(BaseModel):
    """Structured clinical facts extracted from a single caregiver message.

    This keeps Yoko's raw vocabulary (yes/no, etc.) because it's what the
    LLM is most reliable at. The agent translates to Alex's vocabulary
    after extraction.
    """
    # Raw numeric / typed fields
    age_months:           Optional[int]   = Field(None, description="Child age in months (convert years → months).")
    temperature_f:        Optional[float] = Field(None, description="Temperature in °F (convert °C if needed).")
    fever_duration_days:  Optional[float] = Field(None, description="How many days the fever has been present.")
    current_medication:   Optional[str]   = Field(None, description="Name of medication the child is on, or 'none'.")
    medication_last_dose: Optional[str]   = Field(None, description="When the last medication dose was given.")
    weight_kg:            Optional[float] = Field(None, description="Child weight in kilograms if mentioned.")

    # Symbolic fields — Yoko's vocabulary, translated later
    fever:           Optional[str] = Field(None, description="'yes' or 'no' — is there a fever?")
    vomiting:        Optional[str] = Field(None, description="'none', 'once', or 'repeated'.")
    alert:           Optional[str] = Field(None, description="'yes' (normal) or 'no' (lethargic/reduced responsiveness).")
    drinking:        Optional[str] = Field(None, description="'yes' (normal intake), 'some' (reduced), or 'no' (refusing).")
    urination_8h:    Optional[str] = Field(None, description="'yes' or 'no' — has the child urinated in the last 8h?")
    breathing_issues:Optional[str] = Field(None, description="'yes' or 'no' — is the child having breathing trouble?")
    seizure:         Optional[str] = Field(None, description="'yes' or 'no' — any seizure activity?")
    rash:            Optional[str] = Field(None, description="'yes' or 'no' — any new rash?")


EXTRACTION_SYSTEM_PROMPT = """\
You are a clinical information extractor for a pediatric triage system.

YOUR ONLY JOB is to extract structured clinical facts from the caregiver's message.

STRICT RULES:
1. Extract ONLY what the caregiver explicitly states or clearly implies.
2. Set a field to null if the message provides NO new information about it.
3. Do NOT infer, guess, or fill in fields that were not mentioned.
4. Do NOT generate clinical advice, diagnosis, treatment, or disposition — ever.
5. Age conversion: if given in years, convert to months (e.g. "6 years" → 72).
6. Temperature conversion: if given in Celsius, convert to Fahrenheit.
7. Vomiting: 'none' = no vomiting, 'once' = one episode, 'repeated' = more than once.
8. For alertness: "alert", "talking", "responsive" → 'yes'; "lethargic", "out of it",
   "barely responding", "hard to wake" → 'no'.
9. For drinking: "drinking normally" → 'yes'; "sipping", "a little", "not much" → 'some';
   "won't drink", "refusing fluids" → 'no'.
"""


# ── Yoko → Alex vocabulary translators ────────────────────────────────────────

def _translate_to_facts(extracted: ExtractionResult) -> dict:
    """Translate Yoko's raw vocabulary into Alex's fact vocabulary.

    Returns only keys the extractor actually found — callers merge into
    state["facts"] so missing keys don't overwrite existing knowledge.
    """
    facts: dict = {}

    if extracted.fever is not None:
        facts["fever"] = extracted.fever

    if extracted.alert is not None:
        facts["alert"] = "normal" if extracted.alert == "yes" else "reduced"

    if extracted.drinking is not None:
        facts["intake"] = {"yes": "normal", "some": "reduced", "no": "none"}.get(
            extracted.drinking
        )

    if extracted.urination_8h is not None:
        facts["urination"] = "normal" if extracted.urination_8h == "yes" else "none"

    if extracted.vomiting in ("once", "repeated"):
        facts["vomiting"] = extracted.vomiting
    # vomiting == "none" intentionally not written — absence is not a fact we track

    if extracted.breathing_issues is not None:
        facts["breathing"] = (
            "difficulty" if extracted.breathing_issues == "yes" else "normal"
        )

    if extracted.seizure is not None:
        facts["seizure"] = extracted.seizure

    if extracted.rash is not None:
        facts["rash"] = extracted.rash

    # Filter out any None values (e.g. unknown drinking key)
    return {k: v for k, v in facts.items() if v is not None}


def _derive_raw_symptoms(facts: dict, temperature_f: Optional[float]) -> list[str]:
    """Build canonical symptom phrases from the current fact set.

    These are the strings the KG adapter will try to ground. We keep them
    deterministic (one phrase per fact signal) so both backends behave the
    same way.
    """
    symptoms: list[str] = []

    if facts.get("fever") == "yes":
        if temperature_f is not None and temperature_f >= 104.0:
            symptoms.append("high fever")
        else:
            symptoms.append("fever")

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


# ── LLM plumbing ──────────────────────────────────────────────────────────────

def _get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _next_missing_fact(merged_facts: dict) -> Optional[str]:
    """Return the highest-priority required fact still unknown, or None."""
    for key in REQUIRED_FACTS_FOR_HOME:
        if key not in merged_facts:
            return key
    return None


# ── LangGraph node ────────────────────────────────────────────────────────────

def interpret(state: ClinicalState) -> dict:
    """Extract facts from the latest caregiver message and merge into state."""
    messages = state.get("messages", [])
    if not messages:
        return {}

    # Pick the latest HumanMessage
    latest = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage) or (hasattr(m, "type") and m.type == "human"):
            latest = m
            break
    if latest is None:
        return {}

    # Build conversation history (excluding latest) so the extractor has context
    history = "\n".join(
        f"{'Caregiver' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in messages[:-1]
    )

    prompt = [SystemMessage(content=EXTRACTION_SYSTEM_PROMPT)]
    if history:
        prompt.append(SystemMessage(content=f"Conversation so far:\n{history}"))
    prompt.append(HumanMessage(content=f"Latest caregiver message: {latest.content}"))

    llm = _get_llm()
    extractor = llm.with_structured_output(ExtractionResult)
    extracted: ExtractionResult = extractor.invoke(prompt)

    # ── Merge raw fields (non-None only, never overwrite with None) ──────────
    updates: dict = {}
    for field in ("age_months", "temperature_f", "fever_duration_days",
                  "current_medication", "medication_last_dose", "weight_kg"):
        new_val = getattr(extracted, field, None)
        if new_val is not None:
            updates[field] = new_val

    # ── Translate Yoko vocab → Alex facts, merge into facts dict ─────────────
    existing_facts: dict = dict(state.get("facts") or {})
    new_facts = _translate_to_facts(extracted)
    existing_facts.update(new_facts)
    updates["facts"] = existing_facts

    # ── Derive raw symptoms list for the KG to ground against ───────────────
    temp = updates.get("temperature_f", state.get("temperature_f"))
    updates["raw_symptoms"] = _derive_raw_symptoms(existing_facts, temp)

    # ── Missing-fact tracking + follow-up question ──────────────────────────
    next_missing = _next_missing_fact(existing_facts)
    updates["missing_required"] = [
        k for k in REQUIRED_FACTS_FOR_HOME if k not in existing_facts
    ]
    updates["follow_up_question"] = FACT_QUESTIONS.get(next_missing) if next_missing else None

    updates["turn"] = state.get("turn", 0) + 1
    return updates
