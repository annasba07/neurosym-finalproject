"""Interpretation agent — LLM-based structured extraction of clinical facts.

Takes the caregiver's free-text message and extracts structured clinical
fields using Pydantic + structured output. Only writes fields the LLM
is confident about; leaves unknowns as None.
"""

import os
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from caretrace.state import ClinicalState, CLINICAL_FIELDS


# ---- Pydantic schema for structured extraction ----

class ExtractionResult(BaseModel):
    """Structured clinical facts extracted from caregiver message."""

    age_months: Optional[int] = Field(
        None,
        description="Child's age in months. Convert years to months (e.g. 6 years = 72 months).",
    )
    temperature_f: Optional[float] = Field(
        None,
        description="Temperature in Fahrenheit. Convert from Celsius if needed.",
    )
    fever: Optional[str] = Field(
        None,
        description="Does the child have a fever? 'yes' or 'no'.",
    )
    vomiting: Optional[str] = Field(
        None,
        description="Has the child vomited? 'none', 'once', or 'repeated'.",
    )
    alert: Optional[str] = Field(
        None,
        description="Is the child alert and responsive? 'yes' or 'no'.",
    )
    drinking: Optional[str] = Field(
        None,
        description="Is the child drinking fluids? 'yes', 'some', or 'no'.",
    )
    urine_hours: Optional[int] = Field(
        None,
        description="Hours since last urination. Estimate from context if possible.",
    )
    breathing_issue: Optional[str] = Field(
        None,
        description="Does the child have breathing difficulty? 'yes' or 'no'.",
    )
    medications: Optional[str] = Field(
        None,
        description="Current medications the child is taking, or 'none'.",
    )
    seizure: Optional[str] = Field(
        None,
        description="Has the child had a seizure? 'yes' or 'no'.",
    )
    rash: Optional[str] = Field(
        None,
        description="Does the child have a rash? 'yes' or 'no'.",
    )
    local_context: Optional[str] = Field(
        None,
        description="Any local illness context (e.g. 'stomach virus going around school').",
    )


EXTRACTION_SYSTEM_PROMPT = """\
You are a clinical fact extractor for a pediatric triage system.

Given a caregiver's message about their sick child, extract ONLY the clinical
facts that are explicitly stated or strongly implied. Do NOT guess or infer
facts that are not supported by the text.

Rules:
- Convert age to months (e.g. "6 years old" → 72)
- Convert temperature to Fahrenheit if given in Celsius
- For vomiting: "threw up once" → "once", "keeps throwing up" → "repeated"
- For drinking: "sipping a little" → "some", "won't drink anything" → "no"
- For alertness: "barely responding" or "lethargic" → "no"
- Leave fields as null if the caregiver hasn't mentioned them
- Extract medication names exactly as stated
"""


def _get_llm():
    """Get the LLM for extraction. Uses Groq by default."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def interpret(state: ClinicalState) -> dict:
    """LangGraph node: extract clinical facts from the latest user message.

    Reads the most recent HumanMessage, runs structured extraction,
    and merges new facts into state (without overwriting existing known facts).
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    # Find the latest human message
    latest_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (
            hasattr(msg, "type") and msg.type == "human"
        ):
            latest_msg = msg
            break

    if latest_msg is None:
        return {}

    # Build context from previously known facts
    known_facts = []
    for field in CLINICAL_FIELDS:
        val = state.get(field)
        if val is not None:
            known_facts.append(f"  {field}: {val}")

    context_str = ""
    if known_facts:
        context_str = (
            "\n\nAlready known facts (update if the new message provides corrections):\n"
            + "\n".join(known_facts)
        )

    llm = _get_llm()
    structured_llm = llm.with_structured_output(ExtractionResult)

    result = structured_llm.invoke([
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT + context_str),
        HumanMessage(content=latest_msg.content),
    ])

    # Merge: only update fields that are currently None or that the LLM updated
    updates = {}
    for field in CLINICAL_FIELDS:
        new_val = getattr(result, field, None)
        if new_val is not None:
            # New extraction overrides (caregiver might correct earlier info)
            updates[field] = new_val

    updates["turn"] = state.get("turn", 0) + 1
    return updates
