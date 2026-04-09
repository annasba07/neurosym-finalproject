"""Explanation agent — verbalizes the structured triage decision for caregivers.

The LLM receives the full decision object (disposition, rules fired,
positives/negatives, thresholds, care plan) and turns it into natural,
empathetic caregiver-facing language. It does NOT make clinical decisions.
"""

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from caretrace.state import ClinicalState, FIELD_QUESTIONS


EXPLAIN_SYSTEM_PROMPT = """\
You are a caring pediatric nurse providing triage guidance to a worried parent.

You will receive a structured clinical decision. Your job is to turn it into
clear, empathetic, actionable guidance. You must:

1. State the recommendation clearly (go to ER now / see doctor urgently / safe to monitor at home)
2. Explain WHY using the key findings provided — be specific
3. List what to watch for (go-now thresholds)
4. If home management: give the overnight care plan
5. If medication info is available: include it

Rules:
- NEVER contradict the disposition — if the system says ER, you say ER
- Use simple language — the caregiver is stressed and may not be medically trained
- Be warm but direct — don't hedge on safety-critical advice
- Keep it concise — this is a triage summary, not a medical textbook
- End with reassurance appropriate to the situation
"""

FOLLOWUP_SYSTEM_PROMPT = """\
You are a caring pediatric triage nurse collecting information from a worried parent.

The system still needs some information to complete the assessment. Ask for the
missing information naturally and warmly. You may combine 1-2 questions if
they flow naturally together, but don't overwhelm with too many questions at once.

Prioritize safety-critical questions first (alertness, breathing).
"""


def _get_llm():
    """Get the LLM for explanation generation."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def explain(state: ClinicalState) -> dict:
    """LangGraph node: generate caregiver-facing explanation of the decision."""
    disposition = state.get("disposition")
    if disposition is None:
        return {}

    # Build the decision summary for the LLM
    decision_summary = _build_decision_summary(state)

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=EXPLAIN_SYSTEM_PROMPT),
        HumanMessage(content=decision_summary),
    ])

    return {
        "explanation": response.content,
        "messages": [AIMessage(content=response.content)],
        "is_complete": True,
    }


def ask_followup(state: ClinicalState) -> dict:
    """LangGraph node: ask for missing information when disposition can't be determined."""
    missing = state.get("missing_required", [])
    if not missing:
        return {}

    # Pick the highest-priority missing fields (max 2)
    questions_to_ask = []
    for field in missing[:2]:
        q = FIELD_QUESTIONS.get(field)
        if q:
            questions_to_ask.append(q)

    if not questions_to_ask:
        return {}

    # Build context about what we already know
    known_summary = _build_known_summary(state)

    llm = _get_llm()
    prompt = (
        f"What we know so far:\n{known_summary}\n\n"
        f"We still need to ask:\n"
        + "\n".join(f"- {q}" for q in questions_to_ask)
        + "\n\nGenerate a warm, natural follow-up message combining these questions."
    )

    response = llm.invoke([
        SystemMessage(content=FOLLOWUP_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {
        "messages": [AIMessage(content=response.content)],
    }


def _build_decision_summary(state: ClinicalState) -> str:
    """Build a structured decision summary for the explanation LLM."""
    lines = []
    lines.append(f"DISPOSITION: {state.get('disposition', 'unknown').upper()}")
    lines.append("")

    # Key findings
    positives = state.get("key_positives", [])
    negatives = state.get("key_negatives", [])
    if positives:
        lines.append("KEY CONCERNS:")
        for p in positives:
            lines.append(f"  - {p}")
    if negatives:
        lines.append("REASSURING FINDINGS:")
        for n in negatives:
            lines.append(f"  - {n}")

    # Rules fired
    rules = state.get("rules_fired", [])
    if rules:
        lines.append("")
        lines.append("CLINICAL REASONING:")
        for r in rules:
            lines.append(f"  - [{r['category']}] {r['reason']}")

    # Medication decision
    med = state.get("medication_decision")
    if med and med.get("allowed"):
        lines.append("")
        lines.append(f"MEDICATION: {med.get('reason', '')}")

    # Go-now thresholds
    thresholds = state.get("go_now_thresholds", [])
    if thresholds:
        lines.append("")
        lines.append("GO TO ER IMMEDIATELY IF:")
        for t in thresholds:
            lines.append(f"  - {t}")

    # Overnight plan
    plan = state.get("overnight_plan", [])
    if plan:
        lines.append("")
        lines.append("OVERNIGHT CARE PLAN:")
        for step in plan:
            lines.append(f"  - {step}")

    return "\n".join(lines)


def _build_known_summary(state: ClinicalState) -> str:
    """Summarize what we know so far for follow-up context."""
    facts = []
    if state.get("age_months") is not None:
        age = state["age_months"]
        if age >= 24:
            facts.append(f"Age: {age // 12} years old")
        else:
            facts.append(f"Age: {age} months old")
    if state.get("temperature_f") is not None:
        facts.append(f"Temperature: {state['temperature_f']}°F")
    if state.get("fever"):
        facts.append(f"Fever: {state['fever']}")
    if state.get("vomiting"):
        facts.append(f"Vomiting: {state['vomiting']}")
    if state.get("medications"):
        facts.append(f"Medications: {state['medications']}")

    return "\n".join(facts) if facts else "Limited information so far."
